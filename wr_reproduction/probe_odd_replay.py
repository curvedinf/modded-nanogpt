"""Bounded first-Adam replay diagnostic, never a training/checkpoint facility.

The sparse table is restored only at the exact union of rows the first event
can write. All ordinary model/optimizer/rank/sampler storage is copied in place.
Two no-probe replays distinguish ordinary replay variation from probe effects.
"""
import dataclasses
import random
import time
import numpy as np
import torch


class InPlaceSnapshot:
    """Host copies of explicitly selected state; preserve tensor/container identity."""
    def __init__(self):
        self.tensors = []; self.arrays = []; self.containers = []; self.attributes = []
        self.seen = set(); self.tensor_keys = set()
        self.rng = (torch.get_rng_state(), torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None,
                    random.getstate(), np.random.get_state())

    @torch.no_grad()
    def capture(self, value, path):
        if isinstance(value, torch.Tensor):
            key = (value.device, value.data_ptr(), tuple(value.shape), tuple(value.stride()), value.dtype)
            if key not in self.tensor_keys:
                self.tensor_keys.add(key)
                self.tensors.append((path, value, value.detach().to('cpu', copy=True), value.data_ptr()))
            return
        if id(value) in self.seen: return
        self.seen.add(id(value))
        if isinstance(value, np.ndarray): self.arrays.append((path, value, value.copy()))
        elif isinstance(value, dict):
            self.containers.append((value, value.copy()))
            for key, item in value.items(): self.capture(item, path+'/'+str(getattr(key, 'label', key)))
        elif isinstance(value, list):
            self.containers.append((value, value.copy()))
            for i, item in enumerate(value): self.capture(item, path+'/'+str(i))
        elif isinstance(value, tuple):
            for i, item in enumerate(value): self.capture(item, path+'/'+str(i))
        elif isinstance(value, set): self.containers.append((value, value.copy()))
        elif dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                item = getattr(value, field.name)
                if not value.__dataclass_params__.frozen:
                    self.attributes.append((value, field.name, item))
                self.capture(item, path+'/'+field.name)

    def object(self, value, path, exclude=()):
        # Snapshot direct bindings as well as mutable contents. Opaque objects
        # (CUDA events/graphs, executors, callables) retain their identity.
        for key, item in vars(value).items():
            if key not in exclude:
                self.attributes.append((value, key, item)); self.capture(item, path+'/'+key)

    @torch.no_grad()
    def restore(self):
        for obj, key, value in self.attributes: setattr(obj, key, value)
        for target, saved in self.containers:
            if isinstance(target, list): target[:] = saved
            else: target.clear(); target.update(saved)
        for _, target, saved in self.arrays: target[...] = saved
        for path, target, saved, pointer in self.tensors:
            assert target.data_ptr() == pointer, ('storage_rebound', path)
            target.copy_(saved)
        torch.set_rng_state(self.rng[0])
        if self.rng[1] is not None: torch.cuda.set_rng_state(self.rng[1])
        random.setstate(self.rng[2]); np.random.set_state(self.rng[3])

    @torch.no_grad()
    def verify(self):
        changed = []
        for path, target, saved, pointer in self.tensors:
            live = target.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            wanted = saved.contiguous().reshape(-1).view(torch.uint8)
            if target.data_ptr() != pointer or not torch.equal(live, wanted): changed.append(path)
        for path, target, saved in self.arrays:
            if not np.array_equal(target.view(np.uint8), saved.view(np.uint8)): changed.append(path)
        return dict(passed=not changed, checked_tensors=len(self.tensors), checked_arrays=len(self.arrays), changed=changed,
                    saved_tensor_bytes=sum(t.numel()*t.element_size() for _, t, _, _ in self.tensors))


class FixedBatches:
    def __init__(self, batches): self.batches = batches
    def peek(self, step): return self.batches[step]
    def take(self, step): return self.batches[step]


def ready(r):
    for sampler in r.samplers:
        if sampler.staging.pending: sampler.staging.pending[1].result()
    torch.cuda.synchronize()


@torch.no_grad()
def snapshot(r):
    ready(r); s = InPlaceSnapshot()
    s.object(r, 'runner')
    for name, module in r.model.named_modules(): s.object(module, 'model/'+name)
    for i, yarn in enumerate(r.model.yarns): s.object(yarn, 'yarn/'+str(i))
    for name, obj in [('tm', r.tm), ('optimizer', r.tm.optimizer), ('rank', r.rank_states),
                      ('gradients', r.gradients), ('replicated_adam', r.tm.optimizer.replicated),
                      ('bank_scalars', r.tm.optimizer.bank_scalars)]: s.object(obj, name)
    s.object(r.tables, 'tables', exclude=('table',))
    for rank, sampler in enumerate(r.samplers):
        for name, obj in [('sampler', sampler), ('builder', sampler.builder), ('staging', sampler.staging)]:
            s.object(obj, f'{name}/{rank}')
    s.capture(r.sampled_head.COUNTER, 'sampled_head_counter')
    s.capture(r.graphs.adam_accumulators, 'graph/adam_accumulators')
    # Only the first-stage forward/backward graphs are used by this diagnostic.
    for step in (1, 2):
        s.capture(r.graphs.captured[r.graphs.step_keys[step]], 'graph/step'+str(step))
    for name, value in r.graphs.baked_tensors(): s.capture(value, 'graph/'+name)
    return s


@torch.no_grad()
def tensor_outputs(r, touched):
    out = {f'parameter/{n}': p.detach().cpu().clone() for n, p in r.model.named_parameters()}
    for p, state in r.tm.optimizer.param_states.items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor): out['optimizer/'+p.label+'/'+key] = value.detach().cpu().clone()
    for name, bank in r.tm.optimizer.banks.items():
        for key, value in bank.mutated().items():
            if key != 'p_slice': out['anvil/'+name+'/'+key] = value.detach().cpu().clone()
    out['sparse/touched_rows'] = r.tables.table[touched].cpu()
    out['sparse/touched_moment'] = r.tables.moment[touched].cpu()
    out['sparse/touched_last'] = r.tables.last[touched].cpu()
    return out


def compare_outputs(left, right):
    assert left.keys() == right.keys()
    rows = []
    for key, a in left.items():
        b = right[key]
        exact = torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))
        if exact: continue
        # Chunk the calculation to bound temporary host memory for big banks.
        aa = a.reshape(-1); bb = b.reshape(-1); ds = bs = 0.; maximum = 0.; changed = 0
        for offset in range(0, aa.numel(), 1048576):
            x = aa[offset:offset+1048576].double(); y = bb[offset:offset+1048576].double(); d = x-y
            ds += d.square().sum().item(); bs += y.square().sum().item()
            maximum = max(maximum, d.abs().max().item()); changed += (x != y).sum().item()
        rows.append(dict(component=key, changed_elements=changed, total_elements=a.numel(), max_abs=maximum,
                         relative_rms=(ds/max(bs, 1e-30))**.5))
    return dict(all_bitwise_equal=not rows, checked_tensors=len(left), changed=rows)


def diagnostic(r):
    from track_1_short.model.prefix_prediction import build_prefix_table_bucket
    from wr_reproduction.probe_evaluation import FixedProbe
    from wr_reproduction.probe_calibration import state, fingerprint
    from wr_reproduction.probe_precision_calibration import audit
    from wr_reproduction.probe_study import atomic, stopped
    from wr_reproduction.virtual_tables import Event
    from wr_reproduction.matched_architecture import audit_model
    from unittest.mock import patch
    assert audit_model(r.model, 'ks3_experts')['active_conventional_mlp_instances'] == 5
    cfg = r.research_config; m = r.model; m.limit_yarn_rebuild(r.max_tokens)
    m.prefix_table.copy_(build_prefix_table_bucket(m.vocab_size, 0, 1))
    for sampler in r.samplers: sampler.set_prefix_table(m.prefix_table.cpu().numpy())
    batches = r.loaders([0, 1, 2]); start = time.perf_counter()
    def check_bound():
        if stopped(): raise RuntimeError('Explicit study STOP')
        if time.perf_counter()-start > 360.: raise RuntimeError('First-event diagnostic exceeded360s after warmup')
    loss0 = r.step(0, batches, next_step=1, prefetch=True); ready(r)
    r.log(event='odd_replay_initial_update', step=1, loss=loss0.item(), not_a_screening_reference=True)
    fixed = [FixedBatches({step:b.peek(step) for step in (1, 2)}) for b in batches]
    ready(r)
    assert r.tables.event == 0 and r.tables.event_policy(1) and r.tm.is_adam_step(1)
    assert all(r.tables.pending) and all(r.gradients.pending)
    touched = torch.unique(torch.cat([ids for entries in r.tables.pending for ids, _ in entries]
                                    +[b.peek(1).ngram_ids for b in fixed]), sorted=True).long()
    table_pointer = r.tables.table.data_ptr(); sparse_original = r.tables.table[touched].cpu()
    sparse_original_hash = fingerprint(sparse_original)
    saved = snapshot(r); initial_state = state(r)
    batches_hash = [{str(step):{name:fingerprint(getattr(b.peek(step), name))
        for name in ('inputs','targets','cum_seqlens','ngram_ids','targets_cpu')}
        for step in (1, 2)} for b in fixed]
    original_optimizer = r.tm.step_optimizers; original_launch = Event.launch_ngram
    traces = []; branches = []; outputs = {}; active = None
    def traced_optimizer(step, tables, labels):
        ready(r)
        trace = dict(branch=active, step=step,
            gradients={p.label:fingerprint(p.grad) for p in r.tm.optimizer.param_cfgs if p.grad is not None},
            pending_sparse=[[(fingerprint(ids), fingerprint(g)) for ids,g in entries] for entries in r.tables.pending],
            sampled_head_counters=[x.item() for x in r.counter_states],
            sampler_offsets=[s.builder.sweep_offset for s in r.samplers])
        traces.append(trace)
        return original_optimizer(step, tables, labels)
    def traced_launch(event):
        # Assert the entire write domain, not a random sparse-table sample.
        domain = torch.unique(torch.cat([ids for entries in event.t.pending for ids,_ in entries]), sorted=True).long()
        assert torch.equal(domain, touched), 'Sparse write domain escaped saved rows'
        original_launch(event); ready(r)
        traces[-1]['sparse_merged_payload'] = [(fingerprint(ids), fingerprint(g)) for ids,g in event.payload]
    try:
        r.tm.step_optimizers = traced_optimizer
        with patch.object(Event, 'launch_ngram', traced_launch):
            for active in ('no_probe_a', 'no_probe_b', 'with_probe'):
                check_bound()
                if active != 'no_probe_a':
                    ready(r); saved.restore(); r.tm.step_optimizers = traced_optimizer
                    assert r.tables.table.data_ptr() == table_pointer
                    r.tables.table.index_copy_(0, touched, sparse_original.to(r.tables.table.device))
                    r.model.zero_grad(set_to_none=True); ready(r)
                restoration = saved.verify(); assert restoration['passed'], restoration
                assert fingerprint(r.tables.table[touched]) == sparse_original_hash
                current = state(r)
                # In-place replay restoration increments PyTorch's version counter.
                # Contents/pointers are audited; a version cannot be decremented.
                current['sparse/table_contract']['version'] = initial_state['sparse/table_contract']['version']
                rewind_audit = audit(initial_state, current)
                probe_row = None; probe_audit = None
                if active == 'with_probe':
                    probe = FixedProbe(r, cfg['probe_protocol'], cfg['backend_hash'])
                    before = state(r); probe_row = probe.observe(1, time.perf_counter())
                    probe_audit = audit(before, state(r))
                    probe_row = {k:v for k,v in probe_row.items() if k not in ('chunk_losses','ks3_activation_diagnostics','probe_signature')}
                    probe_row.update(not_a_screening_reference=True, diagnostic_replay=True)
                r.log(event='odd_replay_branch_begin', branch=active, not_a_screening_reference=True)
                odd = r.step(1, fixed, next_step=2, prefetch=True); ready(r)
                assert r.tables.event == 1 and not any(r.tables.pending) and not any(r.gradients.pending)
                odd_outputs = tensor_outputs(r, touched)
                even = r.step(2, fixed, prefetch=False); ready(r)
                even_outputs = tensor_outputs(r, touched)
                branch = dict(branch=active, odd_objective=odd.item(), next_even_objective=even.item(),
                    rewind_state_audit=rewind_audit, snapshot_restoration=restoration,
                    probe_state_audit=probe_audit, probe=probe_row, sparse_event_count=r.tables.event)
                assert torch.isfinite(odd).item() and torch.isfinite(even).item()
                if active == 'no_probe_a': outputs = dict(odd=odd_outputs, even=even_outputs)
                else:
                    branch['odd_vs_no_probe_a'] = compare_outputs(outputs['odd'], odd_outputs)
                    branch['even_vs_no_probe_a'] = compare_outputs(outputs['even'], even_outputs)
                branches.append(branch); r.log(event='odd_replay_branch_complete', **branch)
                del odd_outputs, even_outputs
                check_bound()
    finally: r.tm.step_optimizers = original_optimizer
    result = dict(passed=True, diagnostic='first-adam-and-sparse-event-replay-v1',
        original_logical_ranks=8, full_graph_warmup=True, first_update_objective=loss0.item(),
        exact_sparse_write_domain_rows=touched.numel(), exact_sparse_write_domain_sha256=fingerprint(touched),
        sparse_table_pointer_preserved=r.tables.table.data_ptr()==table_pointer,
        batch_fingerprints=batches_hash, branches=branches, component_traces=traces,
        diagnostic_seconds=time.perf_counter()-start, not_a_screening_reference=True,
        continuous_from_initialization=False, full_convergence_verified=False, reserved_text_used=False,
        limitations=['Three replays at one initial checkpoint are not a training-seed variance estimate.',
            'No-probe repeat differences measure replay nondeterminism; with-probe differences must be interpreted against that baseline.',
            'Pre-optimizer tracing synchronizes and hashes state; diagnostic runtime is not normal training throughput.',
            'The exact first-event sparse write domain is snapshotted; the entire130GB table is not copied.',
            'Shared math, kernels, optimizer and screening protocol remain unchanged.'])
    atomic(r.research_run/'calibration.json', result); r.log(event='calibration_complete', **result)
    return result


def cpu_check():
    from types import SimpleNamespace
    @dataclasses.dataclass
    class SmallState:
        step: int
        moment: torch.Tensor
    tensor = torch.arange(12., dtype=torch.float32).reshape(3,4)
    pending = [{'x':tensor[:, :2]}]; array = np.arange(5, dtype=np.int64)
    root = SimpleNamespace(table=tensor, pending=pending, state=SmallState(1,tensor[:,2:]),
                           scratch=array, active={1,3}, next_slot=2, uploaded=None)
    snap = InPlaceSnapshot(); snap.object(root, 'test'); pointer=tensor.data_ptr()
    expected_random=(torch.rand(3),random.random(),np.random.rand(3))
    tensor.add_(20); pending[0].clear(); pending.clear(); array[:]=7
    root.state.step=4; root.active.clear(); root.next_slot=7; root.uploaded=('new',)
    snap.restore(); report=snap.verify()
    assert report['passed'] and tensor.data_ptr()==pointer and torch.equal(tensor,torch.arange(12.).reshape(3,4))
    assert root.pending is pending and root.pending[0]['x'].data_ptr()==pointer
    assert root.state.step==1 and root.active=={1,3} and root.next_slot==2 and root.uploaded is None
    assert torch.equal(torch.rand(3),expected_random[0]) and random.random()==expected_random[1]
    assert np.array_equal(np.random.rand(3),expected_random[2])
    exact=compare_outputs({'x':tensor},{'x':tensor.clone()}); assert exact['all_bitwise_equal']
    altered=tensor.clone();altered[0,0]=1
    different=compare_outputs({'x':tensor},{'x':altered});assert different['changed'][0]['changed_elements']==1
    return dict(passed=True,scope='nested pending state, aliases/storage, Adam scalar, numpy scratch, rings, RNG and exact/different comparisons',**{k:v for k,v in report.items() if k!='passed'})


if __name__ == '__main__':
    import json
    print(json.dumps(cpu_check(),indent=2))
