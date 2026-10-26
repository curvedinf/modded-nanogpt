"""Keep unused dense FP8 training caches read-only during BF16 evaluation.

The original dense refresh first packs the separate BF16 evaluation weights,
then also refreshes FP8 training weights unconditionally. Only the latter call
is suppressed in eval mode. Training calls delegate unchanged to the original
producer, including exact/delayed scale selection and graph-owned storage.
"""
from wr_reproduction import ks3_dense_fp8 as dense

ORIGINAL_REFRESH = dense.refresh_weights
EXECUTED_MODES = set()


def refresh_weights(model, exact):
    if not model.training:
        EXECUTED_MODES.add('bf16_evaluation_skips_unused_fp8_refresh')
        return None
    EXECUTED_MODES.add('original_training_fp8_refresh')
    return ORIGINAL_REFRESH(model, exact)


def install():
    if dense.refresh_weights not in (ORIGINAL_REFRESH, refresh_weights):
        raise RuntimeError('Unexpected dense FP8 weight-refresh overlay')
    dense.refresh_weights = refresh_weights


def cpu_check():
    from types import SimpleNamespace
    from unittest.mock import patch
    calls=[];sentinel=object()
    def original(model,exact):
        calls.append((model,exact));return sentinel
    model=SimpleNamespace(training=True)
    with patch(__name__+'.ORIGINAL_REFRESH',original):
        for exact in (True,False):
            assert refresh_weights(model,exact) is sentinel
            assert calls[-1]==(model,exact)
        model.training=False
        for exact in (True,False):assert refresh_weights(model,exact) is None
        assert len(calls)==2
    return dict(passed=True,training_delegation_args_and_return_unchanged=True,
                evaluation_does_not_call_unused_producer=True)


def gpu_check():
    import torch
    from wr_reproduction.backend import install as backend
    backend('pr393')
    from wr_reproduction.check_ks3_feedforward_fp8 import model
    torch.manual_seed(3269);torch.set_num_threads(1)
    m=model(2816);m.retained_slots=(0,1,2,3)
    for _,b in m.named_buffers():
        if 'float8' in str(b.dtype):b.zero_()
    pointers={n:b.data_ptr() for n,b in m.named_buffers()}
    def snapshot():return {n:b.detach().clone() for n,b in m.named_buffers()}
    def equal(a,b):return torch.equal(a.reshape(-1).view(torch.uint8),b.reshape(-1).view(torch.uint8))
    def changed(saved):return [n for n,b in m.named_buffers() if not equal(b,saved[n])]
    @torch.no_grad()
    def restore(saved):
        for n,b in m.named_buffers():b.copy_(saved[n])
    checks=[]
    for exact in (True,False,True):
        with torch.no_grad():m.mlp_bank.add_(torch.randn_like(m.mlp_bank)*.001)
        before=snapshot();ORIGINAL_REFRESH(m,exact);expected=snapshot()
        restore(before);refresh_weights(m,exact)
        assert not changed(expected)
        assert pointers=={n:b.data_ptr() for n,b in m.named_buffers()}
        checks.append(dict(exact=exact,all_original_training_cache_bytes_equal=True,
                           pointers_preserved=True))
    # Arrange a pending training refresh. Eval must not consume its old amax
    # or update original/native packed operands even when raw weights differ.
    with torch.no_grad():
        m.mlp_bank.add_(torch.randn_like(m.mlp_bank)*.001)
        m.ks3_dense_weight_amax.fill_(.25)
    m.eval();before=snapshot()
    refresh_weights(m,False);assert not changed(before)
    ORIGINAL_REFRESH(m,False);unguarded_changes=changed(before)
    assert 'ks3_dense_weight_scales' in unguarded_changes and 'ks3_dense_weight_amax' in unguarded_changes
    assert any('native' in n for n in unguarded_changes)
    restore(before);install();install();dense.refresh_weights(m,False)
    assert not changed(before) and pointers=={n:b.data_ptr() for n,b in m.named_buffers()}
    m.train();dense.refresh_weights(m,False)
    assert changed(before)
    assert EXECUTED_MODES=={'bf16_evaluation_skips_unused_fp8_refresh','original_training_fp8_refresh'}
    return dict(passed=True,features=2816,active_dense_slot=4,retained_slots=[0,1,2,3],
        original_training_cache_checks=checks,evaluation_all_cache_bytes_and_pointers_preserved=True,
        unguarded_evaluation_cache_changes=unguarded_changes,training_refresh_resumes=True,
        install_idempotent=True,full_model_probe_and_continuation_tested=False,
        scope='Evaluation-only unused FP8 cache producer guard; BF16 packing, forward/backward and every training producer remain original.')
