"""GPU check of accepted-window restoration using the actual pinned YaRN code."""
from types import MethodType,SimpleNamespace
import torch


@torch.no_grad()
def check_gpu():
    # The preceding standalone MLP check installs only MLP kernels. Load the
    # audited attention import hook before importing the original model: it
    # replaces the unavailable Hopper binary binding and leaves YaRN intact.
    from wr_reproduction.backend import install
    install('pr393')
    from track_1_short.model.attention import Yarn
    from track_1_short.model.gpt import NARROW_HEAD_DIM,NARROW_ATTN_SCALE,WIDE_ATTN_SCALE
    from track_1_short.config import (Hyperparameters,TRAINING_STAGES,LR_COOLDOWN_FRAC,
                                     SPLIT_EMBED_STAGE,WS_POST_YARN_EXT)
    from track_1_short.schedule import TrainingSchedule
    from track_1_short.training import TrainingManager
    from wr_reproduction.research_evaluation import AcceptedEvaluation,WINDOWS
    hp=Hyperparameters();rows=hp.val_batch_size//8
    schedule=TrainingSchedule(TRAINING_STAGES,hp.num_scheduled_iterations,hp.num_extension_iterations,
        'cpu',LR_COOLDOWN_FRAC,SPLIT_EMBED_STAGE,WS_POST_YARN_EXT)
    def manager():
        yarns=(Yarn(NARROW_HEAD_DIM,rows,attn_scale=NARROW_ATTN_SCALE,device='cuda'),
               Yarn(NARROW_HEAD_DIM,rows,paired=True,attn_scale=NARROW_ATTN_SCALE,device='cuda'),
               Yarn(128,rows,attn_scale=WIDE_ATTN_SCALE,device='cuda'))
        for yarn in yarns:yarn.rebuild_rows=max(s.batch_size for s in TRAINING_STAGES)//8
        model=SimpleNamespace(yarns=yarns,yarn=yarns[0],yarn_paired_head=yarns[1],yarn_wide=yarns[2],
                              learned_weight_fixture=torch.arange(32,device='cuda',dtype=torch.float32))
        first=TRAINING_STAGES[0]
        result=SimpleNamespace(model=model,schedule=schedule,ws_short=first.window_sizes[0],
            ws_long=first.window_sizes[1],train_max_seq_len=first.train_max_seq_len)
        result.advance_schedule=MethodType(TrainingManager.advance_schedule,result)
        result.get_forward_args=MethodType(TrainingManager.get_forward_args,result)
        return result
    candidate=manager();reference=manager();policy=AcceptedEvaluation()
    weight=candidate.model.learned_weight_fixture.clone()
    addresses=[(y.factor1.data_ptr(),y.factor2.data_ptr()) for y in candidate.model.yarns]
    starts=[start for start,_ in schedule.boundaries]
    for step in starts+[schedule.total_steps]:
        candidate.advance_schedule(step);policy.observe(candidate,step)
    for step in starts[:3]:reference.advance_schedule(step)
    for yarn in reference.model.yarns:yarn.ensure_full()
    cfg,report=policy.prepare(candidate)
    assert report['restored'] and report['compatible']
    assert (cfg.ws_short,cfg.ws_long)==WINDOWS
    assert (candidate.ws_short,candidate.ws_long)==(6,13)
    for a,b,address in zip(candidate.model.yarns,reference.model.yarns,addresses,strict=True):
        assert torch.equal(a.angular_freq,b.angular_freq)
        assert a.attn_scale==b.attn_scale
        assert torch.equal(a.factor1,b.factor1) and torch.equal(a.factor2,b.factor2)
        assert (a.factor1.data_ptr(),a.factor2.data_ptr())==address
    assert torch.equal(candidate.model.learned_weight_fixture,weight)
    # Identical in discovery168 and all three fresh verification seeds.
    expected='f578f7187981f9a8220c90ea99eaaa35616aa87bf99b49cfdd2307f3628f3612'
    assert report['yarn_snapshot_sha256']==expected,report
    torch.cuda.synchronize()
    return dict(passed=True,device='cuda',table_rows=rows,source_window_schedule_traversed=True,
                original_buffer_addresses_preserved=True,rotary_tables_bitwise_equal=True,
                learned_weight_fixture_unchanged=True,matches_verified_yarn_snapshot=True,
                evaluation_protocol=report,full_model_validation_performed=False)
