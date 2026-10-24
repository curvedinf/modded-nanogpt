"""Plan initial LR/stage timing only; a trial has no step or wall-time stop budget."""
import copy
import json
import math
from pathlib import Path
import statistics
from itertools import accumulate, pairwise
import torch
from mi300.config import (TRAINING_STAGES, LR_COOLDOWN_FRAC, SPLIT_EMBED_STAGE, WS_POST_YARN_EXT,
                          SCHEDULE_GROWTH_STAGE, SCHEDULE_GROWTH_STEPS)
from mi300.schedule import TrainingSchedule
from ks3_research.continuation import HoldLast
from ks3_research.sampling import candidate_stages


def stages_for(config):
    stages = copy.deepcopy(TRAINING_STAGES)
    batches = config['batch_tokens']
    if len(batches) != len(stages) or any(n < 1024 or n % 1024 for n in batches):
        raise ValueError('Specify five positive batch sizes, each a multiple of 1024')
    for stage, batch in zip(stages, batches):
        stage.batch_size = batch
    durations=config.get('stage_durations')
    if durations is not None:
        if (len(durations)!=len(stages)-1 or
                any(not math.isfinite(x) or x<=0 for x in durations) or
                not math.isclose(sum(durations),1.,rel_tol=0.,abs_tol=1e-9)):
            raise ValueError('Specify four positive finite stage durations summing to one')
        for stage,duration in zip(stages[:-1],durations):
            stage.duration=duration
    return stages


def schedule_for(config, base_steps, device):
    schedule = TrainingSchedule(stages_for(config), base_steps, 20, device,
                                LR_COOLDOWN_FRAC, SPLIT_EMBED_STAGE, WS_POST_YARN_EXT)
    schedule.mtp_weights = HoldLast(schedule.mtp_weights)
    schedule.prefix_weights = HoldLast(schedule.prefix_weights)
    return schedule


def signature(config):
    return dict({k: config[k] for k in ('charts', 'rank', 'nonlinear', 'batch_tokens')},
                attention_backend=config.get('attention_backend','original_sdpa'),
                ks3_backward=config.get('ks3_backward','vendor'),
                ks3_forward=config.get('ks3_forward','original'),
                optimizer_backend=config.get('optimizer_backend','eager'),
                loss_backend=config.get('loss_backend','original'),
                loss_slab_tokens=config.get('loss_slab_tokens',1024),
                sampled_candidates=candidate_stages(config))


def stage_lengths(config, base):
    stages = stages_for(config)
    ends = [0, *[round(c*base) for c in accumulate(s.duration for s in stages[:-1])],
            base+SCHEDULE_GROWTH_STEPS+20]
    ends[SCHEDULE_GROWTH_STAGE+1:-1] = [v+SCHEDULE_GROWTH_STEPS for v in ends[SCHEDULE_GROWTH_STAGE+1:-1]]
    return [b-a for a,b in pairwise(ends)]


def plan(config, pilot):
    pilot_signature = dict(pilot['signature'])
    pilot_signature.setdefault('attention_backend','original_sdpa')
    pilot_signature.setdefault('ks3_backward','vendor')
    pilot_signature.setdefault('ks3_forward','original')
    pilot_signature.setdefault('optimizer_backend','eager')
    pilot_signature.setdefault('loss_backend','original')
    pilot_signature.setdefault('loss_slab_tokens',1024)
    pilot_signature.setdefault('sampled_candidates',candidate_stages({}))
    if signature(config) != pilot_signature:
        raise ValueError('Run a pilot for this chart/rank/precision/batch configuration')
    rates = []
    for stage in range(5):
        samples = [r['step_seconds'] for r in pilot['steps'] if r['stage'] == stage]
        if len(samples) < 2:
            raise ValueError(f'Pilot has insufficient samples for stage {stage}')
        # Discard the first sample of each stage (first-use specialization).
        rates.append(statistics.median(samples[1:]))
    limit = config['control_wall_seconds'] * config['schedule_time_fraction']
    val_seconds = (pilot['validation_setup_seconds']
                   +statistics.median(pilot['validation_batch_seconds'][1:])*40)
    evaluations = max(1, int(limit // config['validation_interval_seconds']))
    reserve = 1.25 * val_seconds * evaluations + pilot['setup_seconds'] + 15.
    available = (limit - reserve) * .9
    if available <= 0:
        available = 0.
    def predicted(base):
        return sum(n*rate for n,rate in zip(stage_lengths(config,base),rates))
    # Keep the original tail windows valid and finish on a complete sparse update cycle.
    low, high = 648, 100000
    # A slow candidate still trains. This is a schedule estimate, never a failure
    # or stop condition merely because it cannot finish before the control.
    while high-low >= 4:
        middle = ((low+high)//8)*4
        if middle <= low:
            break
        if predicted(middle) <= available:
            low = middle
        else:
            high = middle
    low = int(config.get('schedule_base_steps', low))
    if low < 648 or low % 4:
        raise ValueError('The initial schedule base must be >=648 and divisible by four')
    return dict(base_steps=low, total_steps=low+72, stage_seconds=rates,
                estimated_training_seconds=predicted(low), estimated_validation_seconds=val_seconds,
                estimated_validation_count=evaluations, schedule_planning_seconds=limit,
                estimated_total_seconds=predicted(low)+reserve,
                termination='full_validation_target_only',
                rationale='Pilot estimates initial LR/stage timing only. Continue at final settings until full CE target; no step or wall-time cutoff.')


def load_config(path):
    config = json.loads(Path(path).read_text())
    # Legacy config fields are accepted as planning inputs, never stop policies.
    config.setdefault('schedule_time_fraction', config.get('wall_budget_fraction', 1.))
    config.setdefault('validation_interval_seconds', 600.)
    config.setdefault('first_validation_seconds', config['validation_interval_seconds'])
    config.setdefault('target_val_loss', 3.793633)
    config['sampled_candidates'] = candidate_stages(config)
    config.setdefault('loss_backend','original')
    config.setdefault('optimizer_backend','eager')
    if config['optimizer_backend'] not in ('eager','bank_graphs'):
        raise ValueError('Unknown optimizer backend')
    config.setdefault('ks3_backward','vendor')
    if config['ks3_backward'] not in ('vendor','fp32_blas','fp32_blas_all'):
        raise ValueError('Invalid KS3 backward backend')
    config.setdefault('ks3_forward','original')
    if config['ks3_forward'] not in ('original','blas_output','blas_both','blas_fused_unpack') or (config['ks3_forward']!='original' and config['ks3_backward']!='fp32_blas_all'):
        raise ValueError('BLAS forward projections require the all-products BLAS backward')
    config.setdefault('loss_slab_tokens',1024)
    if config['loss_backend'] not in ('original','shared_gather','fused_softcap') or config['loss_slab_tokens']<1:
        raise ValueError('Invalid training loss backend/slab size')
    scales = config.setdefault('lr_stage_scales', [1.]*len(TRAINING_STAGES))
    if len(scales)!=len(TRAINING_STAGES) or any(not math.isfinite(x) or x<=0 for x in scales):
        raise ValueError('Specify five finite positive stage learning-rate scales')
    # Every real training observation is exposed to the machine watchdog. The
    # first eligible regression must not wait for a model monitoring turn.
    config['log_every_steps'] = 1
    if config['target_val_loss'] != 3.793633:
        raise ValueError('This study uses the fixed full-validation CE target 3.793633')
    if config['charts'] < 1 or config['rank'] < 1:
        raise ValueError('Charts and rank must be positive')
    if not 0 < config['lr_scale'] or not 0 < config['schedule_time_fraction']:
        raise ValueError('Invalid learning-rate scale or schedule-time fraction')
    from ks3_research.continuation import TargetLoop
    TargetLoop(config['target_val_loss'], config['validation_interval_seconds'], config['first_validation_seconds'])
    stages_for(config)
    return config
