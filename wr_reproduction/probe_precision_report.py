"""Paired text precision and measured calibration budget; no reserved-text input."""
import math
import statistics
import copy
from wr_reproduction.probe_study import read,paired,digest,signature


def report(mlp,ks3,protocol):
    a,b=copy.deepcopy(mlp['calibration']),copy.deepcopy(ks3['calibration'])
    for r in (a,b):
        if protocol['tokens']!=r['probe_signature']['tokens']:
            contingency=r['contingency'];assert contingency['tokens']==protocol['tokens']
            assert contingency['state_restoration']['passed']
            r.update({k:contingency[k] for k in ('chunk_losses','probe_signature','signature_hash')})
            r.update(cold_probe_seconds=contingency['cold_cost_upper_estimate'],
                     warm_probe_seconds=contingency['probe_seconds'],pending_probe_seconds=contingency['probe_seconds'])
    for r,architecture in ((a,'mlp'),(b,'ks3_experts')):
        assert r['passed'] and r['mode']=='precision' and r['architecture']==architecture
        assert r['state_restoration']['passed'] and r['pending_state_restoration']['passed']
        assert r['continuation_parity']['passed'] and r['oracle_max_abs']<1e-6
        assert r['probe_signature']['purpose']=='screening'
        assert r['signature_hash']==digest(r['probe_signature'])
        measured=r['probe_signature']
        expected=signature(protocol,measured['validation_data'],measured['backend_hash'])
        # A new terminal holdout may be selected before references are frozen.
        # It is never read in calibration and cannot change screening text/math.
        for key in ('reserved_offset','reserved_tokens'):expected[key]=measured[key]
        assert expected==measured,'Final screening protocol differs from measured calibration'
    assert a['probe_signature']==b['probe_signature'],'Both calibration architectures must use identical fixed evaluation'
    block=paired(b['chunk_losses'],a['chunk_losses'])
    # Contiguous groups retain local correlation within32K-token blocks. This
    # sensitivity check is still not a guarantee of independence across text.
    def grouped(values):
        assert len(values)%4==0
        return [statistics.mean(values[i:i+4]) for i in range(0,len(values),4)]
    grouped_block=paired(grouped(b['chunk_losses']),grouped(a['chunk_losses']))
    time_gap=abs(a['prefix_elapsed_seconds']-b['prefix_elapsed_seconds'])
    count=math.ceil(900./protocol['interval_seconds'])-1
    costs={}
    for label,r in (('mlp',a),('ks3',b)):
        warm=max(r['warm_probe_seconds'],r['pending_probe_seconds'])
        # Leave measured-cost headroom for normal screening, logging and update
        # diagnostics. Actual900s runs must report their complete cost separately.
        estimate=r['cold_probe_seconds']+max(0,count-1)*warm
        costs[label]=dict(probes=count,cold=r['cold_probe_seconds'],warm_upper=warm,
                          projected_probe_seconds=estimate,projected_probe_fraction=estimate/900.,
                          with_20_percent_headroom_fraction=1.2*estimate/900.)
    precision=block['resolves_001'] and grouped_block['resolves_001']
    comparable=time_gap<=protocol['max_reference_age_seconds']
    budget=all(v['with_20_percent_headroom_fraction']<=.03 for v in costs.values())
    return dict(status='passed' if precision and comparable and budget else 'needs_revision',
                mlp_run_id=mlp['run_id'],ks3_run_id=ks3['run_id'],protocol=protocol,
                block8192_uncertainty=block,block32768_sensitivity=grouped_block,
                checkpoint_gap_seconds=time_gap,comparable_prefix_times=comparable,
                precision_passed=precision,budget_passed=budget,measured_cost_projection=costs,
                not_a_global_precision_guarantee=True,
                limitation='One paired seed169 model pair near120 training seconds; adjacent text blocks may remain correlated. Continue reporting actual paired uncertainty at all future comparisons.',
                terminal_confirmation_interval=[protocol['reserved_offset'],protocol['reserved_offset']+protocol['reserved_tokens']],
                calibration_runs_are_not_screening_references=True,reserved_text_excluded=True)
