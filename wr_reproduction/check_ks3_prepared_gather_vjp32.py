"""Exact prepared-cache parity against the validated gather/row32 ingredient."""
from wr_reproduction import ks3_moe as core
from wr_reproduction import ks3_bf16_gather_vjp32 as backward
from wr_reproduction import ks3_prepared_gather_vjp32 as combined


def check():
    # The reused checker disables only the prepared geometry consumers for its
    # reference. Both sides therefore use the same validated gather/row32 path.
    # This specifically tests the interaction between the cache and row32 VJP.
    backward.install()
    from wr_reproduction.check_ks3_prepared_geometry import check as check_cache
    result=check_cache()
    assert core.backward is backward.backward
    assert backward.EXECUTED_MODES == {'training_bf16_exact_gather_vjp32','original_untuned_backward'}
    combined.install();combined.install()
    result['combined_vs_gather_vjp32_all_outputs_and_gradients_bitwise_equal']=result.pop('all_original_outputs_and_gradients_bitwise_equal')
    result.update(reference='Validated exact BF16 gather/row32 VJP with original geometry operands; original64-row baseline uses the prior independent reduction audit.',
        reference_numeric_run_id='probe-experts-bf16-gather-vjp32-calibration-20261026T135432-b9af7c',
        original_forward_and_evaluation=True,original_coordinate_gradient_BF16_storage=True,
        tolerance_scope='Every new combination comparison is bitwise, including geometry gradients. Prior ingredient evidence separately bounds32-vs64-row FP32 geometry reduction; no tolerance is relaxed.',
        original_row32_backward_source_unchanged=True,new_math_approximations=False,
        cache_and_row32_interaction_tested=True)
    return result
