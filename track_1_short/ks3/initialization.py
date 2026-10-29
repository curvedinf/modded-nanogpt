"""Preserve PR393's non-expert initialization when resizing KS3 experts."""
import torch


@torch.no_grad()
def restore_original_moe_rng(state, router, model_dim, bound):
    # Immutable PR393 initializes expert matrices and the router on CPU, then
    # the launcher moves the complete model to CUDA. Replay only its five
    # original 8x1024xD draws, independent of the chosen expert width.
    if router.device.type != 'cpu':
        raise RuntimeError('Width ablation requires original CPU initialization')
    torch.set_rng_state(state)
    scratch = torch.empty(8, 1024, model_dim, dtype=torch.float32, device='cpu')
    for _ in range(5):
        scratch.uniform_(-bound, bound)
    router.copy_(.01 * torch.randn(5, 8, model_dim, dtype=torch.float32, device='cpu'))
