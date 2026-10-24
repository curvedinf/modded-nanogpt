"""ROCm-native matrix products for the unchanged ANVIL cascade."""
import torch

def XTX(a, out):
    torch.matmul(a.mT, a, out=out)

def XXT(a, out):
    torch.matmul(a, a.mT, out=out)

def ba_plus_cAA(a, alpha, beta, out):
    torch.baddbmm(a, a, a, beta=beta, alpha=alpha, out=out) if a.ndim == 3 else torch.addmm(a, a, a, beta=beta, alpha=alpha, out=out)
