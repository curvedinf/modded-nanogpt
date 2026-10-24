"""PR393 four-map, MoE-margin replay checks; registered GPU pilots only."""
import torch
from control393.optim import bank_update
from test_ks3_optimizer_graphs import run_device_checks as check_replay


def run_device_checks(device,charts=512):
    return check_replay(device,update=bank_update,benchmark=True,specs=(
        ('qk',torch.bfloat16,(2,128,768)),
        ('vo',torch.bfloat16,(2,768,128)),
        ('mlp_bank',torch.bfloat16,(10,2816,768)),
        ('moe_bank',torch.float32,(80,charts,768)),
    ))
