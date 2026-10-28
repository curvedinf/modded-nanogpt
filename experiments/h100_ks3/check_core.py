import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'wr_reproduction/pr393/upstream'))
from experiments.h100_ks3.port import install
install()
# Independent existing BLAS reference checks forward, input, projection and
# geometry derivatives, including empty experts and highly skewed routing.
from wr_reproduction.check_ks3_moe import main
from wr_reproduction import ks3_moe
ks3_moe.BF16_TUNED_KERNELS=True
main()
