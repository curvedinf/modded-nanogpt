"""Install the pinned NVIDIA assembler fixing Hopper FP8 dual-layout stores."""
from pathlib import Path
import hashlib,json,subprocess,tarfile,urllib.request
ROOT=Path('/root/venvs/h100-ptxas-13.3')
URL='https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-13.3.33-archive.tar.xz'
SHA='93b098bda4a562ebf3541523ce82adc43f106a81dcf28bcbf8f0d8e093d1c66f'

def main():
    ROOT.mkdir(exist_ok=True,parents=True);archive=ROOT/'cuda_nvcc-13.3.33.tar.xz'
    if not archive.exists():urllib.request.urlretrieve(URL,archive)
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==SHA,'NVIDIA archive checksum mismatch'
    with tarfile.open(archive) as tf:
        member=next(v for v in tf.getmembers() if v.name.endswith('/bin/ptxas'))
        binary=ROOT/'ptxas';binary.write_bytes(tf.extractfile(member).read());binary.chmod(0o755)
    print(subprocess.check_output([str(binary),'--version'],text=True))
    print('TRITON_PTXAS_PATH='+str(binary))
if __name__=='__main__':main()
