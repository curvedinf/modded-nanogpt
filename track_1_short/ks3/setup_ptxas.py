"""Pinned assembler for Hopper's FP8 dual-layout stores."""
import hashlib
import os
from pathlib import Path
import tarfile
import urllib.request

URL = 'https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-13.3.33-archive.tar.xz'
SHA256 = '93b098bda4a562ebf3541523ce82adc43f106a81dcf28bcbf8f0d8e093d1c66f'


def main():
    cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'nanogpt-ptxas-13.3.33'
    cache.mkdir(parents=True, exist_ok=True)
    binary = cache / 'ptxas'
    if not binary.exists():
        archive = cache / 'cuda_nvcc.tar.xz'
        if not archive.exists():
            urllib.request.urlretrieve(URL, archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError('NVIDIA assembler archive checksum mismatch')
        with tarfile.open(archive) as files:
            member = next(m for m in files.getmembers() if m.name.endswith('/bin/ptxas'))
            temporary = cache / f'ptxas.{os.getpid()}.tmp'
            temporary.write_bytes(files.extractfile(member).read())
            temporary.chmod(0o755)
            temporary.replace(binary)
    print(binary)


if __name__ == '__main__':
    main()
