"""Fetch the pinned PR367 corpus concurrently and verify every published SHA256."""
import concurrent.futures
import datetime
import hashlib
import json
import os
import struct
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'research_results/wr_reproduction/h100-pr367'


def main():
    metadata = json.loads((RESULTS / 'dataset.json').read_text())
    repo, revision = metadata['repository'], metadata['revision']
    info = HfApi().dataset_info(repo, revision=revision, files_metadata=True)
    expected = {s.rfilename: s.lfs.sha256 for s in info.siblings if s.lfs}
    names = [metadata['validation'], *[f'fineweb_train_{i:06d}.bin' for i in metadata['train_shards']]]
    destination = ROOT / 'data/fineweb100B'
    destination.mkdir(parents=True, exist_ok=True)

    def get(name):
        for attempt in range(3):
            try:
                p = Path(hf_hub_download(repo, name, repo_type='dataset', revision=revision, local_dir=destination))
                digest = hashlib.sha256()
                with p.open('rb') as f:
                    header = struct.unpack('<256i', f.read(1024))
                    assert header[0] == 20240520 and header[1] == 1, (name, header[:3])
                    f.seek(0)
                    for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                        digest.update(chunk)
                assert p.stat().st_size == 1024 + 2 * header[2], name
                assert digest.hexdigest() == expected[name], (name, 'SHA256 mismatch')
                return dict(file=name, sha256=digest.hexdigest(), tokens=header[2], bytes=p.stat().st_size)
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        pending = {pool.submit(get, name): name for name in names}
        for future in concurrent.futures.as_completed(pending):
            row = future.result()
            rows.append(row)
            with (RESULTS / 'dataset-files.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
            if len(rows) % 10 == 0 or len(rows) == len(names):
                print(f'{len(rows)}/{len(names)} files downloaded and SHA256 verified', flush=True)
    (RESULTS / 'dataset-ready.json').write_text(json.dumps(dict(
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), repository=repo,
        revision=revision, files=len(rows), bytes=sum(r['bytes'] for r in rows),
        tokens=sum(r['tokens'] for r in rows), all_sha256_verified=True), indent=2) + '\n')


if __name__ == '__main__':
    main()
