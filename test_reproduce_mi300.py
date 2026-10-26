"""CPU-only tests for frozen, portable reproduction launch and evidence handling."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import reproduce_mi300 as reproduce


class ReproductionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='reproduction test ')
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.manifest = reproduce.read(reproduce.MANIFEST)

    def test_published_sources_and_original_manifests(self):
        self.assertGreaterEqual(reproduce.verify_sources(reproduce.ROOT, self.manifest), 237)
        from wr_reproduction.contracts import audit
        for version in ('pr393','record360'):
            audit(version)
        fence = reproduce.read(reproduce.ROOT / self.manifest['fence'])
        for arm in ('mlp','ks3'):
            cfg = reproduce.read(reproduce.ROOT / self.manifest['arms'][arm]['config'])
            self.assertEqual(cfg['backend_hash'], fence['backend_hash'])
            self.assertEqual(cfg['probe_protocol'], fence['protocol'])
            self.assertEqual({k:cfg['recipe'][k] for k in fence['settings']}, fence['settings'])

    def test_source_corruption_rejected(self):
        source = self.base / 'kernel.py'
        source.write_text('original\n')
        manifest = dict(files={'kernel.py':reproduce.sha(source)})
        source.write_text('changed\n')
        with self.assertRaisesRegex(ValueError, 'kernel.py'):
            reproduce.verify_sources(self.base, manifest)

    def data_fixture(self):
        data = dict(repository='test/frozen',revision='abc123',files=[],tokenizer_files=[])
        for name in ('fineweb_train_000001.bin','fineweb_val_000000.bin'):
            content = name.encode()
            path = self.base / 'data/fineweb10B' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            data['files'].append(dict(path=str(path.relative_to(self.base)),bytes=len(content),
                                      sha256=hashlib.sha256(content).hexdigest()))
        return data

    def test_corrupt_or_additional_data_rejected(self):
        data = self.data_fixture()
        self.assertEqual(reproduce.verify_data(self.base, data), 2)
        extra = self.base / 'data/fineweb10B/fineweb_train_000002.bin'
        extra.touch()
        with self.assertRaisesRegex(ValueError, 'extra='):
            reproduce.verify_data(self.base, data)
        extra.unlink()
        (self.base / data['files'][0]['path']).write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'checksum/size'):
            reproduce.verify_data(self.base, data)

    def test_download_uses_immutable_revision_and_rejects_bad_bytes(self):
        data = self.data_fixture()
        item = data['files'][0]
        path = self.base / item['path']
        content = path.read_bytes()
        path.unlink()
        with patch.object(reproduce.urllib.request,'urlopen',return_value=io.BytesIO(content)) as request:
            reproduce.download(self.base, data)
        self.assertIn('/resolve/abc123/', request.call_args.args[0])
        path.unlink()
        with patch.object(reproduce.urllib.request,'urlopen',return_value=io.BytesIO(b'wrong')):
            with self.assertRaisesRegex(ValueError, 'Downloaded checksum'):
                reproduce.download(self.base, data)
        self.assertFalse(path.exists())
        self.assertFalse(list(path.parent.glob('*.part-*')))

    def test_relocation_preserves_recipe_eval_and_seed_and_venv(self):
        for arm,seed in [('mlp',169),('ks3',370)]:
            output = self.base / 'output with spaces'
            data_root = self.base / 'external data'
            interpreter = self.base / 'venv/bin/python'
            original = reproduce.read(reproduce.ROOT / self.manifest['arms'][arm]['config'])
            run = reproduce.prepare(reproduce.ROOT,self.manifest,arm,seed,output,data_root,python=interpreter)
            for key in ('recipe','source_hashes','backend_hash','backend_fence','probe_protocol','validation_data','code_commit'):
                self.assertEqual(run[key],original[key],key)
            self.assertEqual(run['seed'],seed)
            self.assertTrue(run['comparative_pruning_exempt'])
            self.assertEqual(run['environment']['DATA_PATH'],str(data_root))
            self.assertEqual(run['environment']['WR_PROBE_STATE'],str(output))
            self.assertEqual(run['command'],[str(interpreter),'-m','wr_reproduction.probe_train','--run',run['directory']])
            self.assertTrue(Path(run['directory']).is_relative_to(output))
            self.assertEqual(reproduce.read(Path(run['directory'])/'config.json'),original['recipe'])
            # No pre-existing source-host run path leaks into operational fields.
            self.assertNotIn(original['directory'],json.dumps({k:run[k] for k in ('directory','command','environment','stop_files')}))
        self.assertEqual(len((output/'registry.jsonl').read_text().splitlines()),2)

    def test_respects_stop_and_protects_existing_logs(self):
        output = self.base / 'existing'
        output.mkdir()
        (output/'historical.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'isolated reproduction'):
            reproduce.prepare(reproduce.ROOT,self.manifest,'mlp',169,output,self.base)
        self.assertEqual((output/'historical.json').read_text(),'{}')
        (output/'STOP').touch()
        with self.assertRaisesRegex(ValueError,'STOP'):
            reproduce.prepare(reproduce.ROOT,self.manifest,'mlp',169,output,self.base)

    def test_original_controller_launch_and_result_collection_without_gpu(self):
        output = self.base / 'controller output'
        run = reproduce.prepare(reproduce.ROOT,self.manifest,'mlp',169,output,self.base)
        # Exercise the real frozen controller/environment/child process lifecycle.
        # Only the training subprocess is replaced with a CPU event emitter.
        worker = self.base / 'emit validation.py'
        source = reproduce.read(reproduce.ROOT / self.manifest['arms']['mlp']['records'][0]['result'])
        worker.write_text('import json,sys\nfrom pathlib import Path\n'
                          + 'Path(sys.argv[1],"events.jsonl").write_text('
                          + repr(json.dumps(source['validation'])+'\n') + ')\n')
        run['command'] = [sys.executable,str(worker),run['directory']]
        reproduce.write(Path(run['directory'])/'run.json',run)
        launch = ('import json,sys\nfrom pathlib import Path\n'
                  'from wr_reproduction.probe_controller import execute\n'
                  'run=json.loads(Path(sys.argv[1]).read_text())\n'
                  'assert execute(run)["status"]=="complete"\n')
        process = subprocess.run([sys.executable,'-c',launch,str(Path(run['directory'])/'run.json')],
                                 cwd=reproduce.ROOT,env=dict(os.environ,**run['environment']),
                                 text=True,capture_output=True,timeout=30)
        self.assertEqual(process.returncode,0,process.stdout+process.stderr)
        result = reproduce.read(output/'results'/(run['run_id']+'-result.json'))
        self.assertEqual(result['validation']['loss'],source['validation']['loss'])
        self.assertEqual(reproduce.read(output/'latest-run.json')['status'],'complete')
        self.assertEqual(reproduce.summarize(output,self.manifest)['statistics']['mlp']['count'],1)

    def test_summary_uses_loss_time_and_matching_evaluation(self):
        control_path = self.manifest['arms']['mlp']['records'][0]['result']
        result = reproduce.read(reproduce.ROOT/control_path)
        for index,change in enumerate(({}, {'train_seconds':901.0}, {'evaluation_signature':{}}, {'loss':None})):
            trial = copy.deepcopy(result)
            trial['run_id'] = f'test-{index}'
            trial['validation'].update(change)
            trial['validation']['step'] = 1 if index == 0 else 99999
            if index == 3:
                trial['status'] = 'error'
            reproduce.write(self.base/'results'/f'test-{index}-result.json',trial)
        report = reproduce.summarize(self.base,self.manifest)
        self.assertEqual(report['statistics']['mlp']['count'],1)
        self.assertEqual(report['statistics']['mlp']['mean'],result['validation']['loss'])
        self.assertEqual([r['matches_evaluation_and_budget'] for r in report['runs']],[True,False,False,False])


if __name__ == '__main__':
    unittest.main()
