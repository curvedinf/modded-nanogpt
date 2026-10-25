"""CPU checks for the new isolation, control exemption and architecture map."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from wr_reproduction.research import atomic,append,Screen,References,Metadata
from wr_reproduction.matched_architecture import layer_map
from wr_reproduction import matched_study as study

class MatchedTests(unittest.TestCase):
    def row(self,loss=4.5):
        meta=Metadata(data=[])
        return dict(event='train',step=601,objective=loss,train_seconds=100.,objective_signature=meta.training(600))
    def test_control_reference_and_exemption_and_both_stop_markers(self):
        with tempfile.TemporaryDirectory() as d:
            state=Path(d);run=state/'runs/control';second=state/'results/STOP'
            atomic(run/'run.json',dict(run_id='control',kind='control',continuous_from_initialization=True,
                                      stop_files=[str(state/'STOP'),str(second)]))
            append(state/'registry.jsonl',dict(event='registered',run_id='control'))
            append(run/'events.jsonl',self.row(4.))
            refs=References(state,'ks3').refresh();self.assertEqual(len(refs),1)
            refs[0]['run_id']='other-control'
            screen=Screen(run,state);screen.references=refs
            result=screen.observe(self.row());self.assertFalse(result['abort'])
            self.assertTrue(result['proxy_would_abort']);self.assertEqual(result['reason'],'user_authorized_matched_mlp_control')
            second.parent.mkdir();second.touch()
            self.assertTrue(screen.observe(self.row(8.))['abort'])
    def test_full_and_edge_hybrids_include_parallel_mlp(self):
        for arch,retained in [('ks3_all',set()),('ks3_first_mlp',{0}),('ks3_last_mlp',{10}),('ks3_both_mlp',{0,10})]:
            rows=layer_map(arch)
            self.assertEqual({r['layer'] for r in rows for b in r['feedforward'] if b['type']=='mlp'},retained)
            self.assertEqual(sum(len(r['feedforward']) for r in rows),10)
            self.assertEqual(len(rows[8]['feedforward']),2)
            self.assertEqual(rows[7]['feedforward'],[]);self.assertEqual(rows[9]['feedforward'],[])
    def test_dense_width_is_independent_of_expert_width(self):
        rows=layer_map('ks3_all',features=1024,dense_features=1536)
        for row in rows:
            for branch in row['feedforward']:
                expected=1024 if branch['branch']=='routed_experts' else 1536
                self.assertEqual(branch['projection_shape'][-2],expected)
        self.assertEqual(sum(b['instances'] for r in rows for b in r['feedforward']),45)

    def test_old_study_not_a_new_stop(self):
        self.assertNotIn(study.ROOT/'logs/wr-reproduction/research/STOP',study.STOP_FILES)
        self.assertEqual(len(study.STOP_FILES),2)
    def test_fence_rejects_changed_shared_source_and_setting(self):
        from wr_reproduction.research import sha
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'shared.py';source.write_text('original')
            path=root/'fence.json'
            atomic(path,dict(source_hashes={'shared.py':sha(source)},settings={'shared_mode':'frozen'}))
            with patch.object(study,'ROOT',root),patch.object(study,'FENCE',path),patch('wr_reproduction.contracts.audit'):
                study.assert_fence({'shared_mode':'frozen'})
                with self.assertRaisesRegex(RuntimeError,'setting'):study.assert_fence({'shared_mode':'different'})
                source.write_text('changed')
                with self.assertRaisesRegex(RuntimeError,'source|backend'):study.assert_fence({'shared_mode':'frozen'})
    def test_overlay_preserves_all_nonfeedforward_calls(self):
        import ast
        from collections import Counter
        from wr_reproduction.ks3_feedforward_source import method,forward_tree,refresh_tree
        def calls(tree):
            return Counter(ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call))
        original=calls(method('forward'));replacement=calls(forward_tree())
        for name in set(original)|set(replacement):
            if name not in ('self._mlp','self._matched_dense','self._mlp_fp8_args','quantize_dual_layout_fused',
                            'mlp_in.detach','mlp_in.detach().view','mlp_in.size','self._mlp_up_f8.unbind',
                            'self.retained_mlp_bank.flatten','self.retained_mlp_bank.flatten(0, 1).unbind'):
                self.assertEqual(original[name],replacement[name],name)
        self.assertEqual(original['self._mlp'],2);self.assertEqual(replacement['self._mlp'],0)
        self.assertEqual(replacement['self._matched_dense'],2)
        for count in (0,1,2):compile(ast.Module(body=[refresh_tree(count)],type_ignores=[]),'refresh','exec')
    def test_retained_adapter_binds_original_instance_method(self):
        """Exercise the real adapter and original method's calling convention on CPU."""
        import ast
        from types import SimpleNamespace
        from wr_reproduction.ks3_feedforward_source import method,compile_method
        original,_=compile_method(method('_mlp'),{'ReLUSqrdMLP':lambda *args:args})
        tree=ast.parse((study.ROOT/'wr_reproduction/ks3_feedforward.py').read_text())
        adapter=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_matched_dense')
        scope=dict(gpt=SimpleNamespace(GPT=SimpleNamespace(_mlp=original)),mapping={0:0,3:1})
        dispatch,_=compile_method(adapter,scope)
        model=object();x=object();retained=tuple(object() for _ in range(4))
        for slot,index in ((0,0),(3,1)):
            self.assertEqual(dispatch(model,slot,x,None,None,None,False,retained),
                             (x,retained[2*index],retained[2*index+1]))

if __name__=='__main__':unittest.main()
