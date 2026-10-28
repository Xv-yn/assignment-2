"""Deterministic unit checks for CIFAR GPU pipeline; no CUDA required."""
import unittest
import numpy as np
import torch
from algorithm.cifar_gpu_v5 import augment_batch, make_model, evidence_gate, evaluate_candidate,seed_everything
from algorithm.cifar_gpu_v5_summary import summarize

class GPUPipelineTests(unittest.TestCase):
    def test_augment_dtype_shape_bounds(self):
        seed_everything(2)
        x=torch.arange(3*32*32,dtype=torch.float32).reshape(1,3,32,32)/3072
        y=augment_batch(x)
        self.assertEqual(y.shape,x.shape)
        self.assertEqual(y.dtype,x.dtype)
        self.assertTrue(torch.isfinite(y).all())
        self.assertGreaterEqual(y.min(),x.min())
        self.assertLessEqual(y.max(),x.max())

    def test_forward_model(self):
        x=torch.rand(2,3,32,32)
        logits=make_model(16)(x)
        self.assertEqual(tuple(logits.shape),(2,3))
        self.assertTrue(torch.isfinite(logits).all())

    def test_gate_requires_all_rules(self):
        self.assertTrue(evidence_gate(.02,3,[100,101,102])['eligible_for_downstream_confirmation'])
        self.assertFalse(evidence_gate(.02,500,[200,200,200])['eligible_for_downstream_confirmation'])
        self.assertFalse(evidence_gate(.004,3,[200,200,200])['eligible_for_downstream_confirmation'])
        self.assertFalse(evidence_gate(.03,3,[200,10,200])['eligible_for_downstream_confirmation'])

    def test_collapse_cannot_fabricate_matrix(self):
        logits=np.repeat(np.array([[5.,0.,0.]]),60,axis=0)
        labels=np.tile([0,1,2],20)
        res=evaluate_candidate(logits,logits,labels,labels,np.ones(3)/3,seed=0)
        self.assertIsNone(res['estimated_T'])
        self.assertFalse(res['quality_gate']['eligible_for_downstream_confirmation'])

    def test_rows_valid_on_balanced_pseudo_labels(self):
        logits=np.vstack([np.tile([3.,1.,0.],(100,1)),np.tile([1.,3.,0.],(100,1)),np.tile([0.,1.,3.],(100,1))])
        y=np.tile([0,1,2],100)
        res=evaluate_candidate(logits,logits,y,y,np.ones(3)/3,seed=11)
        self.assertIsNotNone(res['estimated_T'])
        np.testing.assert_allclose(np.asarray(res['estimated_T']).sum(1),1,atol=1e-10)

    def test_reject_cpu_smoke_as_gpu_evidence(self):
        import tempfile,json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'seed0.json').write_text(json.dumps({'status':'cpu_smoke_not_for_reporting'}))
            with self.assertRaisesRegex(ValueError,'No completed FULL CUDA'):
                summarize(Path(d))

if __name__=='__main__':unittest.main()
