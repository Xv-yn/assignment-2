"""CPU-only tests for V4 transition diagnostics and GPU model wiring."""
import unittest
import importlib.util
import numpy as np
from algorithm.cifar_transition_v4 import texture_features,split_cal_audit,row_bootstrap_ci
from algorithm.cifar_cnn_gpu_v4 import quality_guard, make_model
from algorithm.transition_empirical_v3 import empirical_pseudo_confusion

class CIFARV4Tests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('skimage'),
                         'Optional HOG study requires requirements_additional.txt')
    def test_hog_feature_shape_and_finite(self):
        x=np.zeros((2,32,32,3),dtype=np.uint8)
        x[0,:,16:,0]=255
        h=texture_features(x)
        self.assertEqual(h.shape,(2,324))
        self.assertTrue(np.isfinite(h).all())
        self.assertEqual(h.dtype,np.float32)
        self.assertGreater(float(h[0].sum()),0)
        self.assertAlmostEqual(float(h[1].sum()),0)

    def test_hog_rejects_wrong_format(self):
        for raw in (np.zeros((2,32,32,3),dtype=np.float32),np.zeros((2,3,32,32),dtype=np.uint8)):
            with self.assertRaises(ValueError):texture_features(raw)

    def test_disjoint_calibration_and_audit(self):
        v=np.arange(3000)
        a,b=split_cal_audit(v,0)
        self.assertEqual(len(a),1500)
        self.assertEqual(len(b),1500)
        self.assertEqual(len(set(a)&set(b)),0)
        self.assertEqual(set(np.r_[a,b]),set(v))
        x,y=split_cal_audit(v,0)
        np.testing.assert_array_equal(a,x)
        np.testing.assert_array_equal(b,y)

    def test_bootstrap_intervals_contain_planted_transition_approximately(self):
        rng=np.random.default_rng(83)
        p=np.tile(np.eye(3),(100,1))*0.95+0.05/3
        s=np.tile(np.arange(3),100)
        t,meta=empirical_pseudo_confusion(p,s)
        self.assertEqual(meta['pseudo_counts'],[100]*3)
        r=row_bootstrap_ci(p,s,repeats=50,seed=21)
        self.assertEqual(r['bootstrap_repeats'],50)
        lo=np.asarray(r['lo']);hi=np.asarray(r['hi'])
        self.assertEqual(lo.shape,(3,3))
        self.assertTrue((lo<=t).all() and (t<=hi).all())
        self.assertTrue(np.allclose(t.sum(1),1))

    def test_quality_gate_rejects_weak_illconditioned(self):
        self.assertFalse(quality_guard(0.004,40,500)['safe_to_try_in_forward'])
        self.assertFalse(quality_guard(0.03,10,2)['safe_to_try_in_forward'])
        self.assertTrue(quality_guard(0.03,10,150)['safe_to_try_in_forward'])

    def test_cnn_output_shape_and_finiteness(self):
        import torch
        m=make_model()
        m.eval()
        with torch.no_grad():out=m(torch.zeros((2,3,32,32),dtype=torch.float32))
        self.assertEqual(tuple(out.shape),(2,3))
        self.assertTrue(bool(torch.isfinite(out).all()))

if __name__=='__main__':unittest.main()
