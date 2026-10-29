"""Tests of the empirical held-out noise matrix estimator and strategy selection."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from algorithm.transition_empirical_v3 import empirical_pseudo_confusion
from algorithm.summarize_transition_v3 import select_method_by_signal, summarize

class EmpiricalEstimatorTests(unittest.TestCase):
    def test_rows_match_noisy_label_frequencies(self):
        probs=np.zeros((30,3)); y=np.zeros(30,dtype=int)
        for cls in range(3):
            probs[cls*10:(cls+1)*10,cls]=1
            y[cls*10:(cls+1)*10] = np.array([0,0,0,1,1,1,2,2,2,2])
        t,meta=empirical_pseudo_confusion(probs,y)
        np.testing.assert_allclose(t,np.tile([.3,.3,.4],(3,1)))
        self.assertEqual(meta['selected_counts'],[10,10,10])
    def test_rows_and_counts_are_valid(self):
        rng=np.random.default_rng(5);p=rng.dirichlet(np.ones(3),size=300)
        y=rng.integers(0,3,size=300)
        for frac in (.1,.5,1.):
            t,meta=empirical_pseudo_confusion(p,y,fraction=frac)
            self.assertTrue(np.isfinite(t).all());self.assertTrue((t>=0).all());np.testing.assert_allclose(t.sum(1),1)
            self.assertEqual(sum(meta['pseudo_counts']),300)
    def test_invalid_inputs_rejected(self):
        p=np.full((30,3),1/3.);labels=np.array([0,1,2]*10)
        for x,y,f in [(p[:-1],labels,1),(p,labels,0),(p,labels,1.01),(p,labels+5,1),(np.full((30,3),.1),labels,1)]:
            with self.assertRaises(ValueError):empirical_pseudo_confusion(x,y,fraction=f)
    def test_no_class_detected(self):
        p=np.zeros((30,3));p[:,0]=1;y=np.array([0,1,2]*10)
        with self.assertRaisesRegex(ValueError,'insufficient'):empirical_pseudo_confusion(p,y)
    def test_signal_selection(self):
        self.assertEqual(select_method_by_signal(.5),'hard_top50')
        self.assertEqual(select_method_by_signal(.004),'hard_all')
        self.assertEqual(select_method_by_signal(.05),'hard_top50')
        with self.assertRaises(ValueError):select_method_by_signal(float('nan'))
    def test_seed_input_not_using_clean_test(self):
        import inspect
        from algorithm.transition_empirical_v3 import estimates_for_split
        src=inspect.getsource(estimates_for_split)
        self.assertNotIn("load_clean_test",src)
        self.assertNotIn("'Yts'",src)
        self.assertNotIn("'Xts'",src)

if __name__=='__main__':unittest.main()
