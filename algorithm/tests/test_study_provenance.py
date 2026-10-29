"""Diagnostic cache and interrupted-save checks; no dataset/GPU training."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from algorithm import cifar_gpu_v5 as v5
from algorithm import cifar_transition_v4 as v4
from algorithm import cifar_gpu_v5_summary as v5_summary
from algorithm.provenance import config_hash, sha256_file, write_json


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_identity_and_hash_checked_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            identity = {'data': 'a', 'config': 'b', 'source': 'c'}
            v5._save_checkpoint({'identity': identity, 'value': torch.tensor([4])}, path, identity)
            state = v5._load_checkpoint(path, identity, 'cpu')
            self.assertEqual(state['value'].item(), 4)
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                v5._load_checkpoint(path, dict(identity, data='changed'), 'cpu')
            path.write_bytes(b'changed bytes')
            with patch('torch.load', side_effect=AssertionError('must not load corrupt file')):
                with self.assertRaisesRegex(ValueError, 'changed artifact'):
                    v5._load_checkpoint(path, identity, 'cpu')

    def test_old_checkpoint_without_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            path.write_bytes(b'old unchecked checkpoint')
            with self.assertRaisesRegex(ValueError, 'no verified provenance'):
                v5._load_checkpoint(path, {}, 'cpu')

    def test_atomic_torch_save_failure_keeps_previous_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            path.write_bytes(b'previous valid model')
            with patch('torch.save', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    v5._atomic_torch_save({}, path)
            self.assertEqual(path.read_bytes(), b'previous valid model')


def fake_pool():
    return SimpleNamespace(images=np.zeros((300, 32, 32, 3), dtype=np.uint8),
                           noisy_labels=np.resize(np.arange(3), 300))


def tiny_model(width=8):
    return torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 3))


def make_v5_finished_checkpoint(root, *, epochs=1, stale=0):
    """Synthetic already-trained checkpoint lets run() exercise finalization only."""
    data = root / 'data'
    data.mkdir()
    dataset = data / 'CIFAR.npz'
    dataset.write_bytes(b'temporary dataset fixture; loader is mocked')
    output = root / 'output'
    output.mkdir()
    settings = {'seed': 0, 'epochs': epochs, 'batch_size': 64, 'width': 8,
                'patience': 2, 'device': 'cpu', 'smoke': False,
                'torch': str(torch.__version__), 'numpy': np.__version__,
                'cuda_runtime': torch.version.cuda}
    identity = {'schema_version': 1, 'study': 'cifar_gpu_v5', 'data_sha256': sha256_file(dataset),
                'config_sha256': config_hash(settings), 'source_sha256': 'source-a',
                'shared_config_sha256': config_hash({k:v for k,v in settings.items() if k != 'seed'})}
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    state = {'identity': identity, 'epoch': 1, 'model': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'best': 1.0, 'best_epoch': 1, 'best_state': model.state_dict(),
             'history': [{'epoch': 1}], 'bad': stale,
             'torch_rng': torch.get_rng_state(), 'loader_rng': torch.Generator().manual_seed(5000).get_state(),
             'cuda_rng': None}
    v5._save_checkpoint(state, output / 'seed0_checkpoint.pt', identity)
    return data, output, identity


def finalize_v5(data, output, *, epochs=1):
    with patch.object(v5, 'source_hash', return_value='source-a'), \
         patch.object(v5, 'load_noisy_pool', return_value=fake_pool()), \
         patch.object(v5, 'make_model', side_effect=tiny_model), \
         patch.object(v5, 'evaluate_candidate', return_value={'quality_gate': {'eligible_for_downstream_confirmation': False}}):
        return v5.run(data_dir=data, out_dir=output, seed=0, epochs=epochs, batch_size=64,
                      device='cpu', width=8, patience=2, resume=True)


class V5CompletionTests(unittest.TestCase):
    def test_singular_candidate_preserves_gate_failure_with_finite_json(self):
        logits=np.repeat(np.eye(3)*4,100,axis=0)
        cal_labels=np.repeat(np.arange(3),100)
        audit_labels=np.zeros(300,dtype=np.int64)
        with patch.object(v5,'row_bootstrap_ci',return_value={}):
            record=v5.evaluate_candidate(logits,logits,audit_labels,cal_labels,np.ones(3)/3)
        self.assertIsNone(record['estimated_T_condition_number'])
        self.assertFalse(record['quality_gate']['eligible_for_downstream_confirmation'])
        json.dumps(record,allow_nan=False)

    def test_nonfinite_audit_logits_rejected_before_recording(self):
        logits=np.zeros((30,3));labels=np.resize(np.arange(3),30)
        with self.assertRaisesRegex(ValueError,'finite N x 3'):
            v5.evaluate_candidate(np.full((30,3),np.nan),logits,labels,labels,np.ones(3)/3)

    def test_early_stopped_checkpoint_finalizes_without_extra_training(self):
        with tempfile.TemporaryDirectory() as directory:
            data, output, _ = make_v5_finished_checkpoint(Path(directory), epochs=5, stale=2)
            with patch.object(v5, 'augment_batch', side_effect=AssertionError('early stop must survive resume')):
                record = finalize_v5(data, output, epochs=5)
            self.assertEqual(record['epochs_completed'], 1)
            self.assertEqual(record['epoch_budget'], 5)
            self.assertEqual(record['best_epoch'], 1)

    def test_completion_has_all_artifacts_and_resume_avoids_training(self):
        with tempfile.TemporaryDirectory() as directory:
            data, output, identity = make_v5_finished_checkpoint(Path(directory))
            record = finalize_v5(data, output)
            self.assertEqual(record['status'], 'completed_full_cpu_diagnostic')
            v5._verify_completed(record, identity, output / 'seed0_model.pt', output / 'seed0_checkpoint.pt')
            with patch.object(v5, 'source_hash', return_value='source-a'), \
                 patch.object(v5, 'load_noisy_pool', side_effect=AssertionError('completed run must not train')):
                reused = v5.run(data_dir=data, out_dir=output, seed=0, epochs=1, batch_size=64,
                                device='cpu', width=8, patience=2, resume=True)
            self.assertEqual(reused['identity'], identity)

    def test_final_model_failure_never_publishes_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            data, output, _ = make_v5_finished_checkpoint(Path(directory))
            with patch.object(v5, '_atomic_torch_save', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    finalize_v5(data, output)
            self.assertFalse((output / 'seed0.json').exists())

    def test_missing_or_tampered_final_model_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            data, output, identity = make_v5_finished_checkpoint(Path(directory))
            record = finalize_v5(data, output)
            model = output / 'seed0_model.pt'
            original = model.read_bytes()
            model.unlink()
            with self.assertRaisesRegex(ValueError, 'Missing or changed artifact'):
                v5._verify_completed(record, identity, model, output / 'seed0_checkpoint.pt')
            model.write_bytes(original + b'tampered')
            with self.assertRaisesRegex(ValueError, 'Missing or changed artifact'):
                v5._verify_completed(record, identity, model, output / 'seed0_checkpoint.pt')

    def test_changed_data_source_or_config_rejected_before_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            data, output, _ = make_v5_finished_checkpoint(Path(directory))
            finalize_v5(data, output)
            for source, batch in [('changed-source', 64), ('source-a', 32)]:
                with self.subTest(source=source, batch=batch), \
                     patch.object(v5, 'source_hash', return_value=source), \
                     patch.object(v5, 'load_noisy_pool', side_effect=AssertionError('must reject before loading')):
                    with self.assertRaisesRegex(ValueError, 'identity differs'):
                        v5.run(data_dir=data, out_dir=output, seed=0, epochs=1, batch_size=batch,
                               device='cpu', width=8, patience=2, resume=True)
            (data / 'CIFAR.npz').write_bytes(b'replaced at same path')
            with patch.object(v5, 'source_hash', return_value='source-a'):
                with self.assertRaisesRegex(ValueError, 'identity differs'):
                    v5.run(data_dir=data, out_dir=output, seed=0, epochs=1, batch_size=64,
                           device='cpu', width=8, patience=2, resume=True)


def fake_v4_record(*args):
    model = {'audit_noisy_nll': 1.0, 'audit_information_gain_nats': .01,
             'audit_noisy_accuracy': .4, 'estimator': {'T_hat': np.eye(3).tolist(), 'condition_number': 1.0}}
    return {'seed': args[-1], 'models': {'color_logistic': dict(model), 'hog_color_rff': dict(model)}}


class V4CacheTests(unittest.TestCase):
    def make_cache(self, root, *, record_factory=fake_v4_record, seeds=(0,)):
        data = root / 'data'
        data.mkdir()
        (data / 'CIFAR.npz').write_bytes(b'temporary data fixture')
        output = root / 'output'
        with patch.object(v4, 'source_hash', return_value='source-a'), \
             patch.object(v4, 'load_noisy_pool', return_value=fake_pool()), \
             patch.object(v4, 'image_features', return_value=np.zeros((300, 4))), \
             patch.object(v4, 'texture_features', return_value=np.zeros((300, 4))), \
             patch.object(v4, 'evaluate_seed', side_effect=record_factory):
            result = v4.run(data_dir=data, seeds=seeds, out_dir=output)
        return data, output, result

    def test_singular_estimator_has_serializable_null_condition(self):
        probabilities=np.repeat(np.eye(3), 20, axis=0)
        observed=np.zeros(60, dtype=np.int64)
        with patch.object(v4, 'row_bootstrap_ci', return_value={}):
            result=v4._validated_estimator(probabilities, observed, 0)
        self.assertIsNone(result['condition_number'])
        self.assertEqual(result['T_hat'], [[1., 0., 0.]] * 3)
        json.dumps(result, allow_nan=False)

    def test_summary_preserves_singular_diagnostic_without_invalid_json(self):
        def singular_record(*args):
            record=fake_v4_record(*args)
            for model in record['models'].values():
                model['estimator']={'T_hat':[[1., 0., 0.]] * 3, 'condition_number':None}
            return record
        with tempfile.TemporaryDirectory() as directory:
            data, output, summary=self.make_cache(Path(directory), record_factory=singular_record)
            for model in summary['comparison'].values():
                self.assertIsNone(model['estimated_T_condition_number_mean'])
                self.assertEqual(model['estimated_T_condition_number_nonfinite_runs'], 1)
            parsed=json.loads((output/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(parsed, summary)

    def test_mixed_singular_conditions_do_not_look_like_all_runs_are_finite(self):
        def mixed_record(*args):
            record=fake_v4_record(*args)
            if args[-1] == 0:
                for model in record['models'].values():
                    model['estimator']={'T_hat':[[1.,0.,0.]]*3,'condition_number':None}
            return record
        with tempfile.TemporaryDirectory() as directory:
            _,_,summary=self.make_cache(Path(directory),record_factory=mixed_record,seeds=(0,1))
            for model in summary['comparison'].values():
                self.assertIsNone(model['estimated_T_condition_number_mean'])
                self.assertEqual(model['estimated_T_condition_number_finite_mean'],1.0)
                self.assertEqual(model['estimated_T_condition_number_nonfinite_runs'],1)

    def test_verified_cache_reused_without_optional_feature_work(self):
        with tempfile.TemporaryDirectory() as directory:
            data, output, result = self.make_cache(Path(directory))
            with patch.object(v4, 'source_hash', return_value='source-a'), \
                 patch.object(v4, 'load_noisy_pool', side_effect=AssertionError('must reuse verified cache')):
                repeated = v4.run(data_dir=data, seeds=[0], out_dir=output)
            self.assertEqual(result, repeated)

    def test_changed_data_or_source_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data, output, _ = self.make_cache(Path(directory))
            with patch.object(v4, 'source_hash', return_value='changed-source'):
                with self.assertRaisesRegex(ValueError, 'identity differs'):
                    v4.run(data_dir=data, seeds=[0], out_dir=output)
            (data / 'CIFAR.npz').write_bytes(b'new content at same path')
            with patch.object(v4, 'source_hash', return_value='source-a'):
                with self.assertRaisesRegex(ValueError, 'identity differs'):
                    v4.run(data_dir=data, seeds=[0], out_dir=output)

    def test_tampered_json_or_old_cache_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data, output, _ = self.make_cache(Path(directory))
            (output / 'seed0.json').write_text('{}', encoding='utf-8')
            with patch.object(v4, 'source_hash', return_value='source-a'):
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    v4.run(data_dir=data, seeds=[0], out_dir=output)
            (output / 'seed0.manifest.json').unlink()
            with patch.object(v4, 'source_hash', return_value='source-a'):
                with self.assertRaisesRegex(ValueError, 'no verified provenance'):
                    v4.run(data_dir=data, seeds=[0], out_dir=output)


def gpu_summary_fixture(folder,seed=0,*,source='source-a',data='data-a',width=8,condition=3.0):
    """Small verified file fixture; never represents real experimental evidence."""
    settings={'seed':seed,'epochs':1,'batch_size':64,'width':width,'patience':2,
              'device':'cuda','smoke':False,'torch':'test','numpy':'test','cuda_runtime':'test'}
    identity={'schema_version':1,'study':'cifar_gpu_v5','data_sha256':data,
              'source_sha256':source,'config_sha256':config_hash(settings),
              'shared_config_sha256':config_hash({k:v for k,v in settings.items() if k != 'seed'})}
    model=folder/f'seed{seed}_model.pt'
    checkpoint=folder/f'seed{seed}_checkpoint.pt'
    model.write_bytes(b'unit-test model')
    checkpoint.write_bytes(b'unit-test checkpoint')
    manifest=v5._checkpoint_manifest_path(checkpoint)
    write_json(manifest,{'identity':identity,'artifacts':{checkpoint.name:sha256_file(checkpoint)}})
    record={'status':'completed_full_gpu_diagnostic','seed':seed,'epoch_budget':1,'device':'cuda',
            'identity':identity,'study_config':settings,'data_sha256':data,
            'noisy_audit_accuracy':.4,'noisy_audit_information_gain_nats':.02,
            'estimated_T_condition_number':condition,'pseudo_counts':[100,100,100],
            'estimated_T':np.eye(3).tolist(),'quality_gate':{'eligible_for_downstream_confirmation':condition is not None},
            'artifacts':{p.name:sha256_file(p) for p in (model,checkpoint,manifest)}}
    write_json(folder/f'seed{seed}.json',record)
    return record


class V5SummaryTests(unittest.TestCase):
    def test_verified_seeds_can_differ_only_in_seed_and_extra_manifests_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            gpu_summary_fixture(folder,0)
            gpu_summary_fixture(folder,1)
            write_json(folder/'seed0_smoke.json',{'status':'cpu_smoke_not_for_reporting'})
            result=v5_summary.summarize(folder)
            self.assertEqual(result['count'],2)
            self.assertTrue(result['verified_artifacts'])
            self.assertEqual(result['eligible_count'],2)

    def test_legacy_gpu_records_are_not_newly_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            write_json(folder/'seed0.json',{'status':'completed_full_gpu_diagnostic'})
            with self.assertRaisesRegex(ValueError,'Unverified legacy'):
                v5_summary.summarize(folder)

    def test_tampered_model_rejected_by_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            gpu_summary_fixture(folder)
            (folder/'seed0_model.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Missing or changed artifact'):
                v5_summary.summarize(folder)

    def test_mixed_data_source_or_shared_config_rejected(self):
        for change in ({'data':'other-data'},{'source':'other-source'},{'width':16}):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as directory:
                folder=Path(directory)
                gpu_summary_fixture(folder,0)
                gpu_summary_fixture(folder,1,**change)
                with self.assertRaisesRegex(ValueError,'different data/source/config'):
                    v5_summary.summarize(folder)

    def test_null_condition_failure_remains_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            gpu_summary_fixture(folder,condition=None)
            result=v5_summary.summarize(folder)
            self.assertEqual(result['eligible_count'],0)
            self.assertEqual(result['nonfinite_condition_count'],1)
            self.assertIsNone(result['per_seed'][0]['condition_number'])
            self.assertNotIn('mean_candidate_T_from_eligible_runs',result)
            json.dumps(result,allow_nan=False)

    def test_unverified_v4_baseline_not_used_for_paired_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            baseline=folder/'baseline';baseline.mkdir()
            gpu_summary_fixture(folder)
            write_json(baseline/'seed0.json',fake_v4_record(0))
            with self.assertRaisesRegex(ValueError,'Unverified/mismatched V4'):
                v5_summary.summarize(folder,baseline)


if __name__ == '__main__':
    unittest.main()
