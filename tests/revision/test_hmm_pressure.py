import itertools,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from ignition_revision.hmm import path_posterior,fit_segment,DEFAULTS,hmm_candidates
from ignition_revision.pressure_pipeline import frame_manifest,run_pipeline
from types import SimpleNamespace
from PIL import Image

class HMMChecks(unittest.TestCase):
    def test_exact_paths_match_brute_force(self):
        rng=np.random.default_rng(41);e=rng.normal(size=(5,2));p=.8;valid=[];scores=[]
        for path in itertools.product([0,1],repeat=5):
            if path[0] or any(path[i]>path[i+1] for i in range(4)):continue
            log=e[np.arange(5),path].sum()
            for i in range(4):
                if path[i]==0:log+=np.log(p if path[i+1]==0 else 1-p)
            valid.append(path);scores.append(log)
        _,ll,gamma=path_posterior(e,p)
        self.assertAlmostEqual(ll,logsumexp(scores),places=12)
        expected=np.exp(scores-logsumexp(scores))@np.array(valid)
        np.testing.assert_allclose(gamma[:,1],expected,atol=1e-12)
    def test_transition_and_constant_null(self):
        x=np.r_[np.zeros((15,3)),np.ones((15,3))*4]
        k,info=fit_segment(x,DEFAULTS);self.assertEqual(k,15);self.assertGreater(info['bic_gain'],0)
        _,info=fit_segment(np.zeros((30,3)),DEFAULTS);self.assertLess(info['bic_gain'],0)
    def test_no_invented_transition_across_gap(self):
        df=pd.DataFrame({'frame_idx':[0,1,2,10,11,12],'gray_p99':[0,0,0,255,255,255]})
        mod=SimpleNamespace(BASE_GMM_FEATURES=['gray_p99'],DIFF_GMM_FEATURES=[],THERMAL_FEATURE_NAMES=['gray_p99'])
        cfg=SimpleNamespace(use_diff_features=False,min_cold_run=3,min_hot_run=3,require_all_prior_cold=True)
        candidates,_=hmm_candidates(df,mod,cfg,{})
        self.assertEqual(candidates,[])
    def test_frames_keep_physical_indices_and_reject_missing(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for n in [41,43]:Image.new('RGB',(20,20)).save(root/f'new_pressure_{n:08d}.png')
            cfg={'project_root':t,'image_dir':t,'recording_id':'P_NEW'}
            with self.assertRaisesRegex(ValueError,'Missing images'):frame_manifest(cfg)
            cfg['require_contiguous_frames']=False
            self.assertEqual(frame_manifest(cfg).frame_idx.tolist(),[41,43])
    def test_original_entry_rejects_hmm(self):
        with self.assertRaisesRegex(ValueError,'only the original'):run_pipeline({'methods':{'HMM':{'method':'hmm'}}})
    def test_duplicate_suffix_diagnostic_and_explicit_selection(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            Image.new('L',(20,20)).save(root/'MCFF0014_00000000.bmp')
            Image.new('RGB',(20,20)).save(root/'MCFF0014_00000000.jpg')
            cfg={'project_root':t,'image_dir':t,'recording_id':'P_NEW'}
            with self.assertRaisesRegex(ValueError,r'Duplicate physical frame 0:.*MCFF0014_00000000'):
                frame_manifest(cfg)
            cfg['image_extensions']=['.bmp']
            manifest=frame_manifest(cfg)
            self.assertEqual(len(manifest),1)
            self.assertTrue(manifest.image_path.iloc[0].endswith('.bmp'))

if __name__=='__main__':unittest.main()
