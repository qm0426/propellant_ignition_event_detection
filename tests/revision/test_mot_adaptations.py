import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from PIL import Image
from ignition_revision.common import PROJECT,Blocked
from ignition_revision.mot_adapters import author_variant,prepare_scores,load_author_module
from ignition_revision.tracking import external_mot


class MotAdaptationTests(unittest.TestCase):
    def test_registry_and_explicit_threshold_override(self):
        from ignition_revision.review_v2 import registry,config_for,DEFAULT_SETTINGS
        jobs=registry()
        self.assertIsNone(jobs['MOT_BYTE']['blocked_reason'])
        self.assertIsNone(jobs['MOT_OCSORT']['blocked_reason'])
        with tempfile.TemporaryDirectory() as td:
            data=Path(td);(data/'dataset_meta.json').write_text(json.dumps({'fixed_cohort':2779,'source_hashes':{'boxes':'same'}}))
            settings={**DEFAULT_SETTINGS,'native_mot_overrides':{'MOT_BYTE':{'settings':{'track_thresh':.45}}}}
            variant=config_for(jobs['MOT_BYTE'],settings,data)['variants'][0]
            self.assertEqual(variant['settings']['track_thresh'],.45)
            self.assertEqual(variant['settings']['track_buffer'],5)
            self.assertEqual(variant['score_policy']['kind'],'legacy_quality_proxy')

    def candidates(self,qualities=(1.04,.39,.39)):
        return pd.DataFrame([{'recording_id':'one','frame_idx':i,'candidate_id':f'c{i}',
            'x1':10.+.2*i,'y1':10.,'x2':30.+.2*i,'y2':30.,
            'native_score':q,'score_semantics':'legacy_quality_score'} for i,q in enumerate(qualities)])

    def test_quality_proxy_retains_raw_and_requires_explicit_provenance(self):
        cand=self.candidates();before=cand.copy(deep=True);variant=author_variant('bytetrack')
        scores,audit=prepare_scores(cand,variant)
        np.testing.assert_allclose(scores,[.8,.3,.3]);pd.testing.assert_frame_equal(cand,before)
        self.assertEqual(list(audit.score_group),['high','low','low'])
        self.assertEqual(list(audit.input_score),[1.04,.39,.39])
        self.assertEqual(list(audit.eligible_new_track),[True,False,False])
        cand.loc[0,'native_score']=None
        with self.assertRaises(Blocked):prepare_scores(cand,variant)
        cand=self.candidates();cand['score_semantics']='none'
        with self.assertRaises(Blocked):prepare_scores(cand,variant)

    def test_ocsort_fixed_pool_ignores_quality_and_requires_byte_off(self):
        cand=self.candidates((None,0.,9.));variant=author_variant('ocsort')
        score,_=prepare_scores(cand,variant);np.testing.assert_equal(score,np.ones(3))
        variant['settings']['use_byte']=True
        with self.assertRaisesRegex(ValueError,'use_byte=False'):prepare_scores(cand,variant)
        variant=author_variant('ocsort');variant['settings']['det_thresh']=1.
        with self.assertRaisesRegex(ValueError,'EVERY'):prepare_scores(cand,variant)

    def test_actual_author_bytetrack_keeps_low_score_match_without_starting_low_track(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);Image.new('RGB',(80,80)).save(root/'image.png')
            frames=pd.DataFrame([{'recording_id':'one','frame_idx':i,'image_path':str(root/'image.png')} for i in range(4)])
            cand=self.candidates()
            extra=cand.iloc[[1]].copy();extra['candidate_id']='low_new';extra['x1']=50.;extra['x2']=70.
            cand=pd.concat([cand,extra],ignore_index=True)
            variant=author_variant('bytetrack');cfg={'project_root':str(PROJECT)}
            obs=external_mot(cand,frames,cfg,variant,audit_out=root/'audit')
            self.assertEqual(list(obs.frame_idx),[0,1,2])
            self.assertEqual(obs.track_id.nunique(),1)
            audit=pd.read_csv(root/'audit/native_frame_audit.csv')
            self.assertEqual(list(audit.low_score_updated_tracks),[0,1,1,0])
            self.assertEqual(list(audit.low_boxes),[0,2,1,0])
            meta=json.loads((root/'audit/tracking_effective_config.json').read_text())
            self.assertEqual(meta['effective_lost_buffer_source_frames'],5)
            # A second independent invocation must not inherit track counters.
            again=external_mot(cand,frames,cfg,variant)
            pd.testing.assert_frame_equal(obs,again)

    def test_actual_author_ocsort_accepts_fixed_pool_and_advances_empty_frames(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);Image.new('RGB',(80,80)).save(root/'image.png')
            frames=pd.DataFrame([{'recording_id':'one','frame_idx':i,'image_path':str(root/'image.png')} for i in range(4)])
            cand=self.candidates((None,.01,2.));cfg={'project_root':str(PROJECT)};variant=author_variant('ocsort')
            obs=external_mot(cand,frames,cfg,variant,audit_out=root/'audit')
            self.assertEqual(list(obs.frame_idx),[0,1,2]);self.assertEqual(obs.track_id.nunique(),1)
            audit=pd.read_csv(root/'audit/native_frame_audit.csv')
            self.assertEqual(len(audit),4);self.assertTrue(audit.all_candidates_pass_score_gate.all())
            self.assertFalse(audit.use_byte.any())
            cand['native_score']=.9
            again=external_mot(cand,frames,cfg,variant)
            pd.testing.assert_frame_equal(obs,again)

    def test_bbox_compat_preserves_author_inclusive_pixel_convention(self):
        module=load_author_module({'project_root':str(PROJECT)},author_variant('bytetrack'))
        # Shared pixel column x=1: inclusive overlap is 2 / (4+4-2) = 1/3.
        result=module.matching.ious([[0,0,1,1]],[[1,0,2,1]])
        self.assertAlmostEqual(result[0,0],1/3)
        self.assertEqual(module.matching.ious([],[[0,0,1,1]]).shape,(0,1))


if __name__=='__main__':unittest.main()
