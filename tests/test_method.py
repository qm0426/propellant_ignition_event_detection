"""Implementation checks using generated fixtures, not scientific validation data."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ignition_revision.common import load_config, load_legacy
from ignition_revision.pressure_pipeline import frame_manifest, run_pipeline, PipelineRun
from ignition_revision.tracking import advance_track_state
from ignition_revision.judges import extract_track_features, judge_track, rtv_trace

class MethodTests(unittest.TestCase):
    def fixture(self, directory, empty=False):
        root=Path(directory);images=root/'frames';boxes=root/'boxes'
        images.mkdir();boxes.mkdir()
        for frame in range(18):
            image=Image.new('RGB',(160,160),(140,140,140))
            ImageDraw.Draw(image).ellipse((44+frame,64,54+frame,74),fill=(30,30,30) if frame<9 else (255,180,80))
            stem=f'particle_{frame:06d}'
            image.save(images/(stem+'.png'))
            (boxes/(stem+'.txt')).write_text('' if empty else f'unburned_aluminum_droplet {40+frame} 60 {58+frame} 78\n',encoding='utf-8')
        template=Path(__file__).resolve().parents[1]/'configs/method.cached_txt.json'
        cfg=load_config(template)
        cfg.update(image_dir=str(images),output_dir=str(root/'output'))
        cfg['detector']['txt_dir']=str(boxes)
        return cfg

    def test_full_cached_pipeline_resume_and_coordinates(self):
        with tempfile.TemporaryDirectory(prefix='optical method ') as directory:
            cfg=self.fixture(directory)
            with contextlib.redirect_stdout(io.StringIO()):run_pipeline(cfg)
            out=Path(cfg['output_dir'])
            state=json.loads((out/'pipeline_status.json').read_text())
            self.assertEqual(set(state),{'detect','clean','track','judge'})
            self.assertTrue(all(x['status']=='completed' for x in state.values()))
            events=pd.read_csv(out/'04_judgment/events.csv')
            rois=pd.read_csv(out/'03_tracking/roi_manifest.csv')
            self.assertGreater(len(rois),0)
            self.assertGreater(len(events),0)
            self.assertTrue(rois[['cx','cy','x1','y1','x2','y2']].notna().all().all())
            accepted=events[events.has_ignition]
            self.assertGreater(len(accepted),0)
            self.assertTrue(accepted[['onset_cx','onset_cy']].notna().all().all())
            # Exported original-frame positions are metadata and must not change features.
            cfg2=copy.deepcopy(cfg);cfg2['output_dir']=str(Path(directory)/'scratch')
            a,_,_=extract_track_features(rois,cfg2,PipelineRun(Path(directory)/'fa'))
            b,_,_=extract_track_features(rois.drop(columns=['cx','cy','x1','y1','x2','y2']),cfg2,PipelineRun(Path(directory)/'fb'))
            mod=load_legacy('ignition_detection_gmm_rtv_new.py')
            for key in a:
                np.testing.assert_array_equal(a[key][mod.BASE_GMM_FEATURES+mod.DIFF_GMM_FEATURES],b[key][mod.BASE_GMM_FEATURES+mod.DIFF_GMM_FEATURES])
            with contextlib.redirect_stdout(io.StringIO()):run_pipeline(cfg,resume=True)
            with self.assertRaises(FileExistsError):run_pipeline(cfg)
            changed=copy.deepcopy(cfg);changed['crop_padding']=1
            with self.assertRaisesRegex(ValueError,'changed'):run_pipeline(changed,resume=True)
            (out/'04_judgment/events.csv').write_text('tampered',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Stage output changed'):run_pipeline(cfg,resume=True)

    def test_empty_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=self.fixture(directory,empty=True)
            with contextlib.redirect_stdout(io.StringIO()):run_pipeline(cfg)
            table=pd.read_csv(Path(cfg['output_dir'])/'04_judgment/events.csv')
            self.assertEqual(len(table),0)

    def test_stop_and_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=self.fixture(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                run_pipeline(cfg,stop='detect')
                run_pipeline(cfg,start='clean',resume=True)
            self.assertTrue((Path(cfg['output_dir'])/'04_judgment/events.csv').exists())

    def test_frame_and_detection_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=self.fixture(directory)
            image=Path(cfg['image_dir'])/'other_000000.png'
            Image.new('RGB',(160,160)).save(image)
            with self.assertRaisesRegex(ValueError,'Duplicate physical frame'):frame_manifest(cfg)
            image.unlink()
            (Path(cfg['detector']['txt_dir'])/'particle_000000.txt').unlink()
            with self.assertRaises(FileNotFoundError):run_pipeline(cfg)

    def test_incremental_kalman_time(self):
        class Filter:
            def __init__(self):self.steps=[]
            def predict(self,dt):self.steps.append(dt)
        class Track:pass
        track=Track();track.start_frame=0;track.kf=Filter();track.age=1;track.misses=0
        for frame in (1,2,3,3):advance_track_state(track,frame)
        self.assertEqual(track.kf.steps,[1,1,1])
        with self.assertRaises(ValueError):advance_track_state(track,2)

    def test_rtv_raw_or_and_gaps(self):
        mod=load_legacy('ignition_detection_gmm_rtv_new.py');config=mod.IgnitionGMMConfig()
        from ignition_revision.judges import RTV_FIELDS
        df=pd.DataFrame({'frame_idx':range(6),**{col:[0.]*6 for col,_ in RTV_FIELDS}})
        col,field=RTV_FIELDS[0];df.loc[3:,col]=getattr(config,field)+1
        self.assertTrue(rtv_trace(df,3,config)[0])
        df.loc[3,'frame_idx']=30
        self.assertFalse(rtv_trace(df,3,config)[0])

if __name__=='__main__':unittest.main()
