"""Explicit synthetic fixture for software checks only. Never a real experimental result."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from ignition_revision.common import PROJECT, write_csv, write_json


def build_fixture(root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    frames=[];candidates=[];boxes=[];rois=[]
    for fr in range(24):
        im=Image.new('RGB',(72,64),(80,80,80));draw=ImageDraw.Draw(im)
        draw.ellipse((16,26,24,34),fill=(230,150,100) if fr>=12 else (20,20,20))
        draw.ellipse((46,26,54,34),fill=(25,25,25))
        path=root/'frames'/f'{fr:04d}.png';path.parent.mkdir(exist_ok=True);im.save(path)
        frames.append({'recording_id':'synthetic_r1','frame_idx':fr,'image_path':str(path),'annotated':True})
        for tid,(x1,x2) in enumerate([(15,25),(45,55)],start=1):
            obj='a' if tid==1 else 'b'
            boxes.append({'recording_id':'synthetic_r1','frame_idx':fr,'ref_object_id':obj,'x1':x1,'y1':25,'x2':x2,'y2':35,'cx':(x1+x2)/2,'cy':30,'ignore':False})
            if not (tid==1 and fr in (7,8)):
                candidates.append({'recording_id':'synthetic_r1','frame_idx':fr,'candidate_id':f'synthetic_{fr}_{tid}',
                    'x1':x1,'y1':25,'x2':x2,'y2':35,'prompt_group':'hot' if tid==1 and fr>=12 else 'cold','native_score':None,'score_semantics':'none'})
            rp=root/'rois'/str(tid)/f'{fr:04d}.png';rp.parent.mkdir(parents=True,exist_ok=True);im.crop((x1-3,22,x2+3,38)).save(rp)
            rois.append({'recording_id':'synthetic_r1','track_id':tid,'frame_idx':fr,'roi_path':str(rp)})
    labels=[{'recording_id':'synthetic_r1','track_id':1,'ref_object_id':'a','ref_event_id':'synthetic_event_a','has_ignition':True,'onset_frame':12},
            {'recording_id':'synthetic_r1','track_id':2,'ref_object_id':'b','ref_event_id':'synthetic_object_b','has_ignition':False,'onset_frame':None}]
    refs=[{'recording_id':'synthetic_r1','ref_object_id':'a','ref_event_id':'synthetic_event_a','onset_frame':12}]
    for name,rows in [('frames',frames),('candidates',candidates),('reference_boxes',boxes),('rois',rois),('labels',labels),('reference_events',refs)]:write_csv(root/(name+'.csv'),rows)
    base={'project_root':str(PROJECT),'synthetic':True,'output_dir':str(root/'outputs'),'frames':str(root/'frames.csv'),
          'candidates':str(root/'candidates.csv'),'reference_boxes':str(root/'reference_boxes.csv'),
          'reference_events':str(root/'reference_events.csv'),'rois':str(root/'rois.csv'),'labels':str(root/'labels.csv'),
          'label_provenance':'synthetic_by_construction_NOT_REAL_GT','track_bundle_id':'synthetic_shared_cohort_v1',
          'features':{'difference_policy':'contiguous_zero_boundary','background_correction':True},
          'evaluation':{'iou_threshold':.5,'mapping_max_distance':8,'mapping_min_shared_frames':3,'mapping_min_purity':.8,'tolerances':[1,3],'duration_s':None,'fp_pair_time_tolerance':3,'fp_pair_distance':8},
          'gmm_config':{'gmm_n_init':1},
          'tracking':{'min_hits':2,'min_contiguous_observed':2,'max_center_dist':12,'min_ot_prob':.10},
          'detector_mode':'cached','classical':{'log':{'polarity':'dark','sigmas':[1.,2.],'response_threshold':5.}}}
    gray={'method':'gray','signal_column':'gray_p99','threshold':160.,'min_run':3,'rtv':False}
    methods={'J_GRAY_ADAPTED':gray,'J_GRAY_ADAPTED_RTV':{**gray,'rtv':True},
             'J_GMM':{'method':'gmm','rtv':False},'J_GMM_RTV':{'method':'gmm','rtv':True},
             'J_SAS_ADAPTED':{'method':'sas','signal_column':'gray_p99','threshold':160.,'area_column':'largest_area_ratio','area_threshold':.02,'min_run':3},
             'J_DOG_IGNITION':{'method':'dog','sigmas':[1.,2.],'sigma_ratio':1.6,'polarity':'bright','response_threshold':8.,'max_center_distance':5.,'min_run':3}}
    cfgs={family:dict(base) for family in ('detection','assignment','restoration','judges','ablations')}
    cfgs['judges']['methods']=methods
    cfgs['ablations']['background_factor']={'kind':'independent_columns','columns':['bg_med','bg_mad']}
    for family,cfg in cfgs.items():write_json(root/(family+'.json'),cfg)
    write_json(root/'SYNTHETIC_ONLY.json',{'synthetic':True,'purpose':'software_validation_only','real_measurements':False})
    return cfgs


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    build_fixture(args.output);print(Path(args.output).resolve())
