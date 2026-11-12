from __future__ import annotations

import argparse

import copy

import numpy as np

import pandas as pd

from PIL import Image


from .common import boolean, digest, load_legacy, require_columns, resolve, write_json

BOX_COLUMNS=['recording_id','frame_idx','candidate_id','x1','y1','x2','y2','prompt_group','native_score','score_semantics']

def validate_frames(df):
    require_columns(df,['recording_id','frame_idx','image_path'],'frames')
    if df.duplicated(['recording_id','frame_idx']).any():raise ValueError('Duplicate frames')
    if (df.frame_idx<0).any() or (df.frame_idx%1!=0).any():raise ValueError('frame_idx must be 0-based integer')
    return df.sort_values(['recording_id','frame_idx'])

def validate_candidates(df):
    require_columns(df,BOX_COLUMNS,'candidates')
    if df.candidate_id.duplicated().any():raise ValueError('candidate_id must be globally unique')
    if not np.isfinite(df[['x1','y1','x2','y2']].to_numpy(dtype=float)).all():raise ValueError('Nonfinite boxes')
    if ((df.x2<=df.x1)|(df.y2<=df.y1)).any():raise ValueError('Boxes require x2>x1, y2>y1')
    return df

def clean_candidates(raw, frames, config, run, cfg):
    module=load_legacy('postprocess_detection_txt_modified.py',cfg.get('upstream_root'))
    parser=module.build_arg_parser()
    defaults={a.dest:a.default for a in parser._actions if a.dest!='help'}
    unknown=set(config)-set(defaults)
    if unknown:raise ValueError(f'Unknown cleaning settings: {sorted(unknown)}')
    args=argparse.Namespace(**{**defaults,**config})
    all_rows=[]
    for _,f in frames.iterrows():
        subset=raw[(raw.recording_id==f.recording_id)&(raw.frame_idx==f.frame_idx)]
        with Image.open(resolve(cfg,f.image_path)) as source:image=source.convert('RGB')
        working=copy.deepcopy(subset.to_dict('records'))
        for i,box in enumerate(working):
            box.update({'_idx':i,'score':None if pd.isna(box.get('native_score')) else box['native_score']})
            # Legacy quality heuristics remain separate from native_score.
            module.enrich_box_metrics(box,image,args)
            module.apply_rule1_background(box,args)
            module.apply_rule2_geometry(box,*image.size,args)
            module.apply_optional_defocus_filter(box,args)
        module.apply_rule3_nested(working,args)
        if args.enable_duplicate_suppression:module.apply_duplicate_overlap_suppression(working,args)
        if args.enable_nms:module.apply_nms(working,args.nms_iou)
        for box in working:
            box['suppression_parent_id']=None  # legacy does not expose its exact suppression parent
            box['lineage_status']='legacy_reason_only' if not box['keep'] else 'retained'
            box['clean_config_hash']=digest(config)
        all_rows.extend(working)
    write_json(run.out/'cleaning_resolved.json',vars(args))
    audit=pd.DataFrame(all_rows) if all_rows else raw.assign(keep=pd.Series(dtype=bool))
    return audit[audit.keep.map(boolean)].copy(),audit
