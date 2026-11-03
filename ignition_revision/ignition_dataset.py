"""Single-foreground full-frame ignition labels from trusted track/event sources.

Training windows use GT only to sample development images. Inference manifests use
whole track lifetimes and do not reveal onset times. Unknown objects are ignored,
not background; ignored boxes must be handled by both RPN and ROI sampling.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import boolean,file_hash,read_table,require_columns,write_csv,write_json
from .comparison_data import build_dataset,BOX_COLUMNS,tid


def build_ignition_dataset(cfg):
    if cfg.get('trust_tracks_for_training') is not True:raise ValueError('Explicit trust_tracks_for_training=true required for algorithm-box supervision')
    out=Path(cfg['output_dir']).resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'Dataset exists: {out}')
    before=int(cfg.get('pre_frames',20));after=int(cfg.get('post_frames',20));guard=int(cfg.get('onset_guard_frames',0))
    if min(before,after,guard)<0:raise ValueError('Frame windows must be nonnegative')
    source={**cfg,'output_dir':str(out/'source_audit'),'export_rois':False}
    if cfg.get('canonical_source_dir'):
        base=Path(cfg['canonical_source_dir']).resolve()
        for name in ('frames.csv','boxes.csv','all_event_mapping_audit.csv','dataset_meta.json'):
            if not (base/name).is_file():raise ValueError('Incomplete canonical FPN source: '+str(base/name))
    else:base=build_dataset(source)
    frames=read_table(base/'frames.csv');objects=read_table(base/'boxes.csv')
    labels=read_table(base/'all_event_mapping_audit.csv')
    labels=labels.loc[~labels.onset_image_name_conflict.map(boolean)].copy()
    labels=labels.loc[labels.track_id.isin(objects.track_id)].copy()
    known={(r.recording_id,int(r.track_id)):r for r in labels.itertuples()}
    training_keys=set()
    for r in labels.itertuples():
        if boolean(r.has_ignition):training_keys.update((r.recording_id,f) for f in range(max(0,int(r.onset_frame)-before),int(r.onset_frame)+after+1))
    if cfg.get('include_non_event_training_frames',False):
        negatives={(r.recording_id,int(r.track_id)) for r in labels.itertuples() if not boolean(r.has_ignition)}
        training_keys.update((r.recording_id,int(r.frame_idx)) for r in objects.itertuples() if (r.recording_id,int(r.track_id)) in negatives)
    if cfg.get('extra_training_frames'):
        extra=read_table(cfg['extra_training_frames']);require_columns(extra,['recording_id','frame_idx'],'extra training frames')
        training_keys.update(zip(extra.recording_id,extra.frame_idx.astype(int)))
    frame_keys=set(zip(frames.recording_id,frames.frame_idx));training_keys&=frame_keys
    if not training_keys:raise ValueError('No GT-window training frames in selected source scope')
    selected=frames.loc[[k in training_keys for k in zip(frames.recording_id,frames.frame_idx)]].copy()
    # Optional explicit split ranges; no random per-frame split is invented.
    for entry in cfg.get('split_ranges',[]):
        split=entry['split']
        if split not in ('train','validation','test'):raise ValueError('Invalid split name')
        sel=(selected.frame_idx>=int(entry['start']))&(selected.frame_idx<=int(entry['end']))
        if entry.get('recording_id'):sel&=selected.recording_id.eq(entry['recording_id'])
        if selected.loc[sel,'split'].ne('unassigned').any():raise ValueError('Overlapping split ranges/manifest')
        selected.loc[sel,'split']=split
    purged_frames=[]
    if cfg.get('purge_cross_split_tracks',False):
        assignments=selected[['recording_id','frame_idx','split']]
        active=assignments.loc[assignments.split.isin(['train','validation','test'])]
        joined=objects.merge(active,on=['recording_id','frame_idx'],how='inner')
        counts=joined.groupby(['recording_id','track_id']).split.nunique()
        crossed=set(counts[counts>1].index)
        bad=joined.loc[[k in crossed for k in zip(joined.recording_id,joined.track_id)]]
        badkeys=set(zip(bad.recording_id,bad.frame_idx))
        purged_frames=[{'recording_id':r,'frame_idx':int(f),'reason':'track_crosses_active_split'} for r,f in sorted(badkeys)]
        selected=selected.loc[[k not in badkeys for k in zip(selected.recording_id,selected.frame_idx)]].copy()
        training_keys=set(zip(selected.recording_id,selected.frame_idx))
        if selected.empty:raise ValueError('No frames remain after purging cross-split tracks')
    overrides={}
    if cfg.get('state_overrides_csv'):
        table=read_table(cfg['state_overrides_csv']);require_columns(table,['recording_id','track_id','frame_idx','is_ignited'],'state overrides')
        if table.duplicated(['recording_id','track_id','frame_idx']).any():raise ValueError('Duplicate state override')
        for r in table.itertuples():overrides[(r.recording_id,tid(r.track_id),int(r.frame_idx))]=None if pd.isna(r.is_ignited) else boolean(r.is_ignited)
    cold_ids=set(map(tid,cfg.get('confirmed_cold_non_event_tracks',[])))
    positive=[];ignore=[];state_audit=[]
    for r in objects.itertuples():
        if (r.recording_id,int(r.frame_idx)) not in training_keys:continue
        event=known.get((r.recording_id,int(r.track_id)));state=None;reason='unlabelled_track'
        if event is not None and boolean(event.has_ignition):
            onset=int(event.onset_frame)
            if r.frame_idx<onset-guard:state=False;reason='trusted_pre_onset_background'
            elif r.frame_idx>=onset+guard:state=True;reason='trusted_post_onset_particle_persistence'
            else:reason='onset_guard'
        elif r.track_id in cold_ids:state=False;reason='explicit_confirmed_cold_non_event'
        elif event is not None:reason='non_event_not_equivalent_to_cold'
        if r.point_type!='detected':state=None;reason='restoration_not_observed_box'
        key=(r.recording_id,int(r.track_id),int(r.frame_idx))
        if key in overrides:state=overrides[key];reason='explicit_state_override'
        row={k:getattr(r,k) for k in BOX_COLUMNS};row.update({'class_id':1 if state else -1,'review_status':'trusted_pseudo','label_source':reason})
        if state is True:positive.append(row)
        elif state is None:ignore.append(row)
        state_audit.append({'recording_id':r.recording_id,'track_id':r.track_id,'frame_idx':r.frame_idx,'is_ignited':state,'action':'positive' if state is True else 'background' if state is False else 'ignore','reason':reason})
    selected['annotation_status']='trusted_pseudo';selected['boxes_complete']=False;selected['states_complete']=False
    unknown_policy=cfg.get('unknown_policy','ignore_regions')
    if unknown_policy not in ('ignore_regions','exclude_frames'):raise ValueError('unknown_policy must be ignore_regions or exclude_frames')
    if unknown_policy=='exclude_frames':
        bad={(r['recording_id'],r['frame_idx']) for r in ignore}
        selected=selected.loc[[k not in bad for k in zip(selected.recording_id,selected.frame_idx)]].copy()
        keep=set(zip(selected.recording_id,selected.frame_idx));positive=[r for r in positive if (r['recording_id'],r['frame_idx']) in keep];ignore=[]
        if selected.empty:raise ValueError('No fully known training frames; use ignore_regions or provide state overrides')
    all_tracks=objects.copy();all_tracks['cx']=(all_tracks.x1+all_tracks.x2)/2;all_tracks['cy']=(all_tracks.y1+all_tracks.y2)/2
    cohort_keys=set(zip(labels.recording_id,labels.track_id));cohort_tracks=all_tracks.loc[[k in cohort_keys for k in zip(all_tracks.recording_id,all_tracks.track_id)]]
    inference_keys=set(zip(cohort_tracks.recording_id,cohort_tracks.frame_idx))
    inference=frames.loc[[k in inference_keys for k in zip(frames.recording_id,frames.frame_idx)]].copy()
    # Preserve all contemporaneous tracks as competing identities, not just labelled ones.
    shared=all_tracks.loc[[k in inference_keys for k in zip(all_tracks.recording_id,all_tracks.frame_idx)]].copy()
    write_csv(out/'purged_split_frames.csv',purged_frames,['recording_id','frame_idx','reason'])
    write_csv(out/'frames.csv',selected);write_csv(out/'boxes.csv',positive,BOX_COLUMNS);write_csv(out/'ignore_boxes.csv',ignore,BOX_COLUMNS)
    write_csv(out/'state_label_audit.csv',state_audit);write_csv(out/'inference_frames.csv',inference)
    write_csv(out/'shared_tracks.csv',shared);write_csv(out/'cohort.csv',labels[['recording_id','track_id']])
    write_csv(out/'fixed_cohort_labels.csv',labels)
    meta={'schema_version':2,'task':'single_class_ignition','class_names':{'1':'ignited_particle'},'scope':'trusted_algorithm_particle_boxes_and_manual_onset_derived_partial_states','unknown_policy':unknown_policy,'partial_labels':bool(ignore),'frames':len(selected),'split_counts':selected.split.value_counts().to_dict(),'purged_split_frames':len(purged_frames),'positive_boxes':len(positive),'ignored_boxes':len(ignore),'inference_frames':len(inference),'cohort_tracks':len(labels),'config':cfg,'source_dataset':str(base),'source_meta_sha256':file_hash(base/'dataset_meta.json'),'auxiliary_source_hashes':{k:file_hash(Path(cfg[k])) for k in ('state_overrides_csv','extra_training_frames') if cfg.get(k)},'notes':['Boxes describe the tracked particle, not an OH flame envelope.','Training frame selection uses GT; inference selection uses complete available track lifetimes.','Non-event tracks are state-unknown unless explicitly confirmed cold.','Ignoring tracked unknown regions cannot correct untracked missed particles.','Pseudo-label validation AP is partial-scope, not independent full-frame detector accuracy.']}
    write_json(out/'dataset_meta.json',meta)
    print(f'IGNITION_DATASET {out}: {len(selected)} frames, {len(positive)} positive / {len(ignore)} ignored boxes, {len(inference)} full-sequence frames')
    return out


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);args=parser.parse_args(argv)
    build_ignition_dataset(json.loads(Path(args.config).read_text(encoding='utf-8-sig')))
