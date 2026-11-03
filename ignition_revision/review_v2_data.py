"""Full-scope canonical inputs from the user's immutable corrected MOT revision."""
from pathlib import Path
import re
import pandas as pd
from .common import PROJECT, Blocked, boolean, digest, file_hash, write_csv, write_json

REC='MCFF0004_N001'


def prepare(settings):
    root=Path(settings['project_root']); material=root/'GT_review_materials_20261025'
    clean=material/'15_MOT_reference_cleaned_20261029'
    policy=settings['event_label_policy']
    if policy not in ('historical_implicit_negatives','reviewed_only'):
        raise ValueError('Choose explicit event_label_policy')
    out=root/settings['input_root']/policy
    sources={'boxes':clean/'统一身份逐帧框.csv','events':clean/'点火事件修订对应表_原392条.csv',
             'known_labels':root/'fpn_ignition_dataset/fixed_cohort_labels.csv'}
    fingerprints={k:file_hash(v) for k,v in sources.items()}
    marker=out/'dataset_meta.json'
    if marker.exists():
        import json
        old=json.loads(marker.read_text(encoding='utf-8'))
        if old['source_hashes']!=fingerprints or old['event_label_policy']!=policy:
            raise ValueError('Input revision changed; choose a new input_root')
        for name,h in old['output_hashes'].items():
            if not (out/name).is_file() or file_hash(out/name)!=h:raise ValueError('Prepared input modified: '+name)
        return out
    if out.exists() and any(out.iterdir()):raise FileExistsError('Incomplete input directory: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    tracks=pd.read_csv(sources['boxes'],low_memory=False)
    events=pd.read_csv(sources['events'])
    known=pd.read_csv(sources['known_labels'])
    if len(tracks)!=128688 or tracks.duplicated(['human_id','frame_id']).any():raise ValueError('Unexpected corrected reference')
    frames=pd.DataFrame({'recording_id':REC,'frame_idx':range(7997)})
    frames['image_path']=[str((material/'02_full_frames/view_png'/f'frame_{f:08d}.png').resolve()) for f in frames.frame_idx]
    if not all(Path(p).is_file() for p in frames.image_path):raise Blocked('Original full-resolution frames missing')
    frames['annotated']=True
    write_csv(out/'frames.csv',frames)
    boxes=tracks.rename(columns={'frame_id':'frame_idx','track_id':'source_track_id'}).copy()
    boxes['recording_id']=REC;boxes['track_id']=boxes.human_id.astype(int)
    boxes['ref_object_id']=boxes.track_id.map(lambda k:f'particle_{k}')
    boxes['cx']=(boxes.x1+boxes.x2)/2;boxes['cy']=(boxes.y1+boxes.y2)/2
    boxes['ignore']=False;boxes['state']='unknown'
    write_csv(out/'reference_boxes.csv',boxes)
    retained=events.loc[events.status.eq('retained_original_global_onset')].copy()
    if retained.human_id.duplicated().any():raise Blocked('Multiple event labels per human identity require adjudication')
    positive={int(r.human_id):int(r.revised_onset_global_0based) for r in retained.itertuples()}
    original_negative=set(known.loc[~known.has_ignition.map(boolean),'track_id'].astype(int))
    # A reviewed no-event original ID is transferred only if its surviving rows
    # belong to exactly one identity; ambiguous transfers are not guessed.
    negatives=set()
    for tid in original_negative:
        ids=tracks.loc[tracks.track_id.eq(tid),'human_id'].unique()
        if len(ids)==1:negatives.add(int(ids[0]))
    if negatives&set(positive):raise ValueError('Conflicting positive/negative reference labels')
    labels=[]
    for hid in sorted(boxes.track_id.unique()):
        if hid not in positive and hid not in negatives and policy=='reviewed_only':continue
        is_positive=int(hid) in positive
        labels.append({'recording_id':REC,'track_id':int(hid),'ref_object_id':f'particle_{hid}',
                       'ref_event_id':f'{REC}:particle_{hid}:first_onset','has_ignition':is_positive,
                       'onset_frame':positive.get(int(hid)),
                       'label_status':'manual_positive' if is_positive else 'reviewed_no_event' if hid in negatives else 'historical_implicit_negative',
                       'label_provenance':'corrected identity and original global onset; explicit historical-negative assumption' if policy=='historical_implicit_negatives' else 'positive onset or explicitly reviewed non-event only'})
    labels=pd.DataFrame(labels)
    write_csv(out/'labels.csv',labels)
    write_csv(out/'reference_events.csv',labels.loc[labels.has_ignition].drop(columns=['has_ignition','track_id']))
    # Use the original exact stored crop for unconflicted rows; virtual crops read
    # the correct physical frame for inherited image-name conflicts.
    bounds=[]
    for tid in sorted(tracks.track_id.unique()):
        path=material/'03_original_tracks'/f'track_{tid}'/'roi_crop_bounds.csv'
        if not path.is_file():raise Blocked('Original crop bounds missing: '+str(path))
        bounds.append(pd.read_csv(path))
    bb=pd.concat(bounds,ignore_index=True).rename(columns={'track_id':'source_track_id','global_frame_id':'frame_idx'})
    rois=boxes.merge(bb,on=['source_track_id','frame_idx'],how='left',validate='one_to_one')
    if rois.roi_original.isna().any():raise ValueError('Missing original ROI geometry')
    conflicts=[];rows=[]
    for r in rois.itertuples():
        frame=int(r.frame_idx)
        conflict=Path(str(r.image_name)).stem!=f'MCFF0004-N001_{frame:08d}'
        if conflict:
            image=frames.iloc[frame].image_path
            crop={k:int(getattr(r,k)) for k in ['crop_left','crop_top','crop_right_exclusive','crop_bottom_exclusive']}
            conflicts.append({'track_id':r.track_id,'source_track_id':r.source_track_id,'frame_idx':frame,'reason':'read_correct_physical_frame_instead_of_inherited_image_name'})
        else:
            image=str((material/str(r.roi_original)).resolve());crop={}
            if not Path(image).is_file():raise Blocked('Original ROI missing: '+image)
        rows.append({'recording_id':REC,'track_id':int(r.track_id),'source_track_id':int(r.source_track_id),
                     'frame_idx':frame,'roi_path':image,'cx':r.cx,'cy':r.cy,'x1':r.x1,'y1':r.y1,'x2':r.x2,'y2':r.y2,
                     'center_roi_x':r.cx-r.crop_left,'center_roi_y':r.cy-r.crop_top,
                     'source':r.point_type,**crop})
    all_rois=pd.DataFrame(rows)
    write_csv(out/'rois_all.csv',all_rois)
    write_csv(out/'rois.csv',all_rois.loc[all_rois.track_id.isin(labels.track_id)])
    write_csv(out/'physical_frame_corrections.csv',conflicts)
    # Full cleaned detector pool, never filter candidate boxes by corrected GT.
    directory=Path(settings['cleaned_detections_dir'])
    if not directory.is_absolute():directory=root/directory
    from .common import load_legacy
    mod=load_legacy('ot_kalman_tracker.py');dets=mod.load_detections_txt_dir(directory,max_txt=-1)
    candidates=[]
    # Legacy loader enumerates sorted files; prove filenames really denote 0..7996.
    paths=sorted(directory.glob('*.txt'))
    if len(paths)<7997:raise Blocked('Fewer than 7997 full-scope detection files')
    for frame,path in enumerate(paths[:7997]):
        if not path.stem.endswith(f'_{frame:08d}'):raise ValueError('Detection filename/global frame mismatch')
        for j,box in enumerate(dets.get(frame,[])):
            candidates.append({'recording_id':REC,'frame_idx':frame,'candidate_id':f'{REC}:{frame}:{j}',
                               **{k:box[k] for k in ['x1','y1','x2','y2']},'prompt_group':'original_cleaned_pool',
                               'native_score':box['score'],'score_semantics':'legacy_quality_score'})
    write_csv(out/'candidates.csv',candidates)
    # Detector experiment uses the independent 300-frame box task, not MOT boxes.
    detroot=root/'revision_outputs/open_vocabulary_review_20261029'
    detframes=pd.read_csv(detroot/'reference/frames.csv')
    detframes['image_path']=detframes.frame_idx.map(frames.set_index('frame_idx').image_path)
    write_csv(out/'detector_frames.csv',detframes)
    detboxes=pd.read_csv(detroot/'reference/reference_boxes.csv');write_csv(out/'detector_boxes.csv',detboxes)
    raw=pd.read_csv(detroot/'BOTH_PROMPTS_RAW/candidates.csv')
    raw['native_score']=None;raw['score_semantics']='none';write_csv(out/'detector_candidates.csv',raw)
    # Corrected canonical object draft for rebuilding single-class FPN with the
    # existing builder; no old, rejected positive boxes are carried forward.
    fullframes=frames.assign(width=1280,height=896,split='unassigned',boxes_complete=False,states_complete=False,annotation_status='trusted_pseudo')
    write_csv(out/'fpn_source/frames.csv',fullframes)
    obj=boxes.assign(box_id=[f'{REC}:{f}:{t}' for f,t in zip(boxes.frame_idx,boxes.track_id)],class_id=-1,
                     review_status='trusted_pseudo',label_source='algorithm_assisted_user_corrected_box')
    write_csv(out/'fpn_source/boxes.csv',obj)
    event_draft=labels.copy();event_draft['onset_image_name_conflict']=False
    event_draft['label_source']=event_draft.label_status
    write_csv(out/'fpn_source/all_event_mapping_audit.csv',event_draft)
    write_json(out/'fpn_source/dataset_meta.json',{'source':'user corrected revision 15','partial_labels':True,'physical_frame_reconstruction':True,'label_policy':policy})
    outputs={str(f.relative_to(out)):file_hash(f) for f in out.rglob('*.csv')}
    outputs['fpn_source/dataset_meta.json']=file_hash(out/'fpn_source/dataset_meta.json')
    write_json(marker,{'schema_version':2,'source_hashes':fingerprints,'event_label_policy':policy,
                       'frames':len(frames),'reference_boxes':len(boxes),'reference_ids':boxes.track_id.nunique(),
                       'fixed_cohort':len(labels),'positive_events':int(labels.has_ignition.sum()),
                       'explicit_negative_events':len(negatives),'implicit_negative_events':int(labels.label_status.eq('historical_implicit_negative').sum()),
                       'detector_frames':len(detframes),'detector_boxes':len(detboxes),'candidate_boxes':len(candidates),
                       'full_scope_no_fixture_no_frame_cap':True,'output_hashes':outputs,
                       'limitations':['historical implicit negatives are an explicit old-protocol assumption, not newly independent no-event annotation',
                                      'single physical recording, retrospective comparisons, not leave-one-recording-out',
                                      'correct physical frames used; original 7 onset image conflicts remain a provenance caveat, not silently adjudicated annotation certainty']})
    return out
