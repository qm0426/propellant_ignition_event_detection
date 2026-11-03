"""Global ignition detections -> one-to-one shared-track assignment -> first event.

Prediction functions never consume onset GT. All contemporaneous tracks compete,
including objects outside the evaluation cohort. Ambiguous detections are withheld.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .common import Blocked,boolean,file_hash,read_table,require_columns,resolve,write_csv,write_json
from .comparison_data import integral,tid
from .detection import iou_matrix
from .evaluation import optimal_pairs


def normalize_tracks(table,cfg):
    t=table.copy()
    if 'frame_idx' not in t and 'frame_id' in t:t=t.rename(columns={'frame_id':'frame_idx'})
    if 'recording_id' not in t:
        if not cfg.get('recording_id'):raise ValueError('Original tracks.csv requires recording_id in config')
        t['recording_id']=cfg['recording_id']
    require_columns(t,['recording_id','track_id','frame_idx','x1','y1','x2','y2'],'shared tracks')
    t['track_id']=t.track_id.map(tid);t['frame_idx']=integral(t.frame_idx,'track frame')
    if t.duplicated(['recording_id','track_id','frame_idx']).any():raise ValueError('Duplicate shared track frame')
    if not np.isfinite(t[['x1','y1','x2','y2']]).all().all() or ((t.x2<=t.x1)|(t.y2<=t.y1)).any():raise ValueError('Invalid shared track box')
    t['cx']=(t.x1+t.x2)/2;t['cy']=(t.y1+t.y2)/2
    if 'point_type' not in t:t['point_type']='detected'
    return t


def assign_frame(pred,tracks,s):
    if not len(pred):return {},[],[]
    if not len(tracks):return {},[],[{'candidate_id':p.candidate_id,'status':'no_tracks'} for p in pred.itertuples()]
    distance_limit=float(s['max_center_distance_px']);min_iou=float(s['min_iou']);margin=float(s['ambiguity_margin']);ratio_limit=float(s['max_area_ratio'])
    if distance_limit<=0 or not 0<min_iou<=1 or margin<0 or ratio_limit<1:raise ValueError('Invalid matching settings')
    ov=iou_matrix(tracks,pred)
    tc=tracks[['cx','cy']].to_numpy(float);pc=np.column_stack(((pred.x1+pred.x2)/2,(pred.y1+pred.y2)/2))
    dist=np.linalg.norm(tc[:,None,:]-pc[None,:,:],axis=2)
    ta=((tracks.x2-tracks.x1)*(tracks.y2-tracks.y1)).to_numpy(float);pa=((pred.x2-pred.x1)*(pred.y2-pred.y1)).to_numpy(float)
    ratios=np.maximum(ta[:,None]/pa[None,:],pa[None,:]/ta[:,None])
    mode=s.get('gate_mode','iou_or_center')
    if mode=='iou':gate=ov>=min_iou
    elif mode=='iou_or_center':gate=(ov>=min_iou)|(dist<=distance_limit)
    else:raise ValueError('gate_mode must be iou or iou_or_center')
    gate&=ratios<=ratio_limit
    wi=float(s.get('iou_weight',.5));wd=float(s.get('center_weight',.5))
    if wi<0 or wd<0 or wi+wd<=0:raise ValueError('Invalid assignment weights')
    cost=wi*(1-ov)+wd*np.minimum(dist/distance_limit,2.)
    edges=[];reasons={};ambiguous=set()
    for j in range(len(pred)):
        ids=np.flatnonzero(gate[:,j]);ordered=sorted(ids,key=lambda i:(cost[i,j],int(tracks.iloc[i].track_id)))
        if len(ordered)>1 and cost[ordered[1],j]-cost[ordered[0],j]<=margin:ambiguous.add(j)
        for i in ids:edges.append({'candidate_id':pred.iloc[j].candidate_id,'track_id':int(tracks.iloc[i].track_id),'iou':float(ov[i,j]),'center_distance_px':float(dist[i,j]),'area_ratio':float(ratios[i,j]),'cost':float(cost[i,j]),'ambiguous':j in ambiguous})
        reasons[j]='ambiguous_track_assignment' if j in ambiguous else 'unmatched_competition' if len(ids) else 'outside_gate'
    eligible=gate.copy()
    if ambiguous:eligible[:,list(ambiguous)]=False
    matches={};used=set();selected_pairs=set()
    for i,j in optimal_pairs(eligible,cost):
        selected_pairs.add((int(tracks.iloc[i].track_id),pred.iloc[j].candidate_id))
        p=pred.iloc[j];matches[int(tracks.iloc[i].track_id)]={'candidate_id':p.candidate_id,'score':float(p.score),'detection_x1':float(p.x1),'detection_y1':float(p.y1),'detection_x2':float(p.x2),'detection_y2':float(p.y2),'match_iou':float(ov[i,j]),'match_distance_px':float(dist[i,j])};used.add(j)
    for edge in edges:edge['selected']=(edge['track_id'],edge['candidate_id']) in selected_pairs
    unmatched=[{'candidate_id':pred.iloc[j].candidate_id,'status':reasons[j]} for j in range(len(pred)) if j not in used]
    return matches,edges,unmatched


def predict_track_events(predictions,tracks,frames,cohort,settings):
    require_columns(frames,['recording_id','frame_idx'],'processed frames')
    require_columns(cohort,['recording_id','track_id'],'cohort identities')
    if frames.empty or cohort.empty:raise ValueError('Empty processed frame/cohort table')
    integral(frames.frame_idx,'processed frame');integral(predictions.frame_idx,'detection frame')
    if frames.duplicated(['recording_id','frame_idx']).any() or cohort.duplicated(['recording_id','track_id']).any():raise ValueError('Duplicate frames/cohort')
    if predictions.candidate_id.duplicated().any():raise ValueError('Duplicate detection candidate_id')
    coords=predictions[['x1','y1','x2','y2','score']].to_numpy(float)
    if not np.isfinite(coords).all() or ((predictions.x2<=predictions.x1)|(predictions.y2<=predictions.y1)).any() or (~predictions.score.between(0,1)).any():raise ValueError('Invalid predicted box/score')
    cutoff=float(settings['score_threshold']);run_length=int(settings['min_run'])
    if not 0<=cutoff<=1 or run_length<1:raise ValueError('Invalid event threshold/persistence')
    cohort=cohort.copy();cohort['track_id']=cohort.track_id.map(tid)
    if cohort.duplicated(['recording_id','track_id']).any():raise ValueError('Duplicate normalized cohort identity')
    keys=set(zip(cohort.recording_id,cohort.track_id));framekeys=set(zip(frames.recording_id,frames.frame_idx))
    subset=tracks.loc[[k in keys for k in zip(tracks.recording_id,tracks.track_id)]]
    present=set(zip(subset.recording_id,subset.track_id))
    if keys-present:raise ValueError('Cohort contains identities absent from shared tracks')
    if set(zip(subset.recording_id,subset.frame_idx))-framekeys:raise ValueError('Missing inference frames in cohort track lifetime; do not convert unprocessed frames to no_event')
    selected=predictions.loc[(predictions.class_id==int(settings.get('class_id',1)))&(predictions.score>=cutoff)]
    trackgroups={k:g.sort_values('track_id') for k,g in tracks.groupby(['recording_id','frame_idx'])}
    predgroups={k:g.sort_values('candidate_id') for k,g in selected.groupby(['recording_id','frame_idx'])}
    timelines=[];edges=[];unmatched=[];allowed=set(settings.get('allowed_point_types',['detected']))
    if not allowed:raise ValueError('allowed_point_types cannot be empty')
    for r in frames.sort_values(['recording_id','frame_idx']).itertuples():
        key=(r.recording_id,int(r.frame_idx));all_tracks=trackgroups.get(key,tracks.iloc[:0])
        # Restored tracks also compete geometrically, so detections cannot be stolen by nearby observed tracks.
        p=predgroups.get(key,selected.iloc[:0]);matches,e,u=assign_frame(p,all_tracks,settings)
        edges.extend({'recording_id':key[0],'frame_idx':key[1],**x} for x in e);unmatched.extend({'recording_id':key[0],'frame_idx':key[1],**x} for x in u)
        for t in all_tracks.itertuples():
            if (r.recording_id,int(t.track_id)) not in keys:continue
            m=matches.get(int(t.track_id));valid=t.point_type in allowed
            timelines.append({'recording_id':r.recording_id,'track_id':int(t.track_id),'frame_idx':int(r.frame_idx),'point_type':t.point_type,'evidence_valid':valid,'matched':bool(m and valid),'raw_matched_score':m['score'] if m else 0.,'score':m['score'] if m and valid else 0.,'status':'source_excluded' if not valid else 'matched' if m else 'no_match','cx':t.cx,'cy':t.cy,'x1':t.x1,'y1':t.y1,'x2':t.x2,'y2':t.y2,**({k:v for k,v in m.items() if k!='score'} if m else {})})
    timeline=pd.DataFrame(timelines);results=[];audit=[]
    for (rec,k),g in timeline.groupby(['recording_id','track_id']):
        g=g.sort_values('frame_idx').reset_index(drop=True);start=None;candidate_count=0;onset=None;confirmation=None
        for i,row in g.iterrows():
            ok=bool(row.matched) and bool(row.evidence_valid)
            if not ok or (i and row.frame_idx!=g.iloc[i-1].frame_idx+1):start=None
            if ok and start is None:start=i;candidate_count+=1
            if ok and i-start+1>=run_length:
                onset=g.iloc[start];confirmation=row;break
        result={'recording_id':rec,'track_id':int(k),'has_ignition':onset is not None,'onset_frame':int(onset.frame_idx) if onset is not None else None,'confirmation_frame':int(confirmation.frame_idx) if confirmation is not None else None,'status':'accepted' if onset is not None else 'no_event','candidate_count':candidate_count,'future_context':False}
        if onset is not None:
            result.update({'onset_'+name:float(onset[name]) for name in ('cx','cy','x1','y1','x2','y2')})
            result['onset_detection_cx']=(float(onset.detection_x1)+float(onset.detection_x2))/2;result['onset_detection_cy']=(float(onset.detection_y1)+float(onset.detection_y2))/2
            audit.append({'recording_id':rec,'track_id':int(k),'candidate_id':onset.candidate_id,'candidate_onset':int(onset.frame_idx),'confirmation_frame':int(confirmation.frame_idx),'rtv_pass':True,'rtv':{'enabled':False}})
        results.append(result)
    return pd.DataFrame(results),timeline,edges,unmatched,audit


def run_fpn_events(cfg,run):
    tracks=normalize_tracks(read_table(resolve(cfg,cfg['shared_tracks'])),cfg)
    frames=read_table(resolve(cfg,cfg['frames']));cohort=read_table(resolve(cfg,cfg['cohort']))
    predictions=read_table(resolve(cfg,cfg['detections']));require_columns(predictions,['recording_id','frame_idx','candidate_id','class_id','score','x1','y1','x2','y2'],'ignition detections')
    provenance_path=resolve(cfg,cfg['inference_provenance']);import json
    provenance=json.loads(provenance_path.read_text(encoding='utf-8'))
    if provenance['model_provenance']['classes']!={'0':'background','1':'ignited_particle'}:raise ValueError('Shared ignition readout requires a single-class ignited_particle checkpoint')
    processed_path=resolve(cfg,cfg.get('inferred_frames',str(provenance_path.parent/'inferred_frames.csv')))
    if provenance.get('detections_sha256')!=file_hash(resolve(cfg,cfg['detections'])) or provenance.get('inferred_frames_sha256')!=file_hash(processed_path):raise ValueError('Detection/processed-frame files do not match inference provenance')
    processed=read_table(processed_path);available=set(zip(processed.recording_id,processed.frame_idx))
    expected=set(zip(frames.recording_id,frames.frame_idx))
    if expected-available:raise Blocked('Some requested full-frame inference was not completed')
    if cfg.get('labels') and not cfg.get('allow_development_evaluation',False):
        meta=provenance['model_provenance'];seen=set(meta['split']['train_images'].values())|set(meta['split']['validation_images'].values())
        used_records=set(meta.get('train_recording_ids',[]))|set(meta.get('validation_recording_ids',[]))
        eval_records=set(frames.recording_id.astype(str))
        scope=processed.loc[[k in expected for k in zip(processed.recording_id,processed.frame_idx)]]
        if used_records&eval_records or seen&set(scope.image_sha256):raise Blocked('Event evaluation overlaps training/validation recordings or image evidence; explicitly mark development evaluation or provide independent recordings')
    methods=cfg.get('methods',{'FPN_IGNITION_SHARED_TRACKS':{'score_threshold':.5,'min_run':1,'min_iou':.1,'max_center_distance_px':10.,'max_area_ratio':9.,'ambiguity_margin':.1}})
    outputs={}
    export_threshold=provenance.get('export_score_threshold')
    if export_threshold is None:raise ValueError('Inference lacks export threshold provenance; rerun inference with current code')
    for name,settings in methods.items():
        if float(settings['score_threshold'])<float(export_threshold):raise ValueError('Readout threshold is lower than cached detection export threshold; rerun inference at a lower export threshold')
        with run.time('shared_track_readout_'+name,units=len(frames),unit='frames'):
            pred,timeline,edges,unmatched,audit=predict_track_events(predictions,tracks,frames,cohort,settings)
        pred['method_id']=name;outputs[name]=pred;out=run.out/name
        write_csv(out/'predictions.csv',pred);write_csv(out/'track_frame_scores.csv',timeline)
        write_json(out/'assignment_edges.json',edges);write_json(out/'unmatched_detections.json',unmatched);write_json(out/'candidate_audit.json',audit);write_json(out/'method_config.json',settings)
    from .experiments import evaluate_fixed_methods
    evaluation_cfg={**cfg,'evaluation_mode':'shared_tracks_development' if cfg.get('allow_development_evaluation') else cfg.get('evaluation_mode','shared_tracks_independent_recording')}
    score,_=evaluate_fixed_methods(outputs,evaluation_cfg,run);write_csv(run.out/'S06_judge_comparison.csv',score)
    write_json(run.out/'scope.json',{'task':'full_frame_single_class_ignition_with_shared_tracks','evaluation_scope':'development' if cfg.get('allow_development_evaluation') else 'independent_recording','shared_track_identity_is_given':True,'note':'No new tracker is evaluated here. Ambiguous/unmatched detections do not create a new identity. Future_context=false applies to readout; shared tracking itself may be offline.'})


def run_fpn_pipeline(cfg,run):
    from .fpn_detector import infer_fpn
    # One full-image forward per unique frame; all track identities read the cache.
    infer_fpn({**cfg,'evaluate':bool(cfg.get('evaluate_boxes',False))},run)
    readout={**cfg,'detections':str(run.out/'full_frame_detections.csv'),'inference_provenance':str(run.out/'inference_provenance.json'),'inferred_frames':str(run.out/'inferred_frames.csv')}
    return run_fpn_events(readout,run)
