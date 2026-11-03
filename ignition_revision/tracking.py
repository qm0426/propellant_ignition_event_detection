from __future__ import annotations
from dataclasses import asdict
import copy
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from .common import Blocked, file_hash, load_legacy, require_columns, resolve, write_csv, write_json
from .detection import validate_frames, validate_candidates

OBS_COLUMNS=['recording_id','track_id','frame_idx','x1','y1','x2','y2','cx','cy','source','detection_id','detection_supported','image_evidence','position_reconstructed']


def solve_association(mod,cost,settings,backend):
    n,m=cost.shape
    if not n or not m:return [],{'backend':backend,'dummy_choices':n+m}
    augmented=mod.append_dustbins(cost,float(settings['dummy_cost']))
    if backend=='hungarian':
        a,b=linear_sum_assignment(augmented)
        matches=[(int(i),int(j),0.,float(cost[i,j])) for i,j in zip(a,b) if i<n and j<m and cost[i,j]<=settings['max_match_cost']]
        return matches,{'backend':backend,'dummy_choices':sum(i>=n or j>=m for i,j in zip(a,b))}
    if backend=='ot':
        p=mod.sinkhorn_transport(augmented,reg=settings['ot_reg'],num_iters=settings['ot_iters'])
        target=1/len(augmented)
        error=max(float(np.max(np.abs(p.sum(0)-target))),float(np.max(np.abs(p.sum(1)-target))))
        if not np.isfinite(p).all() or error>settings.get('sinkhorn_marginal_tolerance',.01):
            raise RuntimeError(f'Sinkhorn did not converge: max marginal residual={error}')
        return mod.extract_matches_from_transport(p,cost,n,m,settings['min_ot_prob'],settings['max_match_cost']),{'backend':backend,'marginal_error':error}
    raise ValueError(backend)


TIME_ADVANCE_POLICY='incremental_state_frame_v1'


def advance_track_state(track,frame_id):
    """Advance from the FILTER state timestamp, retaining last_frame as last detection.

    Calling the legacy predict on every missing frame repeatedly integrated the
    interval from the last detection (1+2+3 instead of 1+1+1).
    """
    previous=getattr(track,'_revision_state_frame',track.start_frame)
    elapsed=int(frame_id)-int(previous)
    if elapsed<0:raise ValueError('Filter state cannot advance backwards')
    if not elapsed:return
    track.kf.predict(dt=elapsed)
    track._revision_state_frame=int(frame_id)
    track.age+=elapsed
    track.misses+=elapsed


def make_tracker(mod,backend,rescue,settings,time_advance_policy=TIME_ADVANCE_POLICY):
    if time_advance_policy not in (TIME_ADVANCE_POLICY,'legacy_last_detection'):
        raise ValueError('Unknown filter time advancement policy')
    class RevisionTracker(mod.OTKalmanTracker):
        """Per-instance policies; no replacement of shared legacy module functions."""
        def __init__(self):
            super().__init__(**settings)
            self.trace=[]
        def step(self,frame_id,detections):
            detections=[d for d in detections if d['score']>=self.min_det_score]
            for tr in self.active_tracks:
                if time_advance_policy==TIME_ADVANCE_POLICY:advance_track_state(tr,frame_id)
                else:tr.predict(frame_id)
            n,m=len(self.active_tracks),len(detections);matches=[];diag={}
            if n and m:
                cost,_=mod.build_cost_matrix(self.active_tracks,detections,self.max_center_dist,self.max_mahalanobis,
                                            self.w_maha,self.w_center,self.w_iou,self.w_size,self.w_direction,self.min_direction_speed)
                matches,diag=solve_association(mod,cost,vars(self),backend)
            used_t=set();used_d=set()
            def update(ti,di,prob,c,source):
                tr=self.active_tracks[ti];tr.update(detections[di],frame_id,ot_prob=prob,match_cost=c)
                tr.history[-1].update({'source':source,'detection_id':detections[di]['candidate_id']})
                self._maybe_confirm(tr)
                if tr.is_confirmed:tr.track_state='confirmed'
                used_t.add(ti);used_d.add(di)
                self.trace.append({'frame_idx':frame_id,'track_id':tr.track_id,'candidate_id':detections[di]['candidate_id'],'source':source,'cost':c,'confidence':prob})
            for ti,di,p,c in matches:update(ti,di,p,c,'detector_primary')
            rescued=[]
            if rescue:
                rescued,_=mod.rescue_unmatched_matches(self.active_tracks,detections,[i for i in range(n) if i not in used_t],
                    [j for j in range(m) if j not in used_d],self.max_center_dist,self.max_mahalanobis,
                    self.max_age_confirmed,self.rescue_center_factor,self.rescue_maha_factor)
            for ti,di,p,c in rescued:update(ti,di,p,c,'detector_rescue')
            new=[]
            for j,det in enumerate(detections):
                if j not in used_d:
                    tr=self._new_track(det,frame_id);tr.history[-1].update({'source':'detector_primary','detection_id':det['candidate_id']});new.append(tr)
            alive=[]
            for i,tr in enumerate(self.active_tracks):
                if i in used_t:alive.append(tr);continue
                age=self.max_age_confirmed if tr.is_confirmed else self.max_age_tentative
                if tr.misses>age:
                    tr.track_state='finished';tr.end_reason=mod.classify_track_end(tr,self.image_width,self.image_height,self.border_margin);self.finished_tracks.append(tr)
                else:
                    tr.track_state='lost' if tr.is_confirmed else 'tentative';alive.append(tr)
            self.active_tracks=alive+new
            return {'matches':len(matches),'rescue_matches':len(rescued),**diag}
    return RevisionTracker()


def tracking_settings(mod,overrides):
    defaults={name:p.default for name,p in inspect.signature(mod.OTKalmanTracker).parameters.items() if p.default is not inspect.Parameter.empty}
    unknown=set(overrides)-set(defaults)
    if unknown:raise ValueError(f'Unknown tracker parameters {unknown}')
    return {**defaults,**overrides}


def candidate_detections(frame):
    result=[]
    for _,x in frame.iterrows():
        d=x.to_dict();d.update({'cx':(x.x1+x.x2)/2,'cy':(x.y1+x.y2)/2,'w':x.x2-x.x1,'h':x.y2-x.y1,
                               'score':float(x.native_score) if pd.notna(x.native_score) and x.score_semantics in ('calibrated_confidence','legacy_quality_score') else 1.})
        # 1.0 is only the legacy no-score inclusion sentinel, never exported as native confidence.
        result.append(d)
    return result


def run_custom(candidates,frames,cfg,variant,run):
    mod=load_legacy('ot_kalman_tracker.py',cfg.get('legacy_root'))
    settings=tracking_settings(mod,cfg.get('tracking',{}));all_obs=[];trace=[]
    policy=cfg.get('time_advance_policy',TIME_ADVANCE_POLICY)
    write_json(run.out/variant['id']/'tracking_effective_config.json',{'tracker':settings,'backend':variant.get('backend','ot'),'rescue':variant.get('rescue',True),'restoration':variant.get('restoration',{}),'time_advance_policy':policy})
    for rec,frame_rows in frames.groupby('recording_id',sort=True):
        tracker=make_tracker(mod,variant.get('backend','ot'),variant.get('rescue',True),settings,policy)
        group=candidates[candidates.recording_id==rec]
        grouped={int(fr):candidate_detections(g) for fr,g in group.groupby('frame_idx',sort=False)}
        for fr in range(int(frame_rows.frame_idx.min()),int(frame_rows.frame_idx.max())+1):
            with run.time('online_association_'+variant['id'],unit='frames'):
                tracker.step(fr,grouped.get(fr,[]))
        tracks=tracker.finish()
        restoration=variant.get('restoration',{})
        if restoration.get('gap',False):
            with run.time('gap_'+variant['id'],len(tracks)):
                tracks,_=mod.postprocess_tracks(tracks,mod.kf_params_from_tracker(tracker),enable_gap_fill=True,
                                               max_gap_fill=cfg.get('restoration',{}).get('max_gap_fill',10),
                                               min_contiguous_observed=1,min_observed_hits=1)
        if restoration.get('prebirth',False):
            frame_map={int(f.frame_idx):resolve(cfg,f.image_path) for _,f in frame_rows.iterrows()}
            with run.time('prebirth_'+variant['id'],len(tracks)):
                tracks,_=mod.recover_prebirth_dark_particles(tracks,frame_map,**cfg.get('restoration',{}).get('prebirth_settings',{}))
        for tr in tracks:
            for h in sorted(tr.history,key=lambda h:h['frame_id']):
                point=h.get('point_type','detected')
                source=h.get('source','gap_roi' if point in ('candidate','gap_filled') else 'prebirth_roi' if 'prebirth' in point else 'detector_primary')
                detected=source in ('detector_primary','detector_rescue')
                all_obs.append({'recording_id':rec,'track_id':tr.track_id,'frame_idx':int(h['frame_id']),
                                **{k:float(h[k]) for k in ('x1','y1','x2','y2','cx','cy')},'source':source,
                                'detection_id':h.get('detection_id'),'detection_supported':detected,'image_evidence':True,
                                'position_reconstructed':not detected})
        trace.extend({'recording_id':rec,**x} for x in tracker.trace)
    return pd.DataFrame(all_obs,columns=OBS_COLUMNS),trace


def external_mot(candidates,frames,cfg,variant,audit_out=None):
    method=variant['backend'];settings=variant.get('settings',{})
    from .mot_adapters import prepare_scores,load_author_module
    scores,score_audit=prepare_scores(candidates,variant)
    source=resolve(cfg,variant.get('source_file'))
    if not source.is_file() or not variant.get('source_sha256'):raise Blocked('Native MOT requires author source_file and verified source_sha256')
    if file_hash(source)!=variant['source_sha256']:raise ValueError('External MOT source hash mismatch')
    roots=[source.parent,resolve(cfg,variant.get('source_root',str(source.parent)))]
    for root in roots:
        if str(root) not in sys.path:sys.path.insert(0,str(root))
    if method in ('bytetrack','ocsort'):
        module=load_author_module(cfg,variant)
    else:
        spec=importlib.util.spec_from_file_location('_revision_native_'+method,source);module=importlib.util.module_from_spec(spec)
        try:spec.loader.exec_module(module)
        except ImportError as e:raise Blocked(f'{method} dependency unavailable: {e}') from e
    work=candidates.copy();work['_tracker_score']=scores
    grouped={(rec,int(fr)):group for (rec,fr),group in work.groupby(['recording_id','frame_idx'],sort=False)}
    outputs=[];frame_audit=[];dimensions={}
    for rec,ff in frames.groupby('recording_id'):
        if method=='sort':tracker=module.Sort(**settings)
        elif method=='bytetrack':
            if variant.get('frame_rate') is None:raise Blocked('ByteTrack frame_rate must be explicitly configured')
            tracker=module.BYTETracker(SimpleNamespace(**settings),frame_rate=variant['frame_rate'])
        elif method=='ocsort':tracker=module.OCSort(**settings)
        else:raise ValueError(method)
        if method!='sort':
            from PIL import Image
            with Image.open(resolve(cfg,ff.iloc[0].image_path)) as image:width,height=image.size
            for key,actual in [('image_height',height),('image_width',width)]:
                if key in variant and int(variant[key])!=actual:raise ValueError('Configured native image dimensions differ from original frames')
            dimensions[str(rec)]={'width':width,'height':height}
            # Keep counters per recording, independent of earlier jobs.
            if method=='bytetrack':
                from_module=sys.modules[module.STrack.__module__]
                from_module.BaseTrack._count=0
        for fr in range(int(ff.frame_idx.min()),int(ff.frame_idx.max())+1):
            rows=grouped.get((rec,fr),work.iloc[:0]);score=rows['_tracker_score'].to_numpy(float)
            det=np.column_stack([rows[['x1','y1','x2','y2']].to_numpy(float),score])
            if method=='sort':native=tracker.update(det)
            else:
                dims=[height,width];native=tracker.update(det,dims,dims)
            item={'recording_id':rec,'frame_idx':fr,'input_boxes':len(rows),'output_tracks':len(native)}
            if method=='bytetrack':
                threshold=float(settings['track_thresh'])
                item.update({'high_boxes':int((score>threshold).sum()),'low_boxes':int(((score>.1)&(score<threshold)).sum()),
                    'ignored_boxes':int((~((score>threshold)|((score>.1)&(score<threshold)))).sum()),
                    'low_score_updated_tracks':sum(t.frame_id==tracker.frame_id and t.score<threshold for t in tracker.tracked_stracks)})
                native=[[t.tlwh[0],t.tlwh[1],t.tlwh[0]+t.tlwh[2],t.tlwh[1]+t.tlwh[3],t.track_id] for t in native]
            elif method=='ocsort':item.update({'all_candidates_pass_score_gate':bool((score>settings['det_thresh']).all()),'use_byte':settings['use_byte']})
            frame_audit.append(item)
            for x in native:
                outputs.append({'recording_id':rec,'track_id':int(x[4]),'frame_idx':fr,'x1':x[0],'y1':x[1],'x2':x[2],'y2':x[3],
                                'cx':(x[0]+x[2])/2,'cy':(x[1]+x[3])/2,'source':'native_mot_output','detection_id':None,
                                'detection_supported':False,'image_evidence':False,'position_reconstructed':False})
    if audit_out is not None:
        write_csv(Path(audit_out)/'tracker_input_score_audit.csv',score_audit)
        write_csv(Path(audit_out)/'native_frame_audit.csv',frame_audit)
        effective={'backend':method,'source_revision':variant.get('source_revision'),'source_sha256':variant['source_sha256'],
            'settings':settings,'score_policy':variant.get('score_policy',{}),'comparison_scope':variant.get('comparison_scope'),
            'image_dimensions':dimensions,'source_candidate_pool_hash':file_hash(resolve(cfg,cfg['candidates'])) if cfg.get('candidates') else None,
            'frame_rate':variant.get('frame_rate'),'frame_rate_meaning':variant.get('frame_rate_meaning'),
            'input_candidates':len(candidates),'input_score_semantics':list(candidates.score_semantics.unique()),
            'acquisition_fps':50000,'native_observation_provenance':'Original author-emitted boxes; original input candidate ID not reconstructed'}
        if method=='bytetrack':effective.update({'high_score_threshold':settings['track_thresh'],'low_score_threshold_exclusive':.1,
            'new_track_threshold_inclusive':settings['track_thresh']+.1,
            'effective_lost_buffer_source_frames':tracker.max_time_lost,
            'quality_scores_above_scale':int((pd.to_numeric(candidates.native_score)>variant.get('score_policy',{}).get('scale',float('inf'))).sum()),
            'total_low_score_updates':sum(x['low_score_updated_tracks'] for x in frame_audit)})
        write_json(Path(audit_out)/'tracking_effective_config.json',effective)
    return pd.DataFrame(outputs,columns=OBS_COLUMNS)


def crop_observations(obs,frames,cfg,out):
    from PIL import Image
    lookup={(f.recording_id,int(f.frame_idx)):f.image_path for _,f in frames.iterrows()};rows=[];padding=cfg.get('crop_padding',4)
    if cfg.get('virtual_tracking_rois'):
        dimensions={}
        for _,o in obs.iterrows():
            key=(o.recording_id,int(o.frame_idx))
            if key not in lookup:raise Blocked(f'No original image for {key}')
            image=resolve(cfg,lookup[key])
            # All frames in a recording must share these dimensions; bounds are
            # still validated by read_roi against each actual image on use.
            if o.recording_id not in dimensions:
                with Image.open(image) as im:dimensions[o.recording_id]=im.size
            width,height=dimensions[o.recording_id]
            left=max(0,int(np.floor(o.x1-padding)));top=max(0,int(np.floor(o.y1-padding)))
            right=min(width,int(np.ceil(o.x2+padding)));bottom=min(height,int(np.ceil(o.y2+padding)))
            if right<=left or bottom<=top:raise ValueError('Invalid virtual tracking crop')
            rows.append({'recording_id':o.recording_id,'track_id':o.track_id,'frame_idx':int(o.frame_idx),
                         'roi_path':str(image),'source':o.source,'crop_left':left,'crop_top':top,
                         'crop_right_exclusive':right,'crop_bottom_exclusive':bottom,
                         'center_roi_x':o.cx-left,'center_roi_y':o.cy-top})
        return pd.DataFrame(rows,columns=['recording_id','track_id','frame_idx','roi_path','source','crop_left','crop_top',
                                         'crop_right_exclusive','crop_bottom_exclusive','center_roi_x','center_roi_y'])
    for _,o in obs.iterrows():
        key=(o.recording_id,int(o.frame_idx))
        if key not in lookup:raise Blocked(f'No original image for reconstructed observation {key}')
        with Image.open(resolve(cfg,lookup[key])) as im:
            x1=max(0,int(np.floor(o.x1-padding)));y1=max(0,int(np.floor(o.y1-padding)))
            x2=min(im.width,int(np.ceil(o.x2+padding)));y2=min(im.height,int(np.ceil(o.y2+padding)))
            if x2<=x1 or y2<=y1:raise ValueError('Invalid/out-of-image reconstructed ROI')
            path=out/'rois'/str(o.recording_id)/str(o.track_id)/(str(int(o.frame_idx))+'.png');path.parent.mkdir(parents=True,exist_ok=True)
            im.crop((x1,y1,x2,y2)).save(path)
        rows.append({'recording_id':o.recording_id,'track_id':o.track_id,'frame_idx':int(o.frame_idx),'roi_path':str(path),'source':o.source})
    return pd.DataFrame(rows,columns=['recording_id','track_id','frame_idx','roi_path','source'])


def assignment_variants():
    return [{'id':f'ASSIGN_{s}_R{r}','backend':backend,'rescue':bool(r)} for s,backend in [('H','hungarian'),('O','ot')] for r in (0,1)]


def restoration_variants():
    # T0 is an explicit NN baseline, not a renamed Hungarian/OT run.
    return [{'id':'REST_T0','backend':'nn','rescue':False},
            *[{'id':name,'backend':'ot','rescue':r,'restoration':{'gap':g,'prebirth':p}}
              for name,r,g,p in [('REST_T1',False,False,False),('REST_T2',True,False,False),
              ('REST_T2B',True,True,False),('REST_FULL',True,True,True),
              ('REST_NO_RESCUE',False,True,True),('REST_NO_GAP',True,False,True),('REST_NO_PREBIRTH',True,True,False)]]]

def run_center_nn(candidates,frames,cfg,variant,run):
    helper=load_legacy('run_tracking_ablation_experiments_gmm_rtv_effective_trace_positive_gt_v2.py',cfg.get('legacy_root'))
    mod=load_legacy('ot_kalman_tracker.py',cfg.get('legacy_root'));settings=tracking_settings(mod,cfg.get('tracking',{}))
    rows=[]
    for rec,ff in frames.groupby('recording_id'):
        group=candidates[candidates.recording_id==rec]
        dets={fr:candidate_detections(group[group.frame_idx==fr]) for fr in range(int(ff.frame_idx.min()),int(ff.frame_idx.max())+1)}
        with run.time('center_nn',len(dets),'frames'):
            tracks,_=helper.run_center_nn_tracker(dets,settings['max_center_dist'],settings['max_age'],settings['max_age_tentative'],settings['max_age_confirmed'],settings['min_hits'])
        for tr in tracks:
            for h in tr.history:
                fr=h['frame_id'];same=group[(group.frame_idx==fr)&np.isclose(group.x1,h['x1'])&np.isclose(group.y1,h['y1'])&np.isclose(group.x2,h['x2'])&np.isclose(group.y2,h['y2'])]
                rows.append({'recording_id':rec,'track_id':tr.track_id,'frame_idx':fr,**{k:h[k] for k in ('x1','y1','x2','y2','cx','cy')},
                             'source':'detector_primary','detection_id':same.iloc[0].candidate_id if len(same)==1 else None,
                             'detection_supported':True,'image_evidence':True,'position_reconstructed':False})
    return pd.DataFrame(rows,columns=OBS_COLUMNS),[]

