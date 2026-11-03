"""Independent, reference-event based evaluation; prediction code never imports GT here."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from .common import boolean, digest, require_columns


def optimal_pairs(eligible, residual):
    """Maximum cardinality first, then minimum normalized residual (no greedy matching)."""
    eligible = np.asarray(eligible, dtype=bool)
    residual = np.asarray(residual, dtype=float)
    if eligible.shape != residual.shape or eligible.ndim != 2:
        raise ValueError('Matching matrices must have the same 2-D shape')
    n, m = eligible.shape
    if not n or not m:
        return []
    if not np.isfinite(residual[eligible]).all() or np.any(residual[eligible] < 0):
        raise ValueError('Eligible residuals must be finite and nonnegative')
    scale = max(1.0, float(residual[eligible].max())) if eligible.any() else 1.0
    dummy = n + 1.0
    costs = np.full((n, m + n), dummy)
    costs[:, :m] = np.where(eligible, residual / scale, (n + 1) * dummy)
    a, b = linear_sum_assignment(costs)
    return [(int(i), int(j)) for i, j in zip(a, b) if j < m and eligible[i, j]]


def counts_metrics(tp, fp, fn, tn=None):
    div = lambda a, b: a / b if b else None
    return {'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn) if tn is not None else None,
            'precision': div(tp, tp + fp), 'recall': div(tp, tp + fn),
            'f1': div(2 * tp, 2 * tp + fp + fn), 'false_positive_rate': div(fp, fp + tn) if tn is not None else None}


def onset_metrics(rows):
    errors = [float(r['signed_frame_error']) for r in rows if r.get('signed_frame_error') is not None]
    if not errors:
        return {'localization_N': 0, 'onset_mae': None, 'onset_median_ae': None, 'onset_bias': None,
                'onset_p95_ae': None, 'acc_pm1': None, 'acc_pm3': None}
    e = np.asarray(errors)
    return {'localization_N': len(e), 'onset_mae': float(np.abs(e).mean()), 'onset_median_ae': float(np.median(np.abs(e))),
            'onset_bias': float(e.mean()), 'onset_p95_ae': float(np.quantile(np.abs(e), .95)),
            'acc_pm1': float((np.abs(e) <= 1).mean()), 'acc_pm3': float((np.abs(e) <= 3).mean())}


def evaluate_fixed(predictions, labels, tolerance=None):
    if tolerance is not None and (not np.isfinite(tolerance) or tolerance<0):
        raise ValueError('Event time tolerance must be finite and nonnegative')
    keys = ['recording_id', 'track_id']
    require_columns(labels, keys + ['ref_object_id','has_ignition','onset_frame'], 'fixed-cohort labels')
    require_columns(predictions, keys + ['has_ignition','onset_frame'], 'predictions')
    if labels.duplicated(keys).any() or predictions.duplicated(keys).any():
        raise ValueError('Duplicate fixed-cohort identities')
    extra = predictions.merge(labels[keys], on=keys, how='left', indicator=True)
    if (extra['_merge'] == 'left_only').any():
        raise ValueError('Prediction outside the frozen labelled cohort')
    merged = labels.merge(predictions, on=keys, how='left', suffixes=('_gt','_pred'), validate='one_to_one', indicator=True)
    if (merged['_merge'] != 'both').any():
        raise ValueError('Missing predictions must not silently become no_event/TN')
    rows=[]; tp=fp=fn=tn=0;negative_fp=0;timing_misses=0;existence_errors=[]
    for _, x in merged.iterrows():
        gt, pred = boolean(x.has_ignition_gt), boolean(x.has_ignition_pred)
        status = 'TP' if gt and pred else 'FN' if gt else 'FP' if pred else 'TN'
        tp += status=='TP'; fp += status=='FP'; fn += status=='FN'; tn += status=='TN'
        error = float(x.onset_frame_pred - x.onset_frame_gt) if status=='TP' else None
        if status=='TP' and not np.isfinite(error):
            raise ValueError('Positive predictions/reference require a valid onset')
        ref_id = str(x.get('ref_event_id', f'{x.recording_id}:{x.ref_object_id}:first_onset'))
        negative_fp+=status=='FP'
        if status=='TP':existence_errors.append({'signed_frame_error':error})
        if status=='TP' and tolerance is not None and abs(error)>tolerance:
            tp-=1;fp+=1;fn+=1;timing_misses+=1
            for unmatched in ('FP','FN'):
                rows.append({'recording_id':x.recording_id,'track_id':x.track_id,'ref_event_id':ref_id,
                             'match_status':unmatched,'signed_frame_error':None,
                             'reason':'onset_outside_tolerance','timing_error_frames':error})
            continue
        rows.append({'recording_id':x.recording_id, 'track_id':x.track_id,'ref_event_id':ref_id,
                     'match_status':status,'signed_frame_error':error})
    result=counts_metrics(tp,fp,fn,tn if tolerance is None else None)
    result.update(onset_metrics(rows));result['cohort_hash']=digest(labels.to_dict('records'))
    result.update({'tolerance_frames':tolerance,'timing_misses':timing_misses,
                   'negative_object_FP':int(negative_fp),'true_negative_objects':int(tn),
                   'negative_object_FAR':negative_fp/(negative_fp+tn) if negative_fp+tn else None})
    result.update({'existence_'+k:v for k,v in onset_metrics(existence_errors).items()})
    return result, rows


def map_tracks_to_reference(observations, boxes, max_distance, min_shared_frames=3, min_purity=.8):
    """Many fragments may map to one object. Mixed tracks below purity are not assigned."""
    keys=['recording_id','frame_idx']
    require_columns(observations, keys+['track_id','cx','cy','detection_supported'], 'track observations')
    require_columns(boxes, keys+['ref_object_id','cx','cy'], 'reference boxes')
    candidates=[]; per_frame=[]
    detected=observations[observations.detection_supported.map(boolean)]
    for key, pred in detected.groupby(keys, sort=True):
        ref=boxes[(boxes.recording_id==key[0]) & (boxes.frame_idx==key[1])]
        if ref.empty:continue
        dist=np.linalg.norm(pred[['cx','cy']].to_numpy()[:,None,:]-ref[['cx','cy']].to_numpy()[None,:,:],axis=2)
        for i,j in np.argwhere(dist<=max_distance):
            candidates.append({'recording_id':key[0],'frame_idx':key[1],'track_id':pred.iloc[i].track_id,
                               'ref_object_id':ref.iloc[j].ref_object_id,'distance':float(dist[i,j])})
        for i,j in optimal_pairs(dist<=max_distance,dist):
            per_frame.append({'recording_id':key[0],'frame_idx':key[1],'track_id':pred.iloc[i].track_id,
                              'ref_object_id':ref.iloc[j].ref_object_id,'distance':float(dist[i,j])})
    mapping=[]
    matched=pd.DataFrame(per_frame)
    for (rec,tid), group in detected.groupby(['recording_id','track_id']):
        sub=matched[(matched.recording_id==rec)&(matched.track_id==tid)] if not matched.empty else matched
        counts=sub.ref_object_id.value_counts() if not sub.empty else pd.Series(dtype=int)
        if counts.empty: continue
        obj=counts.index[0]; shared=int(counts.iloc[0]);purity=shared/len(sub)
        ids=sub.sort_values('frame_idx').ref_object_id.tolist()
        switches=sum(a!=b for a,b in zip(ids,ids[1:]))
        mapping.append({'recording_id':rec,'track_id':tid,'ref_object_id':obj if shared>=min_shared_frames and purity>=min_purity else None,
                        'shared_frames':shared,'purity':purity,'reference_id_switches':switches,
                        'mapping_status':'accepted' if shared>=min_shared_frames and purity>=min_purity else 'rejected'})
    return pd.DataFrame(mapping), candidates, per_frame


def evaluate_events(predictions, reference, mapping, tolerance=None, duration_s=None):
    if tolerance is not None and (not np.isfinite(tolerance) or tolerance<0):
        raise ValueError('Event time tolerance must be finite and nonnegative')
    require_columns(reference,['recording_id','ref_event_id','ref_object_id','onset_frame'],'reference events')
    require_columns(predictions,['recording_id','track_id','pred_event_id','onset_frame'],'predicted events')
    if reference.ref_event_id.duplicated().any() or predictions.pred_event_id.duplicated().any():
        raise ValueError('Event IDs must be unique')
    if not mapping.empty:
        require_columns(mapping,['recording_id','track_id','ref_object_id'],'mapping')
        pred=predictions.merge(mapping[['recording_id','track_id','ref_object_id']],on=['recording_id','track_id'],how='left',validate='many_to_one')
    else:
        pred=predictions.copy();pred['ref_object_id']=None
    edges=np.zeros((len(reference),len(pred)),bool);residual=np.zeros(edges.shape)
    residual=np.abs(reference.onset_frame.to_numpy(dtype=float)[:,None]-pred.onset_frame.to_numpy(dtype=float)[None,:])
    if not np.isfinite(residual).all():raise ValueError('Event times must be finite')
    edges=(reference.recording_id.to_numpy()[:,None]==pred.recording_id.to_numpy()[None,:])
    edges &= reference.ref_object_id.astype(str).to_numpy()[:,None]==pred.ref_object_id.astype(str).to_numpy()[None,:]
    edges &= pred.ref_object_id.notna().to_numpy()[None,:]
    if tolerance is not None:edges &= residual<=tolerance
    pairs=optimal_pairs(edges,residual);matched_ref={i for i,j in pairs};matched_pred={j for i,j in pairs}
    rows=[]
    for i,j in pairs:
        gt,pr=reference.iloc[i],pred.iloc[j]
        rows.append({'recording_id':gt.recording_id,'track_id':pr.track_id,'ref_object_id':gt.ref_object_id,'onset_frame':pr.onset_frame,
                     'ref_event_id':gt.ref_event_id,'pred_event_id':pr.pred_event_id,'match_status':'TP',
                     'signed_frame_error':float(pr.onset_frame-gt.onset_frame),'duplicate_flag':False})
    for i in range(len(reference)):
        if i not in matched_ref:rows.append({'recording_id':reference.iloc[i].recording_id,'ref_object_id':reference.iloc[i].ref_object_id,'ref_event_id':reference.iloc[i].ref_event_id,'pred_event_id':None,'match_status':'FN','signed_frame_error':None})
    for j in range(len(pred)):
        if j not in matched_pred:rows.append({'recording_id':pred.iloc[j].recording_id,'track_id':pred.iloc[j].track_id,'ref_object_id':pred.iloc[j].ref_object_id,
                                             'onset_frame':pred.iloc[j].onset_frame,'location_x':pred.iloc[j].get('location_x'),'location_y':pred.iloc[j].get('location_y'),
                                             'ref_event_id':None,'pred_event_id':pred.iloc[j].pred_event_id,'match_status':'FP',
                                             'signed_frame_error':None,'duplicate_flag':bool(edges[:,j].any())})
    metrics=counts_metrics(len(pairs),len(pred)-len(pairs),len(reference)-len(pairs));metrics.update(onset_metrics(rows))
    metrics.update({'tolerance_frames':tolerance,'fp_per_second':metrics['FP']/duration_s if duration_s and duration_s>0 else None})
    return metrics,rows


def common_tp(evaluations):
    """Input method_id -> match rows, with reference event IDs, not algorithm track IDs."""
    members={k:{str(x['ref_event_id']) for x in rows if x['match_status']=='TP'} for k,rows in evaluations.items()}
    intersection=set.intersection(*members.values()) if members else set()
    result=[]
    for method,rows in evaluations.items():
        subset=[x for x in rows if x['match_status']=='TP' and str(x['ref_event_id']) in intersection]
        result.append({'method_id':method,'common_tp_N':len(intersection),'cohort_hash':digest(sorted(intersection)),**onset_metrics(subset)})
    return result,sorted(intersection)


def detector_metrics(candidates, boxes, frames, iou_threshold=.5):
    """Only explicitly annotated frame scope is evaluated. No synthetic confidence/AP."""
    from .detection import iou_matrix
    require_columns(boxes,['recording_id','frame_idx','x1','y1','x2','y2'],'detection GT')
    totals=[0,0,0];rows=[]
    for _,f in frames.iterrows():
        rec,fr=f.recording_id,int(f.frame_idx)
        p=candidates[(candidates.recording_id==rec)&(candidates.frame_idx==fr)]
        g=boxes[(boxes.recording_id==rec)&(boxes.frame_idx==fr)]
        if 'ignore' in g:
            ignored=g[g['ignore'].map(boolean)];g=g[~g['ignore'].map(boolean)]
        else:ignored=g.iloc[:0]
        ov=iou_matrix(g,p);pairs=optimal_pairs(ov>=iou_threshold,1-ov)
        matched={j for i,j in pairs};unmatched=[j for j in range(len(p)) if j not in matched]
        ignored_overlap=iou_matrix(ignored,p)
        fp=sum(not (len(ignored) and (ignored_overlap[:,j]>=iou_threshold).any()) for j in unmatched)
        counts=(len(pairs),fp,len(g)-len(pairs));totals=[a+b for a,b in zip(totals,counts)]
        rows.append({'recording_id':rec,'frame_idx':fr,**counts_metrics(*counts)})
    return counts_metrics(*totals),rows

def detector_strata(candidates,boxes,frames,iou_threshold):
    """Recall by independently annotated GT state/size. FP state is not invented."""
    from .detection import iou_matrix
    members=[]
    for _,f in frames.iterrows():
        pred=candidates[(candidates.recording_id==f.recording_id)&(candidates.frame_idx==f.frame_idx)]
        gt=boxes[(boxes.recording_id==f.recording_id)&(boxes.frame_idx==f.frame_idx)]
        if 'ignore' in gt:gt=gt[~gt['ignore'].map(boolean)]
        ov=iou_matrix(gt,pred);matched={i for i,j in optimal_pairs(ov>=iou_threshold,1-ov)}
        for i,(_,g) in enumerate(gt.iterrows()):
            members.append({'recording_id':f.recording_id,'frame_idx':int(f.frame_idx),'ref_object_id':g.get('ref_object_id'),
                            'state':g.get('state'),'size_bin':g.get('size_bin'),'detected':i in matched})
    table=pd.DataFrame(members);summaries=[]
    for field in ('state','size_bin'):
        if table.empty:continue
        for label,group in table.dropna(subset=[field]).groupby(field):
            tp=int(group.detected.sum());n=len(group)
            summaries.append({'stratum_type':field,'stratum':label,'reference_N':n,'TP':tp,'FN':n-tp,'recall':tp/n,
                              'precision':None,'note':'GT-state/size recall only; unmatched FP cannot be assigned a true object state'})
    return summaries,members
