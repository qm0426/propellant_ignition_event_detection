"""Yang 2026 sec.3.3 criterion with an explicitly substituted K-means-2 mask."""
from concurrent.futures import ThreadPoolExecutor
import math
import re
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import savgol_filter
from .common import digest,read_table,resolve,write_csv,write_json
from .roi_io import read_roi

SETTINGS={'method':'prior9_normint','mask_method':'gray_histogram_weighted_kmeans2',
    'kmeans_max_iter':50,'kmeans_tol':.01,'mask_min_contrast':5.,'baseline_fraction':.1,
    'baseline_rounding':'ceil','intensity_threshold':140.,'cold_factor':.8,'min_samples':16,
    'sg_window':11,'sg_polyorder':3,'sg_mode':'interp','normalized_threshold':.5,'persistence':5,
    'workers':4,'save_masks':True,'rtv':False,
    'source_doi':'10.1016/j.combustflame.2026.115140',
    'reproduction_level':'paper_sec3.3_criterion_Kmeans2_mask_substitution_shared_tracks',
    'ineligible_policy':'no_event_for_full_cohort_and_separate_eligibility_report',
    'parameter_provenance':'Paper sec3.3 parameters; declared Kmeans mask, ceil rounding and gap-safe SG implementation conventions; no GT tuning'}


def kmeans_mask(gray,cx,cy,s):
    """Exact histogram-weighted Lloyd updates on 8-bit grayscale (not Otsu).

    Fixed min/max initialization, no seed or GT-driven hot/cold class selection.
    The known particle center chooses a cluster; its 8-connected component is kept.
    """
    a=np.asarray(gray,dtype=np.uint8);h,w=a.shape
    if not np.isfinite([cx,cy]).all() or not (0<=cx<w and 0<=cy<h):raise ValueError('Known ROI particle center required')
    hist=np.bincount(a.ravel(),minlength=256).astype(float);occupied=np.flatnonzero(hist)
    if len(occupied)<2:return np.zeros_like(a,bool),{'valid':False,'flags':['constant_roi'],'centers':[]}
    centers=np.array([occupied[0],occupied[-1]],float);levels=np.arange(256)
    for iteration in range(int(s['kmeans_max_iter'])):
        labels=np.argmin(abs(levels[:,None]-centers[None,:]),axis=1)
        new=np.array([np.sum(levels[labels==k]*hist[labels==k])/np.sum(hist[labels==k]) for k in range(2)])
        if np.max(abs(new-centers))<=s['kmeans_tol']:centers=new;break
        centers=new
    labels=np.argmin(abs(levels[:,None]-centers[None,:]),axis=1)[a]
    x=int(np.clip(round(cx),0,w-1));y=int(np.clip(round(cy),0,h-1));cluster=int(labels[y,x])
    _,components=cv2.connectedComponents((labels==cluster).astype('uint8'),connectivity=8)
    mask=components==components[y,x];contrast=float(abs(centers[1]-centers[0]));flags=[]
    if contrast<s['mask_min_contrast']:flags.append('low_cluster_contrast')
    if mask.mean()>.8:flags.append('high_mask_occupancy_check_background')
    if mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any():flags.append('mask_touches_roi_boundary')
    return mask,{'valid':bool(mask.any() and contrast>=s['mask_min_contrast']),
        'centers':centers.tolist(),'contrast':contrast,'selected_cluster':cluster,'iterations':iteration+1,'flags':flags}


def normint_decision(values,frames,s):
    g=np.asarray(values,float);frames=np.asarray(frames,int);n=len(g)
    if n!=len(frames) or len(set(frames.tolist()))!=n or (n>1 and (np.diff(frames)<=0).any()):raise ValueError('Sorted unique physical frames required')
    out={'has_ignition':False,'onset_frame':None,'confirmation_frame':None,'status':'too_short',
         'eligible':False,'g0':None,'g1':None,'onset_idx':None,'t20_frame':None,'t80_frame':None,
         'tau_rise_equivalent_frames':None,'tau_valid_for_statistics':False}
    smooth=np.full(n,np.nan);normalized=np.full(n,np.nan)
    if n<s['min_samples']:return out,normalized,smooth
    count=max(1,int(math.ceil(s['baseline_fraction']*n)))
    if not np.isfinite(g[:count]).all() or not np.isfinite(g[-count:]).all():out['status']='invalid_baseline_mask';return out,normalized,smooth
    g0=float(np.median(g[:count]));g1=float(np.median(g[-count:]));out.update(g0=g0,g1=g1,baseline_samples=count)
    if not (g0<s['cold_factor']*s['intensity_threshold'] and g1>s['intensity_threshold']):out['status']='cold_hot_gate_rejected';return out,normalized,smooth
    out['eligible']=True;normalized=np.clip((g-g0)/(g1-g0),0,1)
    # Do not compress time across missing masks/frames or silently interpolate.
    blocks=[];start=None
    for i in range(n+1):
        valid=i<n and np.isfinite(g[i]);continuous=valid and (i==0 or frames[i]==frames[i-1]+1)
        if start is not None and (not valid or not continuous):blocks.append((start,i));start=None
        if valid and start is None:start=i
    for lo,hi in blocks:
        if hi-lo>=s['sg_window']:smooth[lo:hi]=savgol_filter(normalized[lo:hi],s['sg_window'],s['sg_polyorder'],mode=s['sg_mode'])
    out['status']='no_persistent_crossing';run=0
    for i in range(n):
        ok=np.isfinite(smooth[i]) and smooth[i]>s['normalized_threshold']
        run=(run+1 if i and frames[i]==frames[i-1]+1 else 1) if ok else 0
        if run>=s['persistence']:
            onset=i-s['persistence']+1
            out.update(has_ignition=True,onset_frame=int(frames[onset]),confirmation_frame=int(frames[i]),
                       onset_idx=onset,status='accepted');break
    def crossing(threshold):
        for lo,hi in blocks:
            if hi-lo<s['sg_window']:continue
            if smooth[lo]>=threshold:return float(frames[lo]),(lo,hi),True
            for i in range(lo+1,hi):
                if smooth[i-1]<threshold<=smooth[i]:
                    t=frames[i-1]+(threshold-smooth[i-1])/(smooth[i]-smooth[i-1])
                    return float(t),(lo,hi),False
        return None,None,False
    t20,b20,left20=crossing(.2);t80,b80,left80=crossing(.8)
    out.update(t20_frame=t20,t80_frame=t80)
    if t20 is not None and t80 is not None and t80>=t20 and b20==b80:
        out['tau_rise_equivalent_frames']=(t80-t20)/.6
        out['tau_valid_for_statistics']=bool(out['has_ignition'] and not left20 and not left80 and t20<=out['onset_frame']<=t80)
    return out,normalized,smooth


def process_track(df,method,s,cfg,root):
    df=df.sort_values('frame_idx').reset_index(drop=True);rec=str(df.recording_id.iloc[0]);tid=df.track_id.iloc[0]
    safe=lambda x:re.sub(r'[^a-zA-Z0-9_-]','_',str(x))
    folder=root/'masks'/safe(rec)/('track_'+safe(tid));records=[];values=[]
    for _,row in df.iterrows():
        rgb=read_roi(row,cfg);a=np.asarray(Image.fromarray(rgb).convert('L'),dtype=np.uint8)
        cx=float(row.get('center_roi_x',np.nan));cy=float(row.get('center_roi_y',np.nan))
        mask,info=kmeans_mask(a,cx,cy,s);area=int(mask.sum());signal=float(np.median(a[mask])) if info['valid'] else np.nan
        path=folder/f"frame_{int(row['frame_idx']):08d}.png"
        if s['save_masks']:
            path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray((mask*255).astype('uint8')).save(path)
        yy,xx=np.nonzero(mask)
        mx=float(xx.mean()) if area else np.nan;my=float(yy.mean()) if area else np.nan
        records.append({'recording_id':rec,'track_id':tid,'frame_idx':int(row['frame_idx']),
            'roi_path':row['roi_path'],'mask_path':str(path) if s['save_masks'] else None,
            'g_mask_median_255':signal,'mask_valid':info['valid'],'mask_area_px2':area,
            'mask_equivalent_diameter_px':math.sqrt(4*area/math.pi),
            'mask_centroid_roi_x':mx,'mask_centroid_roi_y':my,
            'mask_centroid_global_x':float(row['cx'])+mx-cx,'mask_centroid_global_y':float(row['cy'])+my-cy,
            'cluster_low':info['centers'][0] if info['centers'] else None,
            'cluster_high':info['centers'][1] if info['centers'] else None,
            'mask_flags':';'.join(info['flags'])});values.append(signal)
    decision,normalized,smooth=normint_decision(values,df.frame_idx.to_numpy(int),s)
    for i,row in enumerate(records):row.update(g_normalized=normalized[i],g_smoothed=smooth[i])
    out={'recording_id':rec,'track_id':tid,'method_id':method,
         **{k:decision[k] for k in ['has_ignition','onset_frame','confirmation_frame','status']}}
    if out['has_ignition']:
        row=df.iloc[decision['onset_idx']]
        for key in ['cx','cy','x1','y1','x2','y2']:out['onset_'+key]=float(row[key])
    return out,{'recording_id':rec,'track_id':tid,**decision,'N':len(df),
                'invalid_mask_frames':sum(not r['mask_valid'] for r in records)},records


def run_prior9(cfg,run):
    """Direct ROI path: do not recompute unrelated GMM feature bundles."""
    from .experiments import evaluate_fixed_methods
    rois=read_table(resolve(cfg,cfg['rois']))
    if rois.duplicated(['recording_id','track_id','frame_idx']).any():raise ValueError('Duplicate ROI observations')
    outputs={}
    cv2.setNumThreads(1)
    for method,settings in cfg['methods'].items():
        s={**SETTINGS,**settings};root=run.out/method;root.mkdir(parents=True,exist_ok=True)
        if s['rtv']:raise ValueError('Prior9 reproduction does not silently add RTV')
        predictions=[];diagnostics=[];measurements=[]
        groups=[df for _,df in rois.groupby(['recording_id','track_id'],sort=True)]
        with run.time('kmeans_masks_and_prior9',len(rois),'roi_frames'):
            with ThreadPoolExecutor(max_workers=int(s['workers'])) as pool:
                for i,(pred,trace,rows) in enumerate(pool.map(lambda df:process_track(df,method,s,cfg,root),groups),1):
                    predictions.append(pred);diagnostics.append(trace);measurements.extend(rows)
                    if i%100==0 or i==len(groups):print(f'PRIOR9_KMEANS {i}/{len(groups)} tracks; {len(measurements)} ROI frames',flush=True)
        table=pd.DataFrame(predictions);outputs[method]=table
        write_csv(root/'predictions.csv',table);write_csv(root/'eligibility_and_rise.csv',diagnostics)
        write_csv(root/'mask_signal_manifest.csv',measurements);write_json(root/'method_config.json',s)
        write_json(root/'eligibility_summary.json',{'tracks':len(groups),'paper_gate_eligible':sum(d['eligible'] for d in diagnostics),
            'status_counts':pd.Series([d['status'] for d in diagnostics]).value_counts().to_dict(),
            'full_cohort_policy':s['ineligible_policy'],'SAM_reproduced':False})
    metrics,matches=evaluate_fixed_methods(outputs,cfg,run)
    write_csv(run.out/'S06_judge_comparison.csv',metrics)
    write_csv(run.out/'onset_method_registry.csv',[{'method_id':k,'implementation':'prior9_normint',
        'reproduction_level':v.get('reproduction_level',SETTINGS['reproduction_level']),'input_kind':'ROI_images_Kmeans2_masks','requires_training':False} for k,v in cfg['methods'].items()])
