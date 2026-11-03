from __future__ import annotations
from dataclasses import asdict, replace
import numpy as np
import pandas as pd
from PIL import Image
from .common import Blocked, digest, load_legacy, read_table, require_columns, resolve, write_json

RTV_FIELDS=[('gray_p99','min_raw_delta_gray_p99'),('corr_p99','min_raw_delta_corr_p99'),
            ('bright_energy_mean','min_raw_delta_bright_energy'),('bright_area_ratio','min_raw_delta_bright_area'),
            ('hsv_v_p99','min_raw_delta_hsv_v_p99')]


def roi_manifest_digest(rois):
    """Fingerprint actual observations, independent of their input row order."""
    import hashlib
    frame=rois.copy()
    for key in ('recording_id','track_id'):frame[key]=frame[key].astype(str)
    frame['frame_idx']=frame.frame_idx.astype('int64')
    frame=frame.sort_values(['recording_id','track_id','frame_idx']).reindex(columns=sorted(frame.columns))
    payload=frame.to_csv(index=False,float_format='%.17g',na_rep='<NA>',lineterminator='\n')
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def extract_track_features(rois,cfg,run):
    mod=load_legacy('ignition_detection_gmm_rtv_new.py',cfg.get('legacy_root'))
    config=mod.IgnitionGMMConfig(**cfg.get('gmm_config',{}))
    write_json(run.out/'gmm_effective_config.json',asdict(config))
    policy=cfg.get('features',{}).get('difference_policy','contiguous_zero_boundary')
    if policy!='contiguous_zero_boundary':raise ValueError('Supported revision difference policy: contiguous_zero_boundary; legacy reproduction remains separate')
    write_json(run.out/'feature_policy.json',{'difference_policy':policy,'normalization':'whole_track_median_MAD_clip6_offline'})
    require_columns(rois,['recording_id','track_id','frame_idx','roi_path'],'ROI manifest')
    if rois.duplicated(['recording_id','track_id','frame_idx']).any():raise ValueError('Duplicate track frame')
    if rois[['recording_id','track_id','frame_idx','roi_path']].isna().any().any():raise ValueError('ROI identity/frame/path must not be missing')
    if (rois.frame_idx<0).any() or (rois.frame_idx%1!=0).any():raise ValueError('ROI frame_idx must be 0-based global integers')
    # Runtime-generated tracking ROIs are NOT cfg['rois'] (the fixed GT table).
    # Include the actual queue, its identities and exact crop geometry. Equal
    # row counts alone cannot distinguish an identity switch or shifted crop.
    manifest_hash=roi_manifest_digest(rois)
    validation_columns=[c for c in ('recording_id','track_id','frame_idx','roi_path',
        'crop_left','crop_top','crop_right_exclusive','crop_bottom_exclusive',
        'center_roi_x','center_roi_y') if c in rois]
    observation_hash=roi_manifest_digest(rois[validation_columns])
    cache=None
    cache_state={'schema_version':2,'roi_manifest_sha256':manifest_hash,
                 'roi_observations_sha256':observation_hash,'rows':len(rois)}
    if cfg.get('feature_cache_dir') and hasattr(run,'inputs'):
        from .common import file_hash
        key=digest(['actual_roi_manifest_v2',manifest_hash,run.inputs,cfg.get('features',{}),asdict(config),file_hash(__file__),file_hash(mod.__file__)])
        cache=resolve(cfg,cfg['feature_cache_dir'])/(key+'.csv')
        cache_state.update({'path':str(cache),'key':key})
        if cache.exists() and cache.with_suffix('.json').exists():
            import json
            info=json.loads(cache.with_suffix('.json').read_text(encoding='utf-8'))
            valid=(info.get('schema_version')==2 and info.get('roi_manifest_sha256')==manifest_hash
                   and info.get('rows')==len(rois) and file_hash(cache)==info.get('sha256'))
            if valid:
                table=pd.read_csv(cache,low_memory=False,float_precision='round_trip',
                    dtype={c:rois[c].dtype for c in ('recording_id','track_id')})
                valid=(len(table)==len(rois) and set(validation_columns)<=set(table.columns)
                       and roi_manifest_digest(table[validation_columns])==observation_hash)
            if valid:
                write_json(run.out/'feature_cache_status.json',{'status':'verified_hit',**cache_state})
                return {(rec,tid):g.sort_values('frame_idx').reset_index(drop=True) for (rec,tid),g in table.groupby(['recording_id','track_id'])},config,mod
            # Preserve the rejected artifact, then regenerate instead of
            # aborting the entire experiment because a derived cache is bad.
            import shutil
            from uuid import uuid4
            quarantine=cache.parent/'rejected'/uuid4().hex;quarantine.mkdir(parents=True)
            shutil.copy2(cache,quarantine/cache.name);shutil.copy2(cache.with_suffix('.json'),quarantine/cache.with_suffix('.json').name)
            cache_state.update({'rejected_cache':str(quarantine),'rebuild_reason':'schema/checksum/actual ROI identity or crop mismatch'})
        write_json(run.out/'feature_cache_status.json',{'status':'extracting',**cache_state})
    rows=[]
    order=['recording_id','frame_idx','track_id'] if 'crop_left' in rois else ['recording_id','track_id','frame_idx']
    for progress,(_,r) in enumerate(rois.sort_values(order).iterrows(),1):
        from .roi_io import read_roi
        rgb=read_roi(r,cfg)
        if cfg.get('features',{}).get('background_correction',True) is not True:
            raise Blocked('background_correction=false needs an independently specified feature formula; not a column deletion')
        with run.time('raw_features',unit='roi_frames'):
            feat=mod.extract_frame_features(rgb,config)
        h,w=rgb.shape[:2]
        for key,value in r.items():
            if key not in feat:feat[key]=value
        feat.update({'recording_id':r.recording_id,'track_id':r.track_id,'frame_idx':int(r.frame_idx),
                     'roi_width':w,'roi_height':h,'roi_area':w*h,'roi_aspect_ratio':w/max(h,1),'roi_path':r.roi_path})
        rows.append(feat)
        if len(rois)>10000 and (progress%10000==0 or progress==len(rois)):
            print(f'FEATURE_EXTRACTION {progress}/{len(rois)} ROI frames',flush=True)
    full=pd.DataFrame(rows)
    result={}
    if full.empty:return result,config,mod
    for key,df in full.groupby(['recording_id','track_id'],sort=True):
        df=df.sort_values('frame_idx').reset_index(drop=True)
        contiguous=df.frame_idx.diff().eq(1)
        for col in [c for c in df if pd.api.types.is_numeric_dtype(df[c]) and c not in ('frame_idx','track_id')]:
            if col in ['gray_p95','gray_p99','gray_max','corr_p95','corr_p99','corr_max','bright_area_ratio','bright_energy_mean','bright_mean','red_ratio','rg_ratio','hsv_v_p95','hsv_v_p99','largest_area_ratio']:
                df['d1_'+col]=df[col].diff().where(contiguous,0.).fillna(0.)
                df['d2_'+col]=df['d1_'+col].diff().where(contiguous & contiguous.shift(fill_value=False),0.).fillna(0.)
        df['difference_valid']=contiguous
        result[key]=df
    if cache is not None:
        from .common import file_hash
        cache.parent.mkdir(parents=True,exist_ok=True)
        from uuid import uuid4
        temporary=cache.with_name(cache.stem+'.'+uuid4().hex+'.tmp.csv')
        pd.concat(result.values(),ignore_index=True).to_csv(temporary,index=False)
        temporary.replace(cache)
        metadata=cache.with_name(cache.stem+'.'+uuid4().hex+'.tmp.json')
        write_json(metadata,{**cache_state,'sha256':file_hash(cache)})
        metadata.replace(cache.with_suffix('.json'))
        write_json(run.out/'feature_cache_status.json',{'status':'created',**cache_state})
    return result,config,mod


def gmm_candidates(df,mod,config,settings):
    columns=settings['feature_columns'] if 'feature_columns' in settings else list(mod.BASE_GMM_FEATURES)+(list(mod.DIFF_GMM_FEATURES) if config.use_diff_features else [])
    if columns is None:raise Blocked('Explicit feature_columns has not been resolved')
    remove=settings.get('remove_columns',[])
    if set(remove)-set(columns):raise ValueError(f'No-op deletion: {set(remove)-set(columns)} are not in model input')
    columns=[c for c in columns if c not in remove]
    require_columns(df,columns,'feature view')
    if not columns or not set(columns)&set(mod.THERMAL_FEATURE_NAMES):raise ValueError('No valid named thermal-order features')
    x=df[columns].to_numpy(float)
    if not np.isfinite(x).all():raise ValueError('Nonfinite raw features')
    med=np.median(x,axis=0);mad=np.median(np.abs(x-med),axis=0)+1e-6
    x=np.clip((x-med)/mad,-6,6).astype(np.float32)
    model,k,info,models=mod.fit_gmm_models_bic(x,max_states=config.gmm_max_states,covariance_type=config.gmm_covariance_type,
                    reg_covar=config.gmm_reg_covar,n_init=config.gmm_n_init,bic_improvement_ratio=config.gmm_bic_improvement_ratio,random_state=config.random_state)
    model,k,info=mod.select_gmm_with_occupancy_validation(x,models,info,min_state_count=config.gmm_min_state_count,min_state_fraction=config.gmm_min_state_fraction)
    trace={'feature_columns':columns,'feature_schema_hash':digest(columns),'selected_k':k,'gmm_info':info}
    if k<2 or info.get('occupancy_valid') is False:return [],{**trace,'reason':'single_or_unstable_state'}
    thermal=mod.compute_state_thermal_scores(model,columns);mapping,_=mod.build_state_rank_mapping(thermal)
    raw=np.asarray([mapping[int(v)] for v in model.predict(x)])
    smooth=mod.smooth_state_sequence_by_run_length(raw,config.min_state_run)
    frames=df.frame_idx.to_numpy(int);candidates=[]
    for i in range(config.min_cold_run,len(df)-config.min_hot_run+1):
        left=i-config.min_cold_run;right=i+config.min_hot_run
        if not np.all(np.diff(frames[left:right])==1):continue
        if config.require_all_prior_cold and np.any(smooth[:left]!=0):continue
        if np.all(smooth[left:i]==0) and np.all(smooth[i:right]>0):candidates.append(i)
    return candidates,{**trace,'raw_rank':raw.tolist(),'smoothed_rank':smooth.tolist(),'state_thermal_scores':thermal,
                       'reason':None if candidates else 'no_persistent_cold_hot_transition'}


def rtv_trace(df,index,config,rule='or',minimum_votes=None):
    a=index-config.min_cold_run;b=index+config.min_hot_run
    if a<0 or b>len(df) or not np.all(np.diff(df.frame_idx.iloc[a:b])==1):
        return False,{'reason':'insufficient_contiguous_rtv_support'}
    channels=[]
    for col,field in RTV_FIELDS:
        pre=df[col].iloc[a:index].to_numpy(float);post=df[col].iloc[index:b].to_numpy(float)
        delta=float(post.mean()-pre.mean());threshold=float(getattr(config,field))
        channels.append({'channel':col,'pre_n':len(pre),'post_n':len(post),'pre_mean':float(pre.mean()),
                         'post_mean':float(post.mean()),'delta':delta,'threshold':threshold,'pass':delta>=threshold})
    votes=sum(x['pass'] for x in channels)
    if rule=='or':passed=votes>=1
    elif rule=='k_of_m':
        if minimum_votes is None or not 1<=minimum_votes<=len(channels):raise ValueError('Set valid minimum_votes')
        passed=votes>=minimum_votes
    else:raise ValueError('rtv_rule must be or or k_of_m')
    return passed,{'channels':channels,'votes':votes,'rule':rule,'minimum_votes':1 if rule=='or' else minimum_votes}


def earliest_runs(mask,frames,min_run):
    candidates=[];start=None
    for i,ok in enumerate(mask):
        if not ok or (i and frames[i]!=frames[i-1]+1):
            if start is not None and i-start>=min_run:candidates.append(start)
            start=None
        if ok and start is None:start=i
    if start is not None and len(mask)-start>=min_run:candidates.append(start)
    return candidates


def propose(df,method,settings,mod,config,cfg):
    frames=df.frame_idx.to_numpy(int)
    if method=='hmm':
        from .hmm import hmm_candidates
        return hmm_candidates(df,mod,config,settings)
    if method in ('sas_paper','dog_paper','cnf_roi_stage'):
        from .paper_judges import propose_paper
        return propose_paper(df,method,settings,cfg)
    if method=='gmm':
        if config.refine_onset_with_change:raise Blocked('Onset refinement is not supported by the separated revision policy; preserve false or reproduce the legacy entry point')
        return gmm_candidates(df,mod,config,settings)
    if method=='legacy_baseline':
        if not settings.get('legacy_method') or not settings.get('params'): raise Blocked('Legacy baseline requires its original method name and fully resolved parameters')
        old=load_legacy('run_ignition_ablation_experiments_gmm_rtv_precheck_ablation_aligned.py',cfg.get('legacy_root'))
        result=old.run_method(settings['legacy_method'],df,mod,config,settings['params'])
        index=int(result.get('ignition_idx_onset',-1))
        return ([index] if result.get('has_ignition') and index>=0 else []),{'legacy_result':result}
    if method in ('gray','change','sas'):
        for k in ('signal_column','threshold','min_run'):
            if k not in settings or settings[k] is None:raise Blocked(f'{method} requires development-selected {k}')
        require_columns(df,[settings['signal_column']],'judge signal')
        signal=df[settings['signal_column']].to_numpy(float)
        if method=='change':
            # Explicit adapted adjacent-frame score, not an alias for the old multi-feature baseline.
            signal=np.r_[0,np.diff(signal)];signal[np.r_[False,np.diff(frames)!=1]]=0
        mask=signal>=settings['threshold'];trace={'signal':signal.tolist()}
        if method=='sas':
            for k in ('area_column','area_threshold'):
                if k not in settings or settings[k] is None:raise Blocked(f'SAS requires {k}')
            require_columns(df,[settings['area_column']],'SAS area proxy')
            area=df[settings['area_column']].to_numpy(float);mask&=area>=settings['area_threshold'];trace['area']=area.tolist()
        return earliest_runs(mask,frames,int(settings['min_run'])),trace
    if method=='dog':
        import cv2
        for k in ('sigmas','sigma_ratio','response_threshold','polarity','max_center_distance','min_run'):
            if k not in settings or settings[k] is None:raise Blocked(f'DoG-ignition requires {k}')
        if settings['polarity'] not in ('bright','dark') or settings['sigma_ratio']<=1:raise ValueError('Invalid DoG polarity/scale ratio')
        values=[];blobs=[];sign=1 if settings['polarity']=='bright' else -1
        for _,row in df.iterrows():
            with Image.open(resolve(cfg,row.roi_path)) as im:gray=np.asarray(im.convert('L'),dtype=np.float32)
            best=(-float('inf'),None)
            yy,xx=np.indices(gray.shape);cy,cx=(np.array(gray.shape)-1)/2
            allowed=(xx-cx)**2+(yy-cy)**2<=settings['max_center_distance']**2
            for sigma in settings['sigmas']:
                if sigma<=0:raise ValueError('DoG sigma must be positive')
                response=sign*(cv2.GaussianBlur(gray,(0,0),sigma)-cv2.GaussianBlur(gray,(0,0),sigma*settings['sigma_ratio']))
                response=np.where(allowed,response,-np.inf);y,x=np.unravel_index(np.argmax(response),response.shape)
                if response[y,x]>best[0]:best=(float(response[y,x]),{'x':int(x),'y':int(y),'sigma':sigma})
            values.append(best[0]);blobs.append(best[1])
        return earliest_runs(np.asarray(values)>=settings['response_threshold'],frames,settings['min_run']),{'response':values,'selected_blob':blobs}
    if method=='pelt':
        try:import ruptures
        except ImportError as e:raise Blocked('Install pinned ruptures dependency to run PELT; no substitute CP result is generated') from e
        for k in ('columns','cost_model','penalty','min_size','jump','direction_column','min_increase'):
            if k not in settings or settings[k] is None:raise Blocked(f'PELT requires {k}')
        require_columns(df,settings['columns']+[settings['direction_column']],'PELT input')
        candidates=[];all_boundaries=[]
        # Missing physical frames split the input; no interpolation is invented.
        starts=np.r_[0,np.where(np.diff(frames)!=1)[0]+1];stops=np.r_[starts[1:],len(df)]
        for lo,hi in zip(starts,stops):
            if hi-lo<2*settings['min_size']:continue
            x=df[settings['columns']].iloc[lo:hi].to_numpy(float)
            ends=ruptures.Pelt(model=settings['cost_model'],min_size=settings['min_size'],jump=settings['jump']).fit(x).predict(pen=settings['penalty'])
            bounds=[0]+ends
            for k,end in enumerate(ends[:-1]):
                i=lo+end;pre=df[settings['direction_column']].iloc[lo+bounds[k]:i].mean();post=df[settings['direction_column']].iloc[i:lo+ends[k+1]].mean()
                all_boundaries.append({'sample_idx':int(i),'frame_idx':int(frames[i]),'increase':float(post-pre)})
                if post-pre>=settings['min_increase']:candidates.append(int(i))
        return candidates,{'boundaries':all_boundaries,'terminal_N_excluded':True,'future_context':True}
    if method=='resnet18':
        from .supervised import infer_state_probabilities
        probabilities=infer_state_probabilities(df,cfg,settings)
        for k in ('threshold','min_run'):
            if k not in settings or settings[k] is None:raise Blocked(f'ResNet onset policy requires {k}')
        return earliest_runs(probabilities>=settings['threshold'],frames,settings['min_run']),{'p_ignited':probabilities.tolist()}
    if method=='prior9':raise Blocked('Prior9 original specification/code not supplied; not replaced by a renamed intensity threshold')
    raise Blocked(f'Unimplemented/unspecified external method {method}; no fabricated result')


def judge_track(df,method_id,settings,mod,base_config,cfg):
    config=replace(base_config,**settings.get('gmm_overrides',{}))
    method=settings['method'];out={'method_id':method_id,'has_ignition':False,'onset_frame':None,'confirmation_frame':None,'status':'no_event'}
    diagnostics={};candidate_rows=[]
    paper=method in ('sas_paper','dog_paper','cnf_roi_stage')
    if len(df)<(int(settings.get('min_sequence_len',1)) if paper else config.min_sequence_len):return {**out,'status':'too_short'},[],{}
    # Screens are opt-in for external methods; no GMM-specific constraints are hidden in them.
    if method=='gmm' or settings.get('apply_legacy_screens',False):
        reject,shape=mod.check_roi_shape_sequence_quality(df,config);diagnostics['geometry']=shape
        if reject:return {**out,'status':'geometry_rejected'},[],diagnostics
        hot,initial=mod.is_initially_hot_sequence(df,config);diagnostics['initial_hot']=initial
        if hot:return {**out,'status':'initial_hot_rejected'},[],diagnostics
    indices,trace=propose(df,method,settings,mod,config,cfg);diagnostics['proposal']=trace
    policy=settings.get('candidate_policy','first_terminal')
    if policy not in ('first_terminal','retry'):raise ValueError('Unknown candidate policy')
    considered=indices[:1] if policy=='first_terminal' else indices
    for order,index in enumerate(considered):
        pass_rtv,rtv=rtv_trace(df,index,config,settings.get('rtv_rule','or'),settings.get('minimum_votes')) if settings.get('rtv',False) else (True,{'enabled':False})
        confirm_n=config.min_hot_run if method=='gmm' else settings.get('min_run',1)
        if method=='legacy_baseline':
            key='gray_min_persist' if settings.get('legacy_method')=='gray_p99_threshold' else 'change_min_persist'
            confirm_n=settings['params'][key]
        if settings.get('rtv',False):confirm_n=max(confirm_n,config.min_hot_run)
        confirmation_idx=int(trace.get('confirmation_indices',{}).get(str(index),min(len(df)-1,index+confirm_n-1)))
        candidate={'candidate_order':order,'sample_idx':index,'candidate_onset':int(df.frame_idx.iloc[index]),
                   'confirmation_frame':int(df.frame_idx.iloc[confirmation_idx]),'rtv_pass':pass_rtv,'rtv':rtv}
        candidate_rows.append(candidate)
        if pass_rtv:
            out.update({'has_ignition':True,'onset_frame':candidate['candidate_onset'],
                        'confirmation_frame':candidate['confirmation_frame'],'status':'accepted'});break
        out['status']='rtv_rejected'
    out['feature_schema_hash']=trace.get('feature_schema_hash')
    out['candidate_count']=len(indices);out['future_context']=trace.get('future_context',True)
    if out['has_ignition']:
        row=df.loc[df.frame_idx==out['onset_frame']].iloc[0]
        for key in ('cx','cy','x1','y1','x2','y2'):out['onset_'+key]=row.get(key)
        if method=='cnf_roi_stage':
            fit=next(f for f in trace['fits'] if f.get('accepted') and f['onset_frame']==out['onset_frame'])
            out['combustion_frame']=fit['combustion_frame']
    return out,candidate_rows,diagnostics
