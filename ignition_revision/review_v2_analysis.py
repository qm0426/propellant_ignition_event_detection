"""Evidence-based analysis jobs; absent human measurements remain explicit blocks."""
from pathlib import Path
import itertools
import json
import numpy as np
import pandas as pd
from .common import Blocked, boolean, require_columns, write_csv, write_json


TEMPLATES={
    'ERROR_CAUSES':['ref_event_id','match_status','decision_stage','error_reason','source','quality'],
    'AGREEMENT':['recording_id','ref_event_id','annotator_id','pass_id','has_ignition','onset_frame'],
    'LOCATION_ERROR':['ref_event_id','source','pred_x','pred_y','manual_core_x','manual_core_y','quality'],
    'APPLICATION_REVIEW':['candidate_id','review_status','reviewer_id','onset_frame','error_reason'],
    'COMBUSTION_MEASUREMENTS':['recording_id','ref_event_id','phase','phase_start_frame','phase_stop_frame','fps',
                               'onset_frame','detachment_frame','height_above_surface_px','um_per_px','apparent_size_px',
                               'pre_onset_velocity_px_per_frame','confirmed_true','eligible_particles'],
    'ILLUMINATION_STRATA':['recording_id','ref_event_id','illumination','has_ignition_gt','has_ignition_pred'],
    'QUALITY_STRATA':['recording_id','ref_event_id','quality','has_ignition_gt','has_ignition_pred'],
}


def templates(out):
    for name,columns in TEMPLATES.items():
        path=out/(name+'.csv')
        if not path.exists():write_csv(path,[],columns)


def run_analysis(name,path,out):
    if not path.is_file():raise Blocked('Missing manual input: '+str(path))
    d=pd.read_csv(path);require_columns(d,TEMPLATES[name],name)
    if d.empty:raise Blocked('Manual template has no completed observations')
    out.mkdir(parents=True,exist_ok=True)
    if name=='ERROR_CAUSES':
        if not d.match_status.isin(['TP','FP','FN','TN']).all():raise ValueError('Unknown event confusion category')
        write_csv(out/'error_cause_counts.csv',d.groupby(['match_status','decision_stage','error_reason','source','quality'],dropna=False).size().reset_index(name='N'))
        write_csv(out/'reviewed_error_members.csv',d)
    elif name=='AGREEMENT':
        if d.duplicated(['recording_id','ref_event_id','annotator_id','pass_id']).any():raise ValueError('Duplicate annotation')
        from sklearn.metrics import cohen_kappa_score
        rows=[]
        def compare(a,b,label):
            m=a.merge(b,on=['recording_id','ref_event_id'],suffixes=('_a','_b'),validate='one_to_one')
            valid=m.has_ignition_a.notna()&m.has_ignition_b.notna();m=m.loc[valid]
            if m.empty:return
            aa=m.has_ignition_a.map(boolean);bb=m.has_ignition_b.map(boolean)
            common=aa&bb&m.onset_frame_a.notna()&m.onset_frame_b.notna()
            errors=(m.loc[common,'onset_frame_a']-m.loc[common,'onset_frame_b']).abs()
            kappa=cohen_kappa_score(aa,bb) if len(set(aa)|set(bb))>1 else None
            rows.append({'comparison':label,'N':len(m),'existence_agreement':float(aa.eq(bb).mean()),'kappa':kappa,
                         'both_positive':int((aa&bb).sum()),'both_negative':int((~aa&~bb).sum()),
                         'positive_negative':int((aa&~bb).sum()),'negative_positive':int((~aa&bb).sum()),
                         'onset_N':len(errors),'onset_MAE':float(errors.mean()) if len(errors) else None,
                         'onset_median_AE':float(errors.median()) if len(errors) else None,
                         'onset_p95_AE':float(errors.quantile(.95)) if len(errors) else None,
                         'onset_within1':float(errors.le(1).mean()) if len(errors) else None,
                         'onset_within3':float(errors.le(3).mean()) if len(errors) else None})
        groups={(str(a),str(p)):g for (a,p),g in d.groupby(['annotator_id','pass_id'])}
        for a,b in itertools.combinations(groups,2):
            if a[0]!=b[0] and a[1]==b[1]:compare(groups[a],groups[b],f'inter:{a}:{b}')
            elif a[0]==b[0]:compare(groups[a],groups[b],f'intra:{a}:{b}')
        if not rows:raise Blocked('Need shared items from independent annotators or repeated annotation passes')
        write_csv(out/'agreement_summary.csv',rows)
    elif name=='LOCATION_ERROR':
        for c in ['pred_x','pred_y','manual_core_x','manual_core_y']:
            if not np.isfinite(d[c]).all():raise ValueError('Missing/nonfinite location; put unassessable items in separate list')
        d['error_x']=d.pred_x-d.manual_core_x;d['error_y']=d.pred_y-d.manual_core_y
        d['distance_px']=np.hypot(d.error_x,d.error_y)
        write_csv(out/'location_members.csv',d)
        rows=[]
        for (source,quality),g in d.groupby(['source','quality'],dropna=False):
            rows.append({'source':source,'quality':quality,'N':len(g),'mean_px':g.distance_px.mean(),
                         'median_px':g.distance_px.median(),'p95_px':g.distance_px.quantile(.95),
                         'bias_x_px':g.error_x.mean(),'bias_y_px':g.error_y.mean()})
        write_csv(out/'location_error_by_source.csv',rows)
    elif name=='APPLICATION_REVIEW':
        if d.candidate_id.duplicated().any():raise ValueError('Duplicate application candidate')
        allowed={'reviewed_true','reviewed_false','uncertain','not_reviewed'}
        if set(d.review_status)-allowed:raise ValueError('Unknown review status')
        true=int(d.review_status.eq('reviewed_true').sum());false=int(d.review_status.eq('reviewed_false').sum())
        write_json(out/'event_review_summary.json',{'N':len(d),'counts':d.review_status.value_counts().to_dict(),
                   'precision_on_decidable_reviewed_candidates':true/(true+false) if true+false else None,
                   'recall':None,'recall_note':'Candidate-only review cannot estimate missed events; exhaustive independent blocks needed'})
    elif name=='COMBUSTION_MEASUREMENTS':
        d=d.loc[d.confirmed_true.map(boolean)].copy()
        if d.empty:raise Blocked('No independently confirmed true application events')
        if (d.fps<=0).any() or (d.um_per_px<=0).any():raise ValueError('Positive measured fps and scale required')
        delay=d.onset_frame-d.detachment_frame
        if delay.dropna().lt(0).any():raise ValueError('Onset earlier than detachment; classify censored cases separately')
        d['detachment_to_optical_onset_us']=delay/d.fps*1e6
        d['height_above_surface_um']=d.height_above_surface_px*d.um_per_px
        d['apparent_box_size_um']=d.apparent_size_px*d.um_per_px
        d['pre_onset_velocity_m_per_s']=d.pre_onset_velocity_px_per_frame*d.um_per_px*d.fps/1e6
        write_csv(out/'application_measurements.csv',d)
        phases=[]
        for (rec,phase),g in d.groupby(['recording_id','phase']):
            cols=['phase_start_frame','phase_stop_frame','fps','eligible_particles']
            if any(g[c].nunique(dropna=False)!=1 for c in cols):raise ValueError('Inconsistent phase exposure/denominator metadata')
            seconds=(g.phase_stop_frame.iloc[0]-g.phase_start_frame.iloc[0])/g.fps.iloc[0]
            if seconds<=0:raise ValueError('Phase uses half-open [start,stop), positive duration required')
            denominator=g.eligible_particles.iloc[0]
            if pd.notna(denominator) and denominator<len(g):raise ValueError('Eligible particle denominator smaller than event count')
            phases.append({'recording_id':rec,'phase':phase,'events':len(g),'duration_ms':seconds*1000,
                           'events_per_ms':len(g)/(seconds*1000),'eligible_particles':denominator,
                           'cohort_conditional_fraction':len(g)/denominator if pd.notna(denominator) and denominator>0 else None})
        write_csv(out/'phase_rates.csv',phases)
    elif name in ('ILLUMINATION_STRATA','QUALITY_STRATA'):
        from .evaluation import counts_metrics
        rows=[]
        field='illumination' if name=='ILLUMINATION_STRATA' else 'quality'
        for (rec,illumination),g in d.groupby(['recording_id',field]):
            a=g.has_ignition_gt.map(boolean);b=g.has_ignition_pred.map(boolean)
            rows.append({'recording_id':rec,field:illumination,'N':len(g),
                         **counts_metrics((a&b).sum(),(~a&b).sum(),(a&~b).sum(),(~a&~b).sum())})
        write_csv(out/(field+'_results.csv'),rows)
    write_json(out/'analysis_scope.json',{'input':str(path.resolve()),'rows':len(d),'analysis':name,
                                         'note':'Uses supplied real annotations; does not create independent human labels or prove blinding'})


def run_special(job,settings,data,out):
    root=Path(settings['project_root']);name=job['id'];dest=out/'analyses'/name;dest.mkdir(parents=True,exist_ok=True)
    if job['family']=='data_audit':
        meta=json.loads((data/'dataset_meta.json').read_text(encoding='utf-8'));write_json(dest/'data_audit.json',meta)
    elif job['family']=='parameters':
        from .common import load_legacy
        from .tracking import tracking_settings
        from dataclasses import asdict
        gmm=asdict(load_legacy('ignition_detection_gmm_rtv_new.py').IgnitionGMMConfig(random_state=2026))
        tracker=tracking_settings(load_legacy('ot_kalman_tracker.py'),{})
        parser=load_legacy('postprocess_detection_txt_modified.py').build_arg_parser()
        cleaning={a.dest:a.default for a in parser._actions if a.dest!='help'}
        rows=[{'module':module,'key':k,'value':str(v),'selection':'historical defaults / explicit retrospective starting values'}
              for module,d in [('GMM_RTV',gmm),('tracking',tracker),('cleaning',cleaning)] for k,v in d.items()]
        write_csv(dest/'parameter_registry.csv',rows)
        write_json(dest/'parameter_sources.json',{'gmm':gmm,'tracking':tracker,'cleaning':cleaning,
                                                 'note':'Per-case method overrides live in resolved_cases and effective run configs; no claim of independently selected optima'})
    elif job['family']=='old_rtv':
        source=root.parent/'ignition_ablation_gmm_rtv_results/run_20260810_192817/per_track_results.csv'
        d=pd.read_csv(source);a=d.loc[d.method.eq('gmm_without_rtv')].set_index('track_id');b=d.loc[d.method.eq('gmm_rtv_full')].set_index('track_id')
        m=a[['pred_has_ignition','gt_has_ignition']].join(b[['pred_has_ignition']],lsuffix='_before',rsuffix='_after',validate='one_to_one')
        before=m.pred_has_ignition_before.map(boolean);after=m.pred_has_ignition_after.map(boolean);gt=m.gt_has_ignition.map(boolean)
        write_csv(dest/'rtv_members.csv',m.reset_index())
        write_json(dest/'rtv_summary.json',{'removed_TP':int((before&~after&gt).sum()),'removed_FP':int((before&~after&~gt).sum()),
                                          'added_events':int((~before&after).sum()),'scope':'historical GT version, not new GT re-evaluation'})
    elif job['family']=='table_figure':
        files=[root.parent/'ignition_ablation_gmm_rtv_results/run_20260810_192817/method_summary.csv',
               root.parent/'feature_group_ablation_gmm_rtv_results/run_20260810_172643/feature_group_summary.csv',
               root.parent/'parameter_sensitivity_results/tracking_sensitivity_fixed/combined_summary.csv']
        tables=[]
        for f in files:
            d=pd.read_csv(f);keep=[c for c in ['method','feature_group','config_id','n_tracks','ignition_f1','onset_mae','n_common_tp','reference_common_mae','is_default_configuration'] if c in d]
            t=d[keep].copy();t['source']=str(f);tables.append(t)
        write_csv(dest/'source_table_comparison.csv',pd.concat(tables,ignore_index=True))
        write_json(dest/'status.json',{'status':'source_differences_exported','note':'Does not resolve manuscript figure labels or prove default protocols identical'})
    elif job['family']=='official_mot':
        import importlib.util
        f=root/'evaluate_unified_mot_official_20261029.py';spec=importlib.util.spec_from_file_location('review_v2_official',f);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        import subprocess,sys
        subprocess.run([sys.executable,str(f)],check=True)
    elif job['family']=='build_fpn':
        from .ignition_dataset import build_ignition_dataset
        cfg={'canonical_source_dir':str(data/'fpn_source'),'output_dir':str(data/'fpn_corrected'),
             'trust_tracks_for_training':True,'pre_frames':20,'post_frames':20,'onset_guard_frames':0,
             'unknown_policy':'ignore_regions','include_non_event_training_frames':False,
             'purge_cross_split_tracks':True,'split_ranges':[{'split':'train','start':0,'end':3999},
                {'split':'validation','start':4500,'end':5499},{'split':'test','start':6000,'end':7996}]}
        cfg.update(settings.get('config_overrides',{}).get(name,{}))
        marker=data/'fpn_corrected/dataset_meta.json'
        if marker.exists():
            old=json.loads(marker.read_text(encoding='utf-8'))
            if old['source_meta_sha256']!=__import__('hashlib').sha256((data/'fpn_source/dataset_meta.json').read_bytes()).hexdigest():raise ValueError('Stale FPN dataset')
        else:build_ignition_dataset(cfg)
        # Complete test lifetimes, not crops/frames selected around the GT onset.
        training_frames=pd.read_csv(data/'fpn_corrected/frames.csv')
        available=set(training_frames.loc[training_frames.split.eq('test'),'frame_idx'].astype(int))
        shared=pd.read_csv(data/'fpn_corrected/shared_tracks.csv')
        labels=pd.read_csv(data/'fpn_corrected/fixed_cohort_labels.csv')
        eligible=[int(tid) for tid,g in shared.groupby('track_id') if set(g.frame_idx.astype(int)).issubset(available)]
        cohort=labels.loc[labels.track_id.isin(eligible)]
        needed=set(shared.loc[shared.track_id.isin(cohort.track_id),'frame_idx'])
        frame_table=pd.read_csv(data/'fpn_corrected/inference_frames.csv')
        frame_table=frame_table.loc[frame_table.frame_idx.isin(needed)].copy();frame_table['split']='test'
        if cohort.empty or frame_table.empty:raise Blocked('No complete track lifetimes in corrected test scope')
        write_csv(data/'fpn_test/frames.csv',frame_table)
        write_csv(data/'fpn_test/shared_tracks.csv',shared.loc[shared.frame_idx.isin(needed)])
        write_csv(data/'fpn_test/labels.csv',cohort)
        write_csv(data/'fpn_test/cohort.csv',cohort[['recording_id','track_id']])
        write_json(data/'fpn_test/scope.json',{'scope':'same recording temporal test, complete lifetimes',
                                             'frames':len(frame_table),'tracks':len(cohort),'selected_by_onset':False})
        write_json(dest/'build_config.json',cfg)
