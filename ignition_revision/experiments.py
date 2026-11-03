from __future__ import annotations
import copy
import itertools
import warnings
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import Blocked, boolean, digest, read_table, require_columns, resolve, write_csv, write_json
from .evaluation import common_tp, evaluate_fixed, evaluate_events, map_tracks_to_reference
from .judges import extract_track_features, judge_track


def decision_variants(cfg):
    full=copy.deepcopy(cfg.get('full_method',{'method':'gmm','rtv':True,'candidate_policy':'first_terminal'}))
    methods={'DEC_FULL':full}
    changes={'DEC_NO_GEOM':{'enable_shape_sequence_reject':False},'DEC_NO_INITIALHOT':{'enable_initial_hot_reject':False},
             'DEC_NO_SMOOTH':{'min_state_run':1},'DEC_NO_PRIORCOLD':{'require_all_prior_cold':False}}
    for name,overrides in changes.items():
        methods[name]=copy.deepcopy(full);methods[name].setdefault('gmm_overrides',{}).update(overrides)
    methods['DEC_NO_RTV']={**copy.deepcopy(full),'rtv':False}
    base=cfg.get('base_feature_columns')
    if base is None:
        from .common import load_legacy
        mod=load_legacy('ignition_detection_gmm_rtv_new.py',cfg.get('legacy_root'))
        base=list(mod.BASE_GMM_FEATURES)+list(mod.DIFF_GMM_FEATURES)
    methods['FEAT_NO_DELTA']={**copy.deepcopy(full),'feature_columns':[c for c in base if not c.startswith(('d1_','d2_'))]}
    methods['FEAT_NO_COLOR']={**copy.deepcopy(full),'feature_columns':[c for c in base if not any(k in c for k in ('red_ratio','rg_ratio','rb_ratio'))]}
    methods['FEAT_NO_MORPH']={**copy.deepcopy(full),'feature_columns':[c for c in base if 'largest_area_ratio' not in c]}
    # Explicit factor definition is required: bg_med/bg_mad are absent from old Full-19.
    bg=cfg.get('background_factor')
    if bg:
        if bg.get('kind') not in ('independent_columns','corrected_columns') or not bg.get('columns'):
            raise ValueError('Specify background_factor.kind and exact column list')
        if bg['kind']=='independent_columns' and set(bg['columns'])&set(base):
            raise ValueError('Independent background columns must be added to an explicitly background-free base')
        for geometry,background in itertools.product((False,True),repeat=2):
            cols=list(base)
            if bg['kind']=='independent_columns' and background:cols+=list(bg['columns'])
            if bg['kind']=='corrected_columns' and not background:cols=[c for c in cols if c not in bg['columns']]
            name=f'GEOM_{int(geometry)}_BG_{int(background)}'
            setting={**copy.deepcopy(full),'feature_columns':cols}
            setting.setdefault('gmm_overrides',{})['enable_shape_sequence_reject']=geometry
            methods[name]=setting
        if bg['kind']=='independent_columns':
            # Explicit new reference: Full19 + background, rather than claiming
            # the old Full19 already contained independent background columns.
            methods['FEAT_FULL_PLUS_BG']={**copy.deepcopy(full),'feature_columns':list(base)+list(bg['columns'])}
            methods['FEAT_NO_BG']={**copy.deepcopy(full),'feature_columns':list(base),
                                   'ablation_reference':'FEAT_FULL_PLUS_BG'}
            methods['FEAT_GRAY']={**copy.deepcopy(full),'feature_columns':['gray_p95','gray_p99']}
            methods['FEAT_GRAY_BG']={**copy.deepcopy(full),'feature_columns':['gray_p95','gray_p99']+list(bg['columns'])}
    methods.update(cfg.get('extra_methods',{}))
    selected=cfg.get("selected_method_ids")
    if selected is not None:
        unknown=set(selected)-set(methods)
        if unknown:raise ValueError(f"Unknown decision methods: {sorted(unknown)}")
        methods={k:v for k,v in methods.items() if k in selected}
    return methods


def predict_feature_bundle(bundle,mod,base,methods,cfg,run,output_root=None):
    output_root=Path(output_root or run.out);outputs={}
    for method_id,settings in methods.items():
        rows=[];audits=[];diagnostics=[]
        try:
            for progress,((rec,tid),df) in enumerate(bundle.items(),1):
                with run.time('judge_'+method_id,unit='tracks',device=settings.get('device','cpu')):
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter('always')
                        if settings.get('method')=='prior9_normint':
                            from .prior9 import process_track,SETTINGS
                            result,trace,signals=process_track(df,method_id,{**SETTINGS,**settings},cfg,output_root/method_id)
                            write_csv(output_root/method_id/'signals'/str(rec)/(str(tid)+'.csv'),signals)
                            candidates=[{'candidate_onset':result['onset_frame'],'confirmation_frame':result['confirmation_frame']}] if result['has_ignition'] else []
                        else:result,candidates,trace=judge_track(df,method_id,settings,mod,base,cfg)
                    trace['warnings']=[{'category':type(w.message).__name__,'message':str(w.message)} for w in captured]
                rows.append({'recording_id':rec,'track_id':tid,**result})
                audits.extend({'recording_id':rec,'track_id':tid,'method_id':method_id,**c} for c in candidates)
                diagnostics.append({'recording_id':rec,'track_id':tid,'trace':trace})
                if len(bundle)>250 and (progress%250==0 or progress==len(bundle)):
                    print(f'JUDGE_PROGRESS {method_id}: {progress}/{len(bundle)} tracks',flush=True)
            table=pd.DataFrame(rows,columns=['recording_id','track_id','method_id','has_ignition','onset_frame','confirmation_frame','status','feature_schema_hash','candidate_count','future_context','onset_cx','onset_cy','onset_x1','onset_y1','onset_x2','onset_y2','combustion_frame'])
            out=output_root/method_id
            write_csv(out/'predictions.csv',table);write_json(out/'candidate_audit.json',audits);write_json(out/'diagnostics.json',diagnostics)
            write_json(out/'method_config.json',settings)
            outputs[method_id]=table
        except Blocked as e:run.record(method_id,'blocked',str(e))
        except Exception as e:
            run.record(method_id,'failed',f'{type(e).__name__}: {e}')
            write_json(output_root/method_id/'error.json',{'exception':type(e).__name__,'message':str(e)})
    return outputs


def evaluate_fixed_methods(outputs,cfg,run,output_root=None):
    root=Path(output_root or run.out);metrics=[];matches={}
    if not cfg.get('labels'):
        for name in outputs:run.record(name,'prediction_only','Independent fixed-cohort labels not configured')
        return metrics,matches
    if not cfg.get('label_provenance') or not cfg.get('track_bundle_id'):
        raise Blocked('Fixed-cohort evaluation requires explicit label_provenance and frozen track_bundle_id')
    labels=read_table(resolve(cfg,cfg['labels']))
    for name,pred in outputs.items():
        try:
            with run.time('evaluation_'+name):score,rows=evaluate_fixed(pred,labels)
            score.update({'evaluation_scope':cfg.get('evaluation_mode','independent_reference'),'method_id':name,'track_bundle_id':cfg['track_bundle_id'],'label_provenance':cfg['label_provenance']})
            metrics.append(score);matches[name]=rows
            write_csv(root/name/'evaluation_members.csv',rows);run.record(name,'completed')
        except Exception as e:run.record(name,'failed',f'Evaluation: {e}')
    ct,members=common_tp(matches)
    write_csv(root/'judge_comparison_fixed_tracks.csv',metrics)
    write_csv(root/'S07_common_tp.csv',ct);write_json(root/'common_tp_members.json',{'methods':list(matches),'members':members,'cohort_hash':digest(members)})
    paired=[]
    for a,b in itertools.combinations(matches,2):
        rows,ids=common_tp({a:matches[a],b:matches[b]})
        paired.extend({**x,'comparison':f'{a}__{b}'} for x in rows)
    write_csv(root/'pairwise_common_tp.csv',paired)
    return metrics,matches


def run_onsets(cfg,run,ablation=False):
    if not ablation and cfg.get('methods') and all(s.get('method')=='prior9_normint' for s in cfg['methods'].values()):
        from .prior9 import run_prior9
        return run_prior9(cfg,run)
    rois=read_table(resolve(cfg,cfg.get('rois')))
    with run.time('feature_bundle',len(rois),'roi_frames'):
        bundle,base,mod=extract_track_features(rois,cfg,run)
    if not bundle:raise ValueError('Empty ROI cohort')
    write_csv(run.out/'raw_feature_superset.csv',pd.concat(bundle.values(),ignore_index=True))
    methods=decision_variants(cfg) if ablation else cfg.get('methods')
    if not methods:raise ValueError('Configure methods explicitly')
    registry=[]
    levels={'sas_paper':'fixed_roi_modality_adapted','dog_paper':'fixed_roi_modality_adapted','cnf_roi_stage':'explicit_offline_stage_adaptation','gmm':'legacy_algorithm_adapted_interface','legacy_baseline':'legacy_reproduction','gray':'project_adapted_rule','change':'project_adapted_rule','sas':'modality_adapted','dog':'modality_adapted','resnet18':'modality_adapted','pelt':'generic_temporal','prior9':'blocked_pending_original_spec'}
    for name,setting in methods.items():
        registry.append({'method_id':name,'implementation':setting['method'],'reproduction_level':levels.get(setting['method'],'blocked'),
                         'rtv':setting.get('rtv',False),'input_kind':'ROI_images' if setting['method'] in ('dog','resnet18','sas_paper','dog_paper','cnf_roi_stage') else 'named_features',
                         'requires_training':setting['method']=='resnet18','source':setting.get('source_key'),'configuration_hash':digest(setting)})
    for row in registry:
        if row['implementation']=='hmm':row['reproduction_level']='offline_constrained_gaussian_HMM'
    write_csv(run.out/'onset_method_registry.csv',registry)
    outputs=predict_feature_bundle(bundle,mod,base,methods,cfg,run)
    metrics,matches=evaluate_fixed_methods(outputs,cfg,run)
    write_csv(run.out/('S08_decision_feature_ablation.csv' if ablation else 'S06_judge_comparison.csv'),metrics)
    # Member-level RTV retention/rejection and error tags are not inferred from rounded recall.
    changes=[]
    for name,rows in matches.items():
        if not name.endswith('_RTV'):continue
        unpaired=name[:-4]
        if unpaired not in matches:continue
        before={x['ref_event_id']:x for x in matches[unpaired]};after={x['ref_event_id']:x for x in rows}
        for eid in before:
            changes.append({'method_before':unpaired,'method_after':name,'ref_event_id':eid,
                            'before':before[eid]['match_status'],'after':after[eid]['match_status']})
    write_csv(run.out/'S09_rtv_member_changes.csv',changes)


def evaluate_tracking_observations(obs,pred,cfg,run,out):
    if not cfg.get('reference_boxes') or not cfg.get('reference_events'):
        return None,[],'Independent reference_boxes/reference_events not configured'
    evaluation=cfg.get('evaluation',{})
    for key in ('mapping_max_distance','mapping_min_shared_frames','mapping_min_purity'):
        if key not in evaluation or evaluation[key] is None:raise Blocked(f'Explicit evaluation.{key} required')
    boxes=read_table(resolve(cfg,cfg['reference_boxes']))
    # Native trackers retain their original emitted boxes. Diagnostic mapping may use them,
    # with a separate support tag; do not invent original detection provenance.
    mapped_obs=obs.copy()
    if not mapped_obs.empty:
        mapped_obs['detection_supported']=mapped_obs.detection_supported | mapped_obs.source.eq('native_mot_output')
    mapping,edges,frame_matches=map_tracks_to_reference(mapped_obs,boxes,evaluation['mapping_max_distance'],evaluation['mapping_min_shared_frames'],evaluation['mapping_min_purity'])
    write_csv(out/'object_mapping.csv',mapping);write_csv(out/'object_mapping_candidates.csv',edges);write_csv(out/'identity_frame_matches.csv',frame_matches)
    reference=read_table(resolve(cfg,cfg['reference_events']))
    events=pred[pred.has_ignition.map(boolean)].copy()
    events['pred_event_id']=[f'{rec}:{tid}:first' for rec,tid in zip(events.recording_id,events.track_id)]
    locations=[]
    for _,ev in events.iterrows():
        point=obs[(obs.recording_id==ev.recording_id)&(obs.track_id==ev.track_id)&(obs.frame_idx==ev.onset_frame)]
        locations.append((float(point.iloc[0].cx),float(point.iloc[0].cy)) if len(point)==1 else (None,None))
    events['location_x']=[x[0] for x in locations];events['location_y']=[x[1] for x in locations]
    events=events[['recording_id','track_id','pred_event_id','onset_frame','location_x','location_y']]
    write_csv(out/'predicted_events.csv',events)
    score,rows=evaluate_events(events,reference,mapping,duration_s=evaluation.get('duration_s'))
    write_csv(out/'event_matches.csv',rows)
    tolerance=[]
    for tau in evaluation.get('tolerances',[1,3]):
        metric,matched=evaluate_events(events,reference,mapping,tolerance=tau)
        tolerance.append(metric);write_csv(out/f'event_matches_pm{tau}.csv',matched)
    write_csv(out/'tolerance_metrics.csv',tolerance)
    windows=tracking_windows(obs,reference,mapping,evaluation)
    write_csv(out/'window_observability_per_reference.csv',windows)
    for metric in ('roi_pre_onset_coverage','detector_pre_onset_coverage','roi_window_completeness','detector_window_completeness','observable_transition'):
        score[metric]=float(np.mean([x[metric] for x in windows])) if windows else None
    score['detector_supported_observations']=int(obs.detection_supported.sum())
    score['gap_roi_observations']=int(obs.source.eq('gap_roi').sum())
    score['prebirth_roi_observations']=int(obs.source.eq('prebirth_roi').sum())
    score['mapping_reference_id_switches']=int(mapping.reference_id_switches.sum()) if not mapping.empty else 0
    score['mapping_support']='native_emitted_boxes' if obs.source.eq('native_mot_output').any() else 'detector_supported'
    score['evaluation_scope']=cfg.get('evaluation_mode','independent_reference')
    score['HOTA']=None;score['IDF1']=None;score['IDSW']=None
    if cfg.get('evaluation_mode')=='legacy_reproduction':
        write_json(out/'standard_mot_metric_status.json',{'status':'not_independent_GT','reason':'Old algorithm tracks are diagnostic references, not manual MOT GT'})
        return score,rows,None
    try:
        from .mot_metrics import evaluate_standard_mot
        from .detection import validate_frames
        full_frames=validate_frames(read_table(resolve(cfg,cfg['frames'])))
        score.update(evaluate_standard_mot(obs,boxes,full_frames,cfg,out))
        write_json(out/'standard_mot_metric_status.json',{'status':'completed','implementation':'official TrackEval'})
    except Blocked as e:
        write_json(out/'standard_mot_metric_status.json',{'status':'blocked','reason':str(e)})
    return score,rows,None


def run_tracking(cfg,run,restoration=False):
    from .tracking import assignment_variants,restoration_variants,run_custom,external_mot,crop_observations,run_center_nn
    from .detection import validate_frames,validate_candidates
    frames=validate_frames(read_table(resolve(cfg,cfg.get('frames'))));cand=validate_candidates(read_table(resolve(cfg,cfg.get('candidates'))))
    membership=cand.merge(frames[['recording_id','frame_idx']],on=['recording_id','frame_idx'],how='left',indicator=True)
    if (membership['_merge']!='both').any():raise ValueError('Candidate frame outside declared input scope')
    variants=cfg.get('variants') or (restoration_variants() if restoration else assignment_variants())
    metrics=[];matches={}
    for variant in variants:
        name=variant['id'];out=run.out/name
        try:
            backend=variant.get('backend','ot')
            from .recovery import reuse_tracking_stage
            recovered=reuse_tracking_stage(cfg,run,variant,out)
            if recovered is not None:obs,rois=recovered;trace=[]
            elif backend in ('hungarian','ot'):obs,trace=run_custom(cand,frames,cfg,variant,run)
            elif backend=='nn':obs,trace=run_center_nn(cand,frames,cfg,variant,run)
            else:
                with run.time('native_'+backend,len(frames),'frames'):
                    obs=external_mot(cand,frames,cfg,variant,audit_out=out)
                trace=[]
            write_csv(out/'track_observations.csv',obs);write_csv(out/'association_trace.csv',trace)
            export_mot(obs,out)
            if cfg.get('tracking_evaluation')=='mot_only':
                from .mot_metrics import evaluate_standard_mot
                gt=read_table(resolve(cfg,cfg['reference_boxes']))
                metric=evaluate_standard_mot(obs,gt,frames,cfg,out)
                metrics.append({'method_id':name,'evaluation_scope':'MOT only; no exhaustive event-negative assumption',**metric})
                run.record(name,'completed')
                continue
            if recovered is None:
                with run.time('roi_extraction_'+name,len(obs),'roi_frames'):
                    rois=crop_observations(obs,frames,cfg,out)
            write_csv(out/'roi_manifest.csv',rois)
            if rois.empty:
                pred=pd.DataFrame(columns=['recording_id','track_id','has_ignition','onset_frame'])
            else:
                bundle,base,mod=extract_track_features(rois,cfg,run)
                output=predict_feature_bundle(bundle,mod,base,{name:cfg.get('judge',{'method':'gmm','rtv':True})},cfg,run,out)
                if name not in output:continue
                pred=output[name]
            score,rows,reason=evaluate_tracking_observations(obs,pred,cfg,run,out)
            if reason:run.record(name,'prediction_only',reason)
            else:
                metrics.append({'method_id':name,**score});matches[name]=rows;run.record(name,'completed')
        except Blocked as e:run.record(name,'blocked',str(e))
        except Exception as e:run.record(name,'failed',f'{type(e).__name__}: {e}')
    write_csv(run.out/('S05_restoration_events.csv' if restoration else 'S04_assignment_and_mot.csv'),metrics)
    common,members=common_tp(matches);write_csv(run.out/'common_tp.csv',common);write_json(run.out/'common_tp_members.json',{'methods':list(matches),'members':members})
    if restoration and 'REST_FULL' in matches:
        full={x['ref_event_id']:x['match_status'] for x in matches['REST_FULL'] if x.get('ref_event_id')}
        changes=[]
        for name,rows in matches.items():
            current={x['ref_event_id']:x['match_status'] for x in rows if x.get('ref_event_id')}
            for eid in full:
                before,after=current.get(eid,'FN'),full[eid]
                changes.append({'baseline':name,'full':'REST_FULL','ref_event_id':eid,'baseline_status':before,'full_status':after,
                                'change':'new_tp' if before!='TP' and after=='TP' else 'lost_tp' if before=='TP' and after!='TP' else 'unchanged'})
        write_csv(run.out/'restoration_paired_changes.csv',changes)
        fpchanges=[]
        for name,rows in matches.items():
            fpchanges.extend(pair_false_positives(rows,matches['REST_FULL'],name,cfg.get('evaluation',{})))
        write_csv(run.out/'restoration_fp_changes.csv',fpchanges)


def export_mot(obs,out):
    for rec,df in obs.groupby('recording_id'):
        path=out/'motchallenge'/str(rec)/'data.txt';path.parent.mkdir(parents=True,exist_ok=True)
        lines=[]
        for _,x in df.iterrows():lines.append(f'{int(x.frame_idx)+1},{int(x.track_id)},{x.x1},{x.y1},{x.x2-x.x1},{x.y2-x.y1},1,-1,-1,-1')
        path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(out/'motchallenge'/'coordinate_protocol.json',{'internal_frame':'0-based global','mot_frame':'global frame + 1','bbox':'top-left xywh pixels','confidence':'output inclusion sentinel, not detector native confidence'})


def tracking_windows(obs,reference,mapping,protocol):
    pre=int(protocol.get('pre_onset_window',10));half=int(protocol.get('window_half_width',5));side=int(protocol.get('transition_support',3))
    if min(pre,half,side)<1:raise ValueError('Positive window sizes required')
    result=[]
    for _,gt in reference.iterrows():
        matched=mapping[(mapping.recording_id==gt.recording_id)&(mapping.ref_object_id==gt.ref_object_id)] if not mapping.empty else mapping
        if not matched.empty:
            chosen=matched.sort_values(['shared_frames','purity'],ascending=False).iloc[0]
            track=obs[(obs.recording_id==gt.recording_id)&(obs.track_id==chosen.track_id)]
        else:track=obs.iloc[:0]
        onset=int(gt.onset_frame);before=set(range(onset-pre,onset));window=set(range(onset-half,onset+half+1))
        roi=set(track.frame_idx.astype(int));det=set(track[track.detection_supported].frame_idx.astype(int))
        result.append({'recording_id':gt.recording_id,'ref_event_id':gt.ref_event_id,'ref_object_id':gt.ref_object_id,
                       'boundary_policy':'fixed_denominator','pre_denominator':pre,'window_denominator':2*half+1,
                       'roi_pre_onset_coverage':len(roi&before)/pre,'detector_pre_onset_coverage':len(det&before)/pre,
                       'roi_window_completeness':len(roi&window)/(2*half+1),'detector_window_completeness':len(det&window)/(2*half+1),
                       'observable_transition':len(roi&set(range(onset-half,onset)))>=side and len(roi&set(range(onset,onset+half+1)))>=side})
    return result


def pair_false_positives(before,after,name,protocol):
    from .evaluation import optimal_pairs
    a=[x for x in before if x['match_status']=='FP'];b=[x for x in after if x['match_status']=='FP']
    tau=protocol.get('fp_pair_time_tolerance');distance=protocol.get('fp_pair_distance')
    if tau is None or distance is None:
        return [{'baseline':name,'status':'not_paired','reason':'Explicit fp_pair_time_tolerance/fp_pair_distance required','baseline_fp_N':len(a),'full_fp_N':len(b)}]
    edges=np.zeros((len(a),len(b)),bool);cost=np.zeros(edges.shape)
    for i,x in enumerate(a):
        for j,y in enumerate(b):
            if x['recording_id']!=y['recording_id']:continue
            if any(v is None for v in [x.get('location_x'),x.get('location_y'),y.get('location_x'),y.get('location_y')]):continue
            dt=abs(x['onset_frame']-y['onset_frame']);ds=np.hypot(x['location_x']-y['location_x'],x['location_y']-y['location_y'])
            compatible=not (pd.notna(x.get('ref_object_id')) and pd.notna(y.get('ref_object_id')) and str(x['ref_object_id'])!=str(y['ref_object_id']))
            edges[i,j]=compatible and dt<=tau and ds<=distance;cost[i,j]=dt/max(tau,1)+ds/max(distance,1e-9)
    pairs=optimal_pairs(edges,cost);used_a={i for i,j in pairs};used_b={j for i,j in pairs};rows=[]
    for i,j in pairs:rows.append({'baseline':name,'change':'unchanged_fp','baseline_pred_event_id':a[i]['pred_event_id'],'full_pred_event_id':b[j]['pred_event_id']})
    for i,x in enumerate(a):
        if i not in used_a:rows.append({'baseline':name,'change':'removed_fp','baseline_pred_event_id':x['pred_event_id'],'full_pred_event_id':None})
    for j,y in enumerate(b):
        if j not in used_b:rows.append({'baseline':name,'change':'new_fp','baseline_pred_event_id':None,'full_pred_event_id':y['pred_event_id']})
    return rows
