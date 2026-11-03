"""Review-v2 experiment registry, full-input preflight, independent/combined runner.

Every executable case resolves to a one-method config. Review tasks needing new
human evidence are explicit blocked jobs, rather than fabricated experiments.
"""
from __future__ import annotations
import argparse
import copy
import fnmatch
import json
from pathlib import Path
import re
import time
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import pandas as pd
from .common import PROJECT, Blocked, digest, file_hash, load_config, write_csv, write_json

DEFAULT_SETTINGS={
    'project_root':str(PROJECT),'input_root':'review_v2_inputs',
    'output_root':'revision_outputs/review_v2_full',
    'event_label_policy':'historical_implicit_negatives',
    'cleaned_detections_dir':str(PROJECT.parent/'locate_results/run_20260703_104159/postprocessed/cleaned_detections_txt'),
    'reply_document':str(PROJECT/'审稿意见回复v2.docx'),
    'parameter_design':'retrospective_prespecified_diagnostic_not_independent_tuning',
    'manual_inputs':{}, 'config_overrides':{}, 'job_overrides':{},
}


def registry():
    result={}
    def add(name,group,family=None,settings=None,reason=None):
        result[name]={'id':name,'group':group,'family':family,'settings':settings or {},'blocked_reason':reason}
    for name,backend in [('H','hungarian'),('O','ot')]:
        for rescue in (0,1):add(f'ASSIGN_{name}_R{rescue}','assignment','assignment',{'id':f'ASSIGN_{name}_R{rescue}','backend':backend,'rescue':bool(rescue)})
    from .tracking import restoration_variants
    for variant in restoration_variants():add(variant['id'],'restoration','restoration',variant)
    add('MOT_SORT','native_mot','assignment',{'id':'MOT_SORT','backend':'sort',
        'source_root':str(PROJECT/'revision_external/sort'),'source_file':str(PROJECT/'revision_external/sort/sort.py'),
        'source_sha256':'41d30e3c3cf94079fad57aa67152704a73837cce27257be2890554b5921652b9',
        'source_revision':'2236dff5019565958b84df7d871d41cc1db58ac7','settings':{'max_age':5,'min_hits':3,'iou_threshold':.3}})
    from .mot_adapters import author_variant
    for backend in ('bytetrack','ocsort'):
        variant=author_variant(backend)
        add(variant['id'],'native_mot','assignment',variant)
    judges={
        'J_GMM':{'method':'gmm','rtv':False},'J_GMM_RTV':{'method':'gmm','rtv':True},
        'J_HMM':{'method':'hmm','rtv':False,'apply_legacy_screens':True},
        'J_GRAY_LEGACY':{'method':'legacy_baseline','legacy_method':'gray_p99_threshold'},
        'J_CHANGE_LEGACY':{'method':'legacy_baseline','legacy_method':'multi_feature_change_score'},
        'J_CHANGE_LEGACY_RTV':{'method':'legacy_baseline','legacy_method':'multi_feature_change_score','rtv':True},
        'SAS_PAPER_ADAPTED':{'method':'sas_paper','normalization':'unit','pixel_threshold':.45,'mean_threshold':.20,'area_unit':'px2','area_threshold':4.,'min_run':1},
        'DOG_PAPER_ADAPTED':{'method':'dog_paper','sigmas':[1.,2.,3.,4.],'sigma_ratio':1.6,'polarity':'bright','response_threshold':.03,'min_run':1},
        'CNF_ROI_STAGE_ADAPTED':{'method':'cnf_roi_stage','mask_mode':'center_component','mask_polarity':'bright','normalization':'unit',
                                 'mask_threshold':.25,'min_low':3,'min_rise':3,'min_high':3,'boundary_step':1,
                                 'min_amplitude':.10,'min_slope_per_frame':.005,'min_bic_improvement':10.,'max_rmse':.15},
        'J_PELT':{'method':'pelt','columns':['gray_p99'],'direction_column':'gray_p99','cost_model':'l2',
                  'penalty':2000.,'min_size':3,'jump':1,'min_increase':25.},
    }
    for name,s in judges.items():
        s=copy.deepcopy(s);s.setdefault('rtv',False)
        if s['method']=='legacy_baseline':
            s['params']={'gray_threshold':.5,'change_threshold':.25,'gray_min_persist':3,'change_min_persist':3}
        s['parameter_provenance']='explicit expert initial settings / historical defaults; retrospective sensitivity, not independently optimized test settings'
        add(name,'judges','judges',s)
        if name in ('J_HMM','SAS_PAPER_ADAPTED','DOG_PAPER_ADAPTED','CNF_ROI_STAGE_ADAPTED','J_PELT'):
            add(name+'_RTV','rtv_pairs','judges',{**copy.deepcopy(s),'rtv':True})
    from .prior9 import SETTINGS as prior9_settings
    add('J_PRIOR9','judges','judges',copy.deepcopy(prior9_settings))
    from .experiments import decision_variants
    decisions=decision_variants({'background_factor':{'kind':'independent_columns','columns':['bg_med','bg_mad']}})
    for name,s in decisions.items():
        group='geometry_background' if name.startswith('GEOM_') else 'features' if name.startswith('FEAT_') else 'decision'
        add(name,group,'judges',s)
    for method in ['SAS_PAPER_ADAPTED','DOG_PAPER_ADAPTED','CNF_ROI_STAGE_ADAPTED','J_PELT']:
        field={'SAS_PAPER_ADAPTED':'mean_threshold','DOG_PAPER_ADAPTED':'response_threshold',
               'CNF_ROI_STAGE_ADAPTED':'min_amplitude','J_PELT':'penalty'}[method]
        for ratio in [.75,1.,1.25]:
            name=f'SENS_{method}_{ratio:g}';s=copy.deepcopy(judges[method]);s[field]*=ratio
            s['parameter_provenance']='prespecified one-factor sensitivity, no automatic winner selection'
            add(name,'robustness','judges',s)
    for gate in (32.,45.):
        name=f'SENS_GATE_{gate:g}'
        add(name,'robustness','assignment',{'id':name,'backend':'ot','rescue':True,
            'restoration':{'gap':True,'prebirth':True},'tracking_overrides':{'max_center_dist':gate}})
    for ratio in (.75,1.,1.25):
        thresholds={'min_raw_delta_gray_p99':25.,'min_raw_delta_corr_p99':20.,'min_raw_delta_bright_energy':8.,
                    'min_raw_delta_bright_area':.04,'min_raw_delta_hsv_v_p99':25.}
        add(f'SENS_RTV_{ratio:g}','robustness','judges',{'method':'gmm','rtv':True,
            'gmm_overrides':{k:v*ratio for k,v in thresholds.items()},'parameter_provenance':'prespecified RTV threshold scaling; not chosen by test performance'})
    for name,selection in [('DET_COLD_RAW','DET_COLD_raw'),('DET_HOT_RAW','DET_HOT_raw'),('DET_BOTH_RAW','DET_BOTH_raw'),
                           ('DET_BOTH_CLEANED','DET_BOTH_cleaned'),('DET_LOG','DET_log'),('DET_MSER','DET_mser')]:
        add(name,'detection','detection',{'selection':selection})
    for rule in ['background','geometry','defocus','nested','duplicates','nms','all']:
        add('CLEAN_NO_'+rule.upper(),'cleaning','detection',{'selection':'DET_BOTH_cleaned','disabled_rules':[rule]})
    # Historical defocus filter is off. Define an explicit on/off diagnostic
    # pair rather than selling a no-op switch-off as a contribution ablation.
    add('CLEAN_WITH_DEFOCUS','cleaning','detection',{'selection':'DET_BOTH_cleaned','enable_defocus_filter':True})
    result['CLEAN_NO_DEFOCUS']['settings'].update({'enable_defocus_filter':True,'comparison_reference':'CLEAN_WITH_DEFOCUS'})
    for name in ['AGREEMENT','LOCATION_ERROR','ERROR_CAUSES','APPLICATION_REVIEW','COMBUSTION_MEASUREMENTS','ILLUMINATION_STRATA','QUALITY_STRATA']:
        add(name,'human_analysis','analysis',{'analysis':name})
    add('DATA_AUDIT','audit','data_audit')
    add('PARAMETER_REGISTRY','audit','parameters')
    add('OLD_RTV_AUDIT','audit','old_rtv')
    add('TABLE_FIGURE_AUDIT','audit','table_figure')
    add('EXISTING_MOT_METRICS','mot_metrics','official_mot')
    add('BUILD_FPN_CORRECTED','fpn','build_fpn')
    add('TRAIN_FPN_CORRECTED','fpn','train-fpn')
    add('TEST_FPN_CORRECTED','fpn','fpn-pipeline')
    for name,family,ref in [('RUNTIME_JUDGMENT','judges','J_GMM_RTV'),('RUNTIME_TRACKING','assignment','ASSIGN_O_R1')]:
        add(name,'runtime','benchmark',{'family':family,'ref':ref})
    add('RUNTIME_VLM','runtime',reason='Real VLM model/revision/prompts and live uncached calls required; cached candidates cannot measure VLM inference')
    add('LORO','generalization',reason='Only one physical recording is currently prepared; independent acquisition manifests and labels required')
    add('BACK_TO_FRONT','generalization',reason='Independent front-illumination recording and training/tuning usage facts required')
    add('PROMPT_SENSITIVITY','detection',reason='Additional actual prompt responses/candidates required; do not relabel existing boxes as a new prompt')
    return result


COMMENT_JOBS={
    1:{1:['ASSIGN_*'],2:['DEC_NO_GEOM','DEC_FULL'],3:['FEAT_*','GEOM_*'],4:[]},
    2:{1:[],2:['DATA_AUDIT','AGREEMENT'],3:['DET_*','CLEAN_*','PROMPT_SENSITIVITY'],4:['RUNTIME_*'],5:['ERROR_CAUSES','LOCATION_ERROR','DEC_*'],6:['PARAMETER_REGISTRY','DATA_AUDIT']},
    3:{1:['J_*','SAS_*','DOG_*','CNF_*','TEST_FPN_CORRECTED'],2:['DET_*'],3:['REST_*'],4:['LORO','BACK_TO_FRONT','SENS_*'],
       5:[],6:[],7:[],8:['ILLUMINATION_STRATA','BACK_TO_FRONT'],9:[],10:[],11:['J_GMM_RTV'],12:['AGREEMENT'],
       13:['REST_*'],14:[],15:[],16:['J_GRAY_LEGACY','J_CHANGE_LEGACY'],17:['TABLE_FIGURE_AUDIT'],
       18:['COMBUSTION_MEASUREMENTS'],19:['J_GMM_RTV','LOCATION_ERROR']},
    5:{1:['AGREEMENT'],2:['DATA_AUDIT','LORO'],3:['REST_*','DEC_*'],4:['J_PRIOR9','MOT_*','J_PELT','ASSIGN_*'],
       5:['COMBUSTION_MEASUREMENTS','APPLICATION_REVIEW'],6:['AGREEMENT'],7:['DATA_AUDIT','LORO','BACK_TO_FRONT'],
       8:['J_PRIOR9','J_GRAY_LEGACY','J_CHANGE_LEGACY'],9:['J_*','SAS_*','DOG_*','CNF_*'],10:['J_PELT'],
       11:['MOT_*','ASSIGN_*','EXISTING_MOT_METRICS'],12:['DET_*','PROMPT_SENSITIVITY','RUNTIME_VLM'],
       13:['APPLICATION_REVIEW','COMBUSTION_MEASUREMENTS'],14:['PARAMETER_REGISTRY'],15:[],16:['TABLE_FIGURE_AUDIT','SENS_GATE_*'],
       17:['OLD_RTV_AUDIT','J_GMM','J_GMM_RTV'],18:['TABLE_FIGURE_AUDIT','J_*'],19:[],20:[],
       21:['LOCATION_ERROR'],22:['ILLUMINATION_STRATA','QUALITY_STRATA'],23:['J_GMM_RTV'],24:['COMBUSTION_MEASUREMENTS'],25:[],26:[]}}


def document_map(settings, jobs):
    path=Path(settings['reply_document'])
    if not path.is_absolute():path=Path(settings['project_root'])/path
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with ZipFile(path) as z:doc=ET.fromstring(z.read('word/document.xml'))
    paras=[''.join(n.text or '' for n in p.findall('.//w:t',ns)) for p in doc.findall('.//w:p',ns)]
    reviewer=None;comment=None;rows=[]
    for i,text in enumerate(paras):
        m=re.match(r'Reviewer\s*#(\d+)',text)
        if m:reviewer=int(m[1])
        m=re.match(r'Comment\s+(\d+)\s*:',text)
        if not m:continue
        comment=int(m[1]);body=[]
        for t in paras[i+1:]:
            if t.strip().startswith('Reply:'):break
            if t.strip():body.append(t.strip())
        if reviewer not in COMMENT_JOBS or comment not in COMMENT_JOBS[reviewer]:raise ValueError(f'Unmapped new v2 comment {reviewer}/{comment}')
        patterns=COMMENT_JOBS[reviewer][comment]
        linked=[name for name in jobs if any(fnmatch.fnmatchcase(name,pat) for pat in patterns)]
        if patterns and not linked:raise ValueError('Empty experiment coverage')
        rows.append({'reviewer':reviewer,'comment':comment,'source_paragraph':i,'requirement':' '.join(body),
                     'experiment_ids':';'.join(linked),'scope':'experiment_or_analysis' if linked else 'manuscript_or_reference_edit',
                     'source_sha256':file_hash(path)})
    if len(rows)!=55:raise ValueError(f'Expected reviewed v2 with 55 comments, got {len(rows)}; inspect document changes')
    return rows


def config_for(job,settings,data):
    root=Path(settings['project_root']);family=job['family'];name=job['id']
    meta=json.loads((data/'dataset_meta.json').read_text(encoding='utf-8'))
    cfg={'project_root':str(root),'output_dir':str(root/settings['output_root']/'jobs'/name),
         'synthetic':False,'protocol_status':'frozen','evaluation_mode':'retrospective_corrected_reference',
         'parameter_design':settings['parameter_design'],'seed':2026,
         'frames':str(data/'frames.csv'),'candidates':str(data/'candidates.csv'),
         'reference_boxes':str(data/'reference_boxes.csv'),'reference_events':str(data/'reference_events.csv'),
         'rois':str(data/'rois.csv'),'labels':str(data/'labels.csv'),
         'label_provenance':str(data/'dataset_meta.json'),'track_bundle_id':meta['source_hashes']['boxes'],
         'feature_cache_dir':str(root/settings['output_root']/'shared_feature_cache'),
         'features':{'difference_policy':'contiguous_zero_boundary','background_correction':True},
         'evaluation':{'iou_threshold':.5,'mapping_max_distance':30.,'mapping_min_shared_frames':3,
                       'mapping_min_purity':.8,'tolerances':[1,3],'duration_s':7997/50000,
                       'fp_pair_time_tolerance':3,'fp_pair_distance':30.},
         'standard_mot_evaluation':{'source_root':'revision_external/TrackEval_compat',
            'source_revision':'12c8791b303e0a0b50f753af204249e622d0281a + numpy_compatibility_patch.json','iou_threshold':.5},
         'gmm_config':{'random_state':2026},'judge':{'method':'gmm','rtv':True},
         'tracking':{},'time_advance_policy':'incremental_state_frame_v1','crop_padding':0,'virtual_tracking_rois':True,
         '_review_v2_scope':{'frames':7997,'reference_boxes':128688,'fixed_cohort':meta['fixed_cohort']}}
    if family=='judges':cfg['methods']={name:copy.deepcopy(job['settings'])}
    if family in ('assignment','restoration'):
        variant=copy.deepcopy(job['settings']);cfg['tracking'].update(variant.pop('tracking_overrides',{}));cfg['variants']=[variant]
        if name in ('MOT_BYTE','MOT_OCSORT'):
            overrides=settings.get('native_mot_overrides',{}).get(name,{})
            unknown=set(overrides)-{'settings','score_policy'}
            if unknown:raise ValueError('Unknown native_mot_overrides: '+str(unknown))
            for key,value in overrides.items():variant[key].update(copy.deepcopy(value))
    if family in ('assignment','restoration') and settings['event_label_policy']=='reviewed_only':
        cfg['tracking_evaluation']='mot_only'
    if family=='detection':
        cfg.update({'frames':str(data/'detector_frames.csv'),'reference_boxes':str(data/'detector_boxes.csv'),
                    'candidates':str(data/'detector_candidates.csv'),'available_prompt_groups':['cold','hot'],
                    'selected_method_ids':[job['settings']['selection']],
                    'cleaning':{'disabled_rules':job['settings'].get('disabled_rules',[])},
                    'classical':{'log':{'polarity':'dark','sigmas':[1.,2.,3.,4.],'response_threshold':.02},
                                 'mser':{'polarity':'dark','min_area':4,'max_area':4096}},
                    '_review_v2_scope':{'frames':300,'reference_boxes':4273}})
        if name=='DET_LOG':
            # Change only the LoG case; unrelated detector presets retain their
            # saved configuration identity for recovery of completed experiments.
            cfg['classical']['log']={'polarity':'dark','intensity_scale':'unit',
                'sigmas':[2.,3.,4.,6.,8.,12.,16.,24.,32.,48.],
                'response_threshold':.02,'nms_iou':.3}
        if 'enable_defocus_filter' in job['settings']:
            cfg['cleaning']['enable_defocus_filter']=job['settings']['enable_defocus_filter']
        if job['settings'].get('comparison_reference'):cfg['comparison_reference']=job['settings']['comparison_reference']
    if family=='benchmark':
        base=registry()[job['settings']['ref']]
        cfg={'project_root':str(root),'output_dir':cfg['output_dir'],'synthetic':False,'protocol_status':'frozen',
             'benchmark_family':job['settings']['family'],'warmup_repeats':1,'measurement_repeats':3,
             'experiment':config_for(base,settings,data)}
    if family in ('train-fpn','fpn-pipeline'):
        source=root/'revision_configs'/('train_fpn_ignition.development.json' if family=='train-fpn' else 'fpn_ignition_pipeline.test_only.json')
        cfg=json.loads(source.read_text(encoding='utf-8'));cfg.update({'project_root':str(root),'output_dir':str(root/settings['output_root']/'jobs'/name),
            'dataset_dir':str(data/'fpn_corrected'),'protocol_status':'frozen','synthetic':False})
        if family=='fpn-pipeline':
            cfg['track_bundle_id']=meta['source_hashes']['boxes']
            cfg['label_provenance']=str(data/'dataset_meta.json')
            cfg['evaluation_mode']='corrected_same_recording_test_complete_lifetimes'
            status=root/settings['output_root']/'suite_status.json'
            records=json.loads(status.read_text(encoding='utf-8')) if status.exists() else []
            complete=[r for r in records if r['id']=='TRAIN_FPN_CORRECTED' and r['status']=='completed']
            explicit=settings.get('config_overrides',{}).get(name,{}).get('checkpoint')
            if not complete and not explicit:raise Blocked('Run TRAIN_FPN_CORRECTED first, or supply an explicit corrected checkpoint override')
            cfg['checkpoint']=explicit or str(Path(complete[-1]['run_dir'])/'best.pt')
            cfg['frames']=str(data/'fpn_test/frames.csv')
            cfg['shared_tracks']=str(data/'fpn_test/shared_tracks.csv')
            cfg['cohort']=str(data/'fpn_test/cohort.csv');cfg['labels']=str(data/'fpn_test/labels.csv')
    cfg.update(copy.deepcopy(settings.get('config_overrides',{}).get(name,{})))
    return cfg


def validate_scope(cfg):
    expected=cfg.get('_review_v2_scope',{})
    for key,column in [('frames',None),('reference_boxes',None),('fixed_cohort','labels')]:
        if key not in expected:continue
        table=pd.read_csv(cfg[column or key],low_memory=False)
        if len(table)!=expected[key]:raise ValueError(f'Full-scope guard: {key} expected {expected[key]}, found {len(table)}')
    if 'fixed_cohort' in expected:
        rois=pd.read_csv(cfg['rois'],usecols=['recording_id','track_id'])
        labels=pd.read_csv(cfg['labels'],usecols=['recording_id','track_id'])
        if set(map(tuple,rois.drop_duplicates().to_numpy()))!=set(map(tuple,labels.to_numpy())):
            raise ValueError('Full labelled cohort not covered by ROI input')


def select(jobs,groups=None,ids=None,exclude=None):
    def parts(s):return [x.strip() for x in (s or '').split(',') if x.strip()]
    group_set=set(parts(groups));patterns=parts(ids);disabled=parts(exclude)
    unknown=group_set-{j['group'] for j in jobs.values()}
    if unknown:raise ValueError('Unknown groups: '+str(sorted(unknown)))
    for pat in patterns+disabled:
        if not any(fnmatch.fnmatchcase(k,pat) for k in jobs):raise ValueError('Unknown experiment pattern: '+pat)
    return [j for k,j in jobs.items() if (not group_set or j['group'] in group_set)
            and (not patterns or any(fnmatch.fnmatchcase(k,x) for x in patterns))
            and not any(fnmatch.fnmatchcase(k,x) for x in disabled)]


def collect_results(history,out,data):
    """Join independent runs by reference event ID, never by algorithm track ID."""
    from .evaluation import common_tp
    comparisons=[];members={};tracking=[];track_members={};detectors=[];runtime=[];prediction_paths={};fpn_test=None
    for r in history:
        if r['status']!='completed' or not r.get('run_dir') or r.get('input_root')!=str(data):continue
        run=Path(r['run_dir']);name=r['id']
        if (run/'S03_detector_comparison.csv').exists():
            try:
                table=pd.read_csv(run/'S03_detector_comparison.csv');table['suite_id']=name;table['suite_group']=r['group'];detectors.append(table)
            except pd.errors.EmptyDataError:pass
        if (run/'runtime_statistics.json').exists():
            runtime.append({'suite_id':name,**json.loads((run/'runtime_statistics.json').read_text(encoding='utf-8'))})
        if name=='TEST_FPN_CORRECTED':
            path=run/'FPN_IGNITION_SHARED_TRACKS/predictions.csv'
            if path.exists():fpn_test=path
        if (run/name/'predictions.csv').exists():prediction_paths[name]=run/name/'predictions.csv'
        score=run/'judge_comparison_fixed_tracks.csv';member=run/name/'evaluation_members.csv'
        if score.exists() and member.exists():
            try:
                table=pd.read_csv(score);table['suite_group']=r['group'];comparisons.append(table)
                members[name]=pd.read_csv(member).where(pd.notna(pd.read_csv(member)),None).to_dict('records')
            except pd.errors.EmptyDataError:pass
        for filename in ('S04_assignment_and_mot.csv','S05_restoration_events.csv'):
            if (run/filename).exists():
                try:tracking.append(pd.read_csv(run/filename))
                except pd.errors.EmptyDataError:pass
        if (run/name/'event_matches.csv').exists():
            track_members[name]=pd.read_csv(run/name/'event_matches.csv').to_dict('records')
    aggregate=out/'combined';aggregate.mkdir(parents=True,exist_ok=True)
    if detectors:write_csv(aggregate/'all_detector_and_cleaning_metrics.csv',pd.concat(detectors,ignore_index=True))
    if runtime:write_csv(aggregate/'runtime_statistics.csv',runtime)
    if comparisons:
        write_csv(aggregate/'all_fixed_cohort_metrics.csv',pd.concat(comparisons,ignore_index=True))
        groups={r['id']:r['group'] for r in history}
        for group in ('judges','decision','features','geometry_background','rtv_pairs','robustness'):
            selected={k:v for k,v in members.items() if groups[k]==group}
            if selected:
                table,ids=common_tp(selected);write_csv(aggregate/(group+'_common_tp.csv'),table)
                write_json(aggregate/(group+'_common_tp_members.json'),{'methods':list(selected),'N':len(ids),'members':ids,'hash':digest(ids)})
        pairs=[];rtv=[]
        import itertools
        for a,b in itertools.combinations(members,2):
            table,ids=common_tp({a:members[a],b:members[b]})
            pairs.extend({**row,'comparison':a+'__'+b,'N_common':len(ids)} for row in table)
        for after in members:
            before=after[:-4] if after.endswith('_RTV') else None
            if after=='DEC_FULL':before='DEC_NO_RTV'
            if before not in members:continue
            a={r['ref_event_id']:r for r in members[before]};b={r['ref_event_id']:r for r in members[after]}
            for eid in a.keys()&b.keys():rtv.append({'before':before,'after':after,'ref_event_id':eid,
                                                   'before_status':a[eid]['match_status'],'after_status':b[eid]['match_status']})
        write_csv(aggregate/'pairwise_common_tp.csv',pairs);write_csv(aggregate/'rtv_member_changes.csv',rtv)
    if fpn_test is not None:
        from .evaluation import evaluate_fixed
        labels=pd.read_csv(data/'fpn_test/labels.csv');predictions={'FPN':pd.read_csv(fpn_test)};results=[];sets={};pairs=[]
        for name,path in prediction_paths.items():
            if name in ('J_GMM_RTV','J_GMM','SAS_PAPER_ADAPTED','DOG_PAPER_ADAPTED','CNF_ROI_STAGE_ADAPTED','J_PELT'):
                table=pd.read_csv(path);predictions[name]=table.loc[table.track_id.isin(labels.track_id)]
        for name,pred in predictions.items():
            score,matched=evaluate_fixed(pred,labels);results.append({'method':name,'scope':'corrected same-recording test complete lifetimes',**score});sets[name]=matched
        import itertools
        for a,b in itertools.combinations(sets,2):
            table,ids=common_tp({a:sets[a],b:sets[b]});pairs.extend({**x,'comparison':a+'__'+b,'N_common':len(ids)} for x in table)
        write_csv(aggregate/'fpn_test_comparison.csv',results);write_csv(aggregate/'fpn_test_pairwise_common_tp.csv',pairs)
    if tracking:write_csv(aggregate/'all_tracking_metrics.csv',pd.concat(tracking,ignore_index=True))
    if 'REST_FULL' in track_members:
        full={r['ref_event_id']:r for r in track_members['REST_FULL'] if r.get('ref_event_id') and pd.notna(r['ref_event_id'])}
        rows=[];fp=[]
        from .experiments import pair_false_positives
        for method,table in track_members.items():
            if not method.startswith('REST_'):continue
            current={r['ref_event_id']:r for r in table if r.get('ref_event_id') and pd.notna(r['ref_event_id'])}
            for eid in full:
                a=current.get(eid,{}).get('match_status','FN');b=full[eid]['match_status']
                rows.append({'baseline':method,'full':'REST_FULL','ref_event_id':eid,'baseline_status':a,'full_status':b,
                             'change':'new_tp' if a!='TP' and b=='TP' else 'lost_tp' if a=='TP' and b!='TP' else 'unchanged'})
            fp.extend(pair_false_positives(table,track_members['REST_FULL'],method,{'fp_pair_time_tolerance':3,'fp_pair_distance':30.}))
        write_csv(aggregate/'restoration_paired_changes.csv',rows);write_csv(aggregate/'restoration_fp_changes.csv',fp)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['list','prepare','check','run','report'])
    p.add_argument('--settings',type=Path)
    p.add_argument('--groups',help='Comma-separated group names')
    p.add_argument('--ids',help='Comma-separated exact IDs or glob patterns, e.g. ASSIGN_*')
    p.add_argument('--exclude','--disable',dest='exclude',help='Disable exact IDs/glob patterns')
    p.add_argument('--resume',action='store_true');p.add_argument('--fail-fast',action='store_true')
    p.add_argument('--device');args=p.parse_args(argv)
    settings=copy.deepcopy(DEFAULT_SETTINGS)
    if args.settings:settings.update(json.loads(args.settings.read_text(encoding='utf-8-sig')))
    jobs=registry()
    for name,overrides in settings.get('job_overrides',{}).items():
        if name not in jobs:raise ValueError('Unknown job override: '+name)
        jobs[name].update(copy.deepcopy(overrides))
    out=Path(settings['project_root'])/settings['output_root'];out.mkdir(parents=True,exist_ok=True)
    coverage=document_map(settings,jobs);write_csv(out/'reviewer_comment_coverage.csv',coverage)
    write_json(out/'experiment_registry.json',jobs)
    from .review_v2_analysis import templates
    templates(out/'human_input_templates')
    chosen=select(jobs,args.groups,args.ids,args.exclude)
    if not chosen:raise ValueError('No enabled experiments')
    if args.command=='list':
        for j in chosen:print(f"{j['id']:36} {j['group']:20} {j['blocked_reason'] or 'implemented'}")
        print(f'{len(chosen)} selected cases; 55 reviewer comments mapped.');return 0
    if args.command=='report':
        print((out/'suite_status.json').read_text(encoding='utf-8') if (out/'suite_status.json').exists() else 'No suite execution yet.');return 0
    from .review_v2_data import prepare
    data=prepare(settings)
    protocol=out/'suite_input_protocol.json'
    input_protocol={'input_root':str(data),'dataset_meta_sha256':file_hash(data/'dataset_meta.json'),
                    'event_label_policy':settings['event_label_policy'],'parameter_design':settings['parameter_design']}
    if protocol.exists() and json.loads(protocol.read_text(encoding='utf-8'))!=input_protocol:
        raise ValueError('Cannot mix different data/label protocols in one output_root')
    write_json(protocol,input_protocol)
    if args.command=='prepare':
        print((data/'dataset_meta.json').read_text(encoding='utf-8'));return 0
    statusfile=out/'suite_status.json'
    history=json.loads(statusfile.read_text(encoding='utf-8')) if statusfile.exists() else []
    from .common import Run
    from .cli import run_family
    current=[]
    for job in chosen:
        begin=time.perf_counter();record={'id':job['id'],'group':job['group'],'status':'planned','input_root':str(data)}
        try:
            if job['blocked_reason']:raise Blocked(job['blocked_reason'])
            if job['family']=='analysis':
                from .review_v2_analysis import run_analysis
                path=settings.get('manual_inputs',{}).get(job['id'])
                if not path:raise Blocked(f"Supply manual_inputs.{job['id']} CSV; see human input templates")
                path=Path(path)
                if not path.is_absolute():path=Path(settings['project_root'])/path
                if not path.is_file():raise Blocked('Manual input missing: '+str(path))
                if args.command=='run':run_analysis(job['id'],path,out/'analyses'/job['id'])
            elif job['family'] in ('data_audit','parameters','old_rtv','table_figure','official_mot','build_fpn'):
                if args.command=='run':
                    from .review_v2_analysis import run_special
                    run_special(job,settings,data,out)
            else:
                cfg=config_for(job,settings,data)
                if args.device:cfg['device']=args.device
                validate_scope(cfg)
                if job['family']=='assignment':
                    for variant in cfg.get('variants',[]):
                        if variant.get('source_file') and file_hash(variant['source_file'])!=variant['source_sha256']:raise ValueError('Native author source hash mismatch')
                        if variant.get('backend') in ('bytetrack','ocsort'):
                            from .mot_adapters import load_author_module,prepare_scores
                            load_author_module(cfg,variant)
                            prepare_scores(pd.read_csv(cfg['candidates']),variant)
                if job['family'] in ('train-fpn','fpn-pipeline') and not (data/'fpn_corrected/dataset_meta.json').exists():raise Blocked('Run BUILD_FPN_CORRECTED first')
                path=out/'resolved_cases'/(job['id']+'.json');write_json(path,cfg)
                record['config']=str(path)
                if args.command=='run':
                    run=Run(cfg,job['family'],resume=args.resume)
                    record['run_dir']=str(run.out)
                    if not run.skip:
                        try:run_family(job['family'],cfg,run);run.finish()
                        except Exception as e:run.finish(e);raise
                        record['status']=run.meta['status']
                        if record['status']!='completed':
                            record['reason']='; '.join(str(s.get('reason')) for s in run.statuses if s.get('reason'))
                    else:record['status']='completed';record['verified_resume']=True
            if record['status']=='planned':record['status']='ready' if args.command=='check' else 'completed'
        except Blocked as e:record.update(status='blocked',reason=str(e))
        except Exception as e:record.update(status='failed',reason=f'{type(e).__name__}: {e}')
        record['elapsed_s']=time.perf_counter()-begin
        current.append(record)
        if args.command=='check':
            target=out/'preflight_status.json';prior=json.loads(target.read_text(encoding='utf-8')) if target.exists() else []
            prior=[r for r in prior if r['id']!=record['id']]+[record];write_json(target,prior)
        else:
            history=[r for r in history if r['id']!=record['id']]+[record];write_json(statusfile,history);write_csv(out/'suite_status.csv',history)
        print(json.dumps(record,ensure_ascii=False),flush=True)
        if args.fail_fast and record['status'] in ('blocked','failed','partial'):break
    if args.command=='check':return 1 if any(r['status']=='failed' for r in current) else 2 if any(r['status']=='blocked' for r in current) else 0
    collect_results(history,out,data)
    return 1 if any(r['status']=='failed' for r in history if r['id'] in {j['id'] for j in chosen}) else 2 if any(r['status'] in ('blocked','partial') for r in history if r['id'] in {j['id'] for j in chosen}) else 0
