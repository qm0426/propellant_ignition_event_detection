#!/usr/bin/env python3
"""Plot registered review-v2 jobs using the original manuscript plotting style.

Only completed, real jobs from one suite input protocol are eligible. Derived
views live inside the new figure output; source experiment files are unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from ignition_revision.common import write_csv, write_json
from ignition_revision.evaluation import common_tp, evaluate_fixed
from plot_revision_results import Plotter, GROUPS, read_csv


def feature_views(available):
    """Keep the three scientifically different feature references separate."""
    definitions = {
        'features_original': ('DEC_FULL', ['DEC_FULL','FEAT_NO_DELTA','FEAT_NO_COLOR','FEAT_NO_MORPH']),
        'features_background': ('FEAT_FULL_PLUS_BG', ['FEAT_FULL_PLUS_BG','FEAT_NO_BG']),
        'features_gray': ('FEAT_GRAY_BG', ['FEAT_GRAY_BG','FEAT_GRAY']),
    }
    return {key: (baseline, [x for x in ids if x in available], ids)
            for key, (baseline, ids) in definitions.items()}


def check_cohort(df):
    for key in ('cohort_hash','track_bundle_id','label_provenance','evaluation_scope'):
        if key in df and df[key].dropna().nunique()>1:
            raise ValueError('Mixed input protocols: '+key)


def build_views(suite, allowed_ids=None):
    history=json.loads((suite/'suite_status.json').read_text(encoding='utf-8'))
    protocol=json.loads((suite/'suite_input_protocol.json').read_text(encoding='utf-8'))
    data=Path(protocol['input_root'])
    fixed={};members={};predictions={};tables={};track_members={};runtime=[];sources={};excluded=[]
    def source(path):
        if path.is_file():sources[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    source(suite/'suite_status.json');source(suite/'suite_input_protocol.json');source(suite/'experiment_registry.json')
    fpn=None
    for record in history:
        if allowed_ids is not None and record['id'] not in allowed_ids:continue
        if record['status']!='completed' or not record.get('run_dir'):continue
        if Path(record['input_root']).resolve()!=data.resolve():
            raise ValueError('Job input root differs from suite protocol: '+record['id'])
        run=Path(record['run_dir']);name=record['id'];group=record['group']
        meta=json.loads((run/'run_meta.json').read_text(encoding='utf-8'))
        if meta.get('synthetic') or meta['status']!='completed':
            excluded.append({'id':name,'reason':'non-completed or synthetic run'});continue
        source(run/'run_meta.json');source(run/'resolved_config.json')
        if group=='runtime':runtime.append((run,meta));continue
        event_path=run/name/'event_matches.csv'
        if event_path.is_file():
            track_members[name]=read_csv(event_path).to_dict('records');source(event_path)
        path=run/'judge_comparison_fixed_tracks.csv';member=run/name/'evaluation_members.csv'
        if path.is_file() and member.is_file():
            table=read_csv(path)
            if set(table.method_id)!={name}:raise ValueError('Expected one registered method: '+name)
            fixed[name]=(group,table);members[name]=read_csv(member).to_dict('records')
            source(path);source(member)
            prediction=run/name/'predictions.csv'
            if prediction.is_file():predictions[name]=prediction;source(prediction)
        for filename,family in [('S03_detector_comparison.csv','detection'),
                                ('S04_assignment_and_mot.csv','assignment'),
                                ('S05_restoration_events.csv','restoration')]:
            path=run/filename
            if not path.is_file():continue
            table=read_csv(path)
            if table.empty:continue
            # Cleaning variants retain the same internal detector name. Their
            # suite IDs are the actual condition identifiers, not duplicates.
            if family=='detection':
                if len(table)!=1:raise ValueError('Expected a single detector condition: '+name)
                table['method_id']=name
            key=group if group!='robustness' else 'robustness_tracking'
            tables.setdefault((key,family),[]).append(table);source(path)
        if name=='TEST_FPN_CORRECTED':
            fpn=run/'FPN_IGNITION_SHARED_TRACKS/predictions.csv'
            if not fpn.is_file():raise ValueError('Missing FPN predictions')
            source(fpn)
    views=[]
    def fixed_view(key,ids,expected=None,baseline=None):
        ids=[x for x in ids if x in fixed]
        if not ids:return
        table=pd.concat([fixed[x][1] for x in ids],ignore_index=True);check_cohort(table)
        common,_=common_tp({x:members[x] for x in ids})
        transitions=[]
        for after in ids:
            before='DEC_NO_RTV' if after=='DEC_FULL' else after[:-4] if after.endswith('_RTV') else None
            if before not in ids:continue
            a={r['ref_event_id']:r for r in members[before]};b={r['ref_event_id']:r for r in members[after]}
            for eid in a.keys()&b.keys():
                transitions.append({'method_before':before,'method_after':after,'ref_event_id':eid,
                                    'before':a[eid]['match_status'],'after':b[eid]['match_status']})
        views.append({'key':key,'family':'ablations' if baseline else 'judges',
                      'table':table,'common':common,'baseline':baseline,
                      'extra_tables':{'S09_rtv_member_changes.csv':transitions} if transitions else {},
                      'missing':sorted(set(expected or ids)-set(ids))})
    registry=json.loads((suite/'experiment_registry.json').read_text(encoding='utf-8'))
    for group in ('decision','geometry_background','judges','rtv_pairs','robustness'):
        expected=[k for k,v in registry.items() if v['group']==group and v['family']=='judges']
        if not any(x in fixed for x in expected):continue
        ids=list(expected)
        if group=='rtv_pairs':ids += [x[:-4] for x in expected if x.endswith('_RTV')]
        fixed_view(group,ids,ids,'DEC_FULL' if group=='decision' else None)
    for key,(baseline,ids,expected) in feature_views(fixed).items():fixed_view(key,ids,expected,baseline)
    for (key,family),parts in tables.items():
        table=pd.concat(parts,ignore_index=True)
        if table.method_id.duplicated().any():raise ValueError('Duplicate suite condition: '+key)
        expected=[k for k,v in registry.items() if v['group']==key]
        views.append({'key':key,'family':family,'table':table,
                      'baseline':'REST_FULL' if family=='restoration' else None,
                      'missing':sorted(set(expected)-set(table.method_id))})
        matched={x:track_members[x] for x in table.method_id if x in track_members}
        if matched:views[-1]['common']=common_tp(matched)[0]
        if family=='restoration' and 'REST_FULL' in matched:
            from ignition_revision.experiments import pair_false_positives
            full={r['ref_event_id']:r for r in matched['REST_FULL'] if pd.notna(r.get('ref_event_id'))}
            changes=[];fps=[]
            for name,rows in matched.items():
                current={r['ref_event_id']:r for r in rows if pd.notna(r.get('ref_event_id'))}
                for eid,row in full.items():
                    a=current.get(eid,{}).get('match_status','FN');b=row['match_status']
                    changes.append({'baseline':name,'full':'REST_FULL','ref_event_id':eid,
                        'change':'new_tp' if a!='TP' and b=='TP' else 'lost_tp' if a=='TP' and b!='TP' else 'unchanged'})
                fps.extend(pair_false_positives(rows,matched['REST_FULL'],name,
                    {'fp_pair_time_tolerance':3,'fp_pair_distance':30.}))
            views[-1]['extra_tables']={'restoration_paired_changes.csv':changes,'restoration_fp_changes.csv':fps}
    if fpn is not None:
        label_path=data/'fpn_test/labels.csv';labels=read_csv(label_path);source(label_path)
        pool={'FPN_IGNITION_SHARED_TRACKS':read_csv(fpn)}
        for name,path in predictions.items():
            if name in ('J_GMM','J_GMM_RTV','SAS_PAPER_ADAPTED','DOG_PAPER_ADAPTED','CNF_ROI_STAGE_ADAPTED','J_PELT','J_PRIOR9'):
                pred=read_csv(path)
                # Evaluate on the SAME test cohort, never mix 649 test identities
                # with the full 2779-identity fixed-ROI scores.
                keys=pd.MultiIndex.from_frame(labels[['recording_id','track_id']])
                pool[name]=pred[pd.MultiIndex.from_frame(pred[['recording_id','track_id']]).isin(keys)]
        scores=[];matched={}
        for name,pred in pool.items():
            score,rows=evaluate_fixed(pred,labels);scores.append({'method_id':name,**score});matched[name]=rows
        common,_=common_tp(matched)
        views.append({'key':'fpn_same_test_cohort','family':'judges','table':pd.DataFrame(scores),
                      'common':common,'baseline':None,'missing':[]})
    return views,runtime,sources,excluded,protocol


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',type=Path,default=Path(__file__).parent/'revision_outputs/review_v2_full')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sections',help='Comma-separated view names; default all available')
    parser.add_argument('--ids',help='Comma-separated exact suite task IDs to include; other tasks are not read')
    parser.add_argument('--formats',default='png,pdf');parser.add_argument('--dpi',type=int,default=400)
    args=parser.parse_args(argv)
    if args.output.exists():parser.error('Use a new output directory to preserve previous figures')
    if not set(args.formats.split(','))<={'png','pdf','svg','tif','tiff'}:parser.error('Unsupported formats')
    if args.dpi<72:parser.error('dpi must be >=72')
    allowed_ids=set(args.ids.split(',')) if args.ids else None
    views,runtime,sources,excluded,protocol=build_views(args.suite,allowed_ids=allowed_ids)
    selected=set(args.sections.split(',')) if args.sections else None
    available={v['key'] for v in views}|({'runtime'} if runtime else set())
    if selected and selected-available:parser.error('No completed results for: '+','.join(sorted(selected-available)))
    if not available:parser.error('No completed eligible results')
    args.output.mkdir(parents=True)
    results=[];coverage=[]
    for view in views:
        key=view['key']
        if selected and key not in selected:continue
        path=args.output/'_plot_inputs'/key;path.mkdir(parents=True)
        filename={'judges':'S06_judge_comparison.csv','ablations':'S08_decision_feature_ablation.csv',
                  'detection':'S03_detector_comparison.csv','assignment':'S04_assignment_and_mot.csv',
                  'restoration':'S05_restoration_events.csv'}[view['family']]
        write_csv(path/filename,view['table'])
        if 'common' in view:write_csv(path/'S07_common_tp.csv',view['common'])
        for filename,rows in view.get('extra_tables',{}).items():write_csv(path/filename,rows)
        # Preserve source provenance explicitly for these derived figure views.
        write_json(path/'resolved_config.json',{'evaluation_mode':protocol['event_label_policy']+'; '+key,
                    'suite_input_protocol':protocol,'missing_conditions':view['missing'],
                    'derived_figure_view':True,'original_source_hashes':sources})
        figure_groups=set(GROUPS)
        if ('evaluation_scope' in view['table'] and
            view['table'].evaluation_scope.astype(str).str.startswith('MOT only;').all()):
            figure_groups={'mot'}
        plot_args=SimpleNamespace(output=args.output/'figures',font_size=14,dpi=args.dpi,
            formats=args.formats,groups=figure_groups,baseline=view['baseline'],
            label='AVAILABLE RESULTS; missing: '+','.join(view['missing']) if view['missing'] else 'Review-v2 same-protocol comparison',clean_figures=False)
        meta={'family':view['family'],'run_id':key,'status':'partial' if view['missing'] else 'completed','synthetic':False}
        results.append(Plotter(plot_args,path,meta).run_all())
        coverage.append({'section':key,'available':list(view['table'].method_id),'missing':view['missing']})
    if not selected or 'runtime' in selected:
        for run,meta in runtime:
            plot_args=SimpleNamespace(output=args.output/'figures',font_size=14,dpi=args.dpi,
                formats=args.formats,groups={'runtime'},baseline=None,label='Review-v2 measured runtime',clean_figures=False)
            results.append(Plotter(plot_args,run,meta).run_all())
    write_json(args.output/'plot_manifest.json',{'suite':str(args.suite.resolve()),'protocol':protocol,
        'source_hashes':sources,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'renderer_sha256':hashlib.sha256(Path(__file__).with_name('plot_revision_results.py').read_bytes()).hexdigest(),
        'sections':coverage,'excluded':excluded,'figures':results,
        'policy':'One suite protocol; exact Common-TP recalculation; separate feature references; missing conditions explicit'})
    print(args.output.resolve())


if __name__=='__main__':main()
