"""Run only the reference-9 criterion with K-means masks and compare saved baselines."""
import argparse
import itertools
import json
from pathlib import Path
import pandas as pd
from ignition_revision.common import PROJECT,write_json,write_csv,file_hash

DEFAULT_ROOT=PROJECT/'revision_outputs/review_v2_prior9_kmeans_20261031'


def compare(root,baseline):
    from ignition_revision.evaluation import common_tp
    status=json.loads((root/'suite_status.json').read_text(encoding='utf-8'))
    record=next(r for r in status if r['id']=='J_PRIOR9')
    if record['status']!='completed':raise ValueError('J_PRIOR9 is incomplete')
    run=Path(record['run_dir']);new=pd.read_csv(run/'judge_comparison_fixed_tracks.csv')
    old=pd.read_csv(baseline/'combined/all_fixed_cohort_metrics.csv')
    old=old[old.suite_group.isin(['judges','rtv_pairs'])|old.method_id.eq('DEC_FULL')]
    if set(old.cohort_hash)!={new.cohort_hash.iloc[0]}:raise ValueError('Fixed reference cohort differs')
    outputs=root/'comparison_with_saved_results';outputs.mkdir(exist_ok=True)
    write_csv(outputs/'fixed_cohort_comparison.csv',pd.concat([old,new],ignore_index=True))
    history=json.loads((baseline/'suite_status.json').read_text(encoding='utf-8'));lookup={r['id']:r for r in history}
    reference=pd.read_csv(run/'J_PRIOR9/evaluation_members.csv').to_dict('records');pairs=[];member_sets={}
    sources={str(run/'J_PRIOR9/evaluation_members.csv'):file_hash(run/'J_PRIOR9/evaluation_members.csv')}
    for name in old.method_id:
        path=Path(lookup[name]['run_dir'])/name/'evaluation_members.csv'
        table=pd.read_csv(path);rows=table.to_dict('records')
        a={(r['recording_id'],str(r['ref_event_id'])) for r in reference};b={(r['recording_id'],str(r['ref_event_id'])) for r in rows}
        if a!=b:raise ValueError('Member-level reference mismatch: '+name)
        scores,members=common_tp({'J_PRIOR9':reference,name:rows})
        pairs.extend({**s,'comparison':'J_PRIOR9__'+name} for s in scores);member_sets[name]=members;sources[str(path)]=file_hash(path)
    write_csv(outputs/'prior9_pairwise_common_tp.csv',pairs);write_json(outputs/'common_tp_members.json',member_sets)
    write_json(outputs/'comparison_provenance.json',{'new_run_dir':str(run),'saved_baseline_suite':str(baseline),
        'existing_results_rerun':False,'mask_substitution':'Kmeans2, not SAM','source_hashes':sources})
    print(outputs)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['run','check','compare','all'],default='all',nargs='?')
    p.add_argument('--output',type=Path,default=DEFAULT_ROOT);p.add_argument('--baseline-suite',type=Path,default=PROJECT/'revision_outputs/review_v2_full')
    args=p.parse_args();root=args.output.resolve()
    if args.command in ['run','check','all']:
        from ignition_revision.review_v2 import main as review
        settings=root.parent/(root.name+'_settings.json')
        write_json(settings,{'output_root':str(root)})
        result=review(['check' if args.command=='check' else 'run','--ids','J_PRIOR9','--settings',str(settings),'--resume','--fail-fast'])
        if result:raise SystemExit(result)
    if args.command in ['compare','all']:compare(root,args.baseline_suite.resolve())


if __name__=='__main__':main()
