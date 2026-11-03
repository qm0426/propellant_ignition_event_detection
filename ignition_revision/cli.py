from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import traceback
from .common import Blocked, Run, load_config, jsonable, resolve, write_csv, write_json

FAMILIES=('detection','assignment','restoration','judges','ablations','train-state','train-fpn','infer-fpn','fpn-events','fpn-pipeline','benchmark')


def run_family(family,cfg,run):
    if family=='detection':
        from .detection import run_detection
        return run_detection(cfg,run)
    if family in ('assignment','restoration'):
        from .experiments import run_tracking
        return run_tracking(cfg,run,restoration=family=='restoration')
    if family in ('judges','ablations'):
        from .experiments import run_onsets
        return run_onsets(cfg,run,ablation=family=='ablations')
    if family=='train-state':
        from .supervised import train_state
        return train_state(cfg,run)
    if family in ('train-fpn','infer-fpn'):
        from .fpn_detector import train_fpn,infer_fpn
        return (train_fpn if family=='train-fpn' else infer_fpn)(cfg,run)
    if family in ('fpn-events','fpn-pipeline'):
        from .fpn_events import run_fpn_events,run_fpn_pipeline
        return (run_fpn_events if family=='fpn-events' else run_fpn_pipeline)(cfg,run)
    if family=='benchmark':return benchmark(cfg,run)
    raise ValueError(family)


def benchmark(cfg,run):
    family=cfg.get('benchmark_family')
    if family not in FAMILIES or family=='benchmark':raise ValueError('Configure valid benchmark_family')
    base=copy.deepcopy(cfg.get('experiment',{}));base.setdefault('project_root',cfg['project_root'])
    base.setdefault('output_dir',str(run.out/'repetitions'))
    rows=[]
    warmup=int(cfg.get('warmup_repeats',1));repeats=int(cfg.get('measurement_repeats',3))
    if warmup<0 or repeats<1:raise ValueError('Invalid repetition counts')
    for index in range(warmup+repeats):
        local=copy.deepcopy(base);local['benchmark_repeat']=index
        child=Run(local,family)
        try:run_family(family,local,child);child.finish()
        except Exception as e:child.finish(e);raise
        if child.meta['status']!='completed':raise Blocked(f'Benchmark includes incomplete experiment: {child.out}')
        if index>=warmup:
            rows.append({'repeat':index-warmup,'elapsed_s':child.meta['elapsed_s'],'run_id':child.run_id,
                         'cache_mode':'cold_new_experiment_directory','model_process_cache':'shared_process; see note'})
    write_csv(run.out/'runtime_repetitions.csv',rows)
    import numpy as np
    values=[x['elapsed_s'] for x in rows]
    write_json(run.out/'runtime_statistics.json',{'N':len(values),'mean_s':float(np.mean(values)),
               'std_s':float(np.std(values)),'min_s':min(values),'max_s':max(values),
               'note':'Data and operating-system cache may be warm after warmup. Not an OS-cold end-to-end benchmark. Model cache behavior must be reported per stage.'})
    run.record('runtime','completed')


def planned_methods(family,cfg):
    if family=='assignment':
        from .tracking import assignment_variants
        return [v['id'] for v in cfg.get('variants',assignment_variants())]
    if family=='restoration':
        from .tracking import restoration_variants
        return [v['id'] for v in cfg.get('variants',restoration_variants())]
    if family=='ablations':
        from .experiments import decision_variants
        return list(decision_variants(cfg))
    if family in ('fpn-events','fpn-pipeline'):return list(cfg.get('methods',{'FPN_IGNITION_SHARED_TRACKS':{}}))
    if family=='judges':return list(cfg.get('methods',{}))
    if family=='detection':return cfg.get('selected_method_ids') or ['DET_'+g+'_'+s for g in ('COLD','HOT','BOTH') for s in ('raw','cleaned')]+['DET_'+m for m in cfg.get('classical',{})]
    return [family]


def execute_config(family,cfg,dry_run=False,resume=False,check_inputs=False):
    from .legacy_inputs import adapt_legacy_inputs
    if not dry_run or check_inputs:
        try:cfg=adapt_legacy_inputs(cfg,family)
        except Blocked as e:
            print(f'BLOCKED: {e}',file=sys.stderr);return 2
    if dry_run:
        print(json.dumps(jsonable({'family':family,'status':'inputs_converted' if check_inputs and cfg.get('legacy_input_audit') else 'planned',
            'methods':planned_methods(family,cfg),'config':cfg}),ensure_ascii=True,indent=2))
        return 0
    run=None
    try:
        run=Run(cfg,family,resume=resume)
        if run.skip:print(f'VERIFIED_RESUME {run.out}');return 0
        run_family(family,cfg,run);run.finish();print(f'{run.meta["status"].upper()} {run.out}')
        return 0 if run.meta['status']=='completed' else 2
    except Blocked as e:
        if run is not None:
            run.record(family,'blocked',str(e));run.finish()
        print(f'BLOCKED: {e}',file=sys.stderr);return 2
    except Exception as e:
        if run is not None and not run.skip:run.finish(e)
        traceback.print_exc();return 1


def main(default_family=None,argv=None):
    from .legacy_inputs import add_legacy_arguments,merge_legacy_arguments
    parser=argparse.ArgumentParser(description='Review experiments with canonical or legacy input adapters')
    if default_family is None:parser.add_argument('family',choices=FAMILIES)
    parser.add_argument('--config',required=True)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--check-inputs',action='store_true',help='With dry-run, validate/convert legacy inputs without running models')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--device');parser.add_argument('--seed',type=int)
    add_legacy_arguments(parser)
    args=parser.parse_args(argv);cfg=merge_legacy_arguments(load_config(args.config),args)
    family=default_family or args.family
    if args.device is not None:cfg['device']=args.device
    if args.seed is not None:cfg.setdefault('gmm_config',{})['random_state']=args.seed;cfg['seed']=args.seed
    return execute_config(family,cfg,args.dry_run,args.resume,args.check_inputs)
