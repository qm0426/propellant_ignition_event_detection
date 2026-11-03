"""Use existing canonical or legacy ROI input config; compare fixed histories."""
import argparse,copy
from pathlib import Path
from ignition_revision.common import load_config
from ignition_revision.cli import execute_config

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--methods',nargs='+',choices=['J_GMM','J_GMM_RTV','J_HMM','J_HMM_RTV'],default=['J_GMM','J_GMM_RTV','J_HMM','J_HMM_RTV'])
    a=p.parse_args();cfg=load_config(a.config);cfg['output_dir']=a.output
    methods={}
    for method in ['gmm','hmm']:
        for rtv in [False,True]:
            setting={'method':method,'rtv':rtv,'apply_legacy_screens':True,'candidate_policy':'first_terminal'}
            if method=='hmm':setting['hmm']=copy.deepcopy(cfg.get('hmm',{}))
            methods['J_'+method.upper()+('_RTV' if rtv else '')]=setting
    cfg['methods']={k:v for k,v in methods.items() if k in a.methods}
    status=execute_config('judges',cfg,a.dry_run,a.resume)
    if status==0 and not a.dry_run and cfg.get('labels'):
        import pandas as pd
        from ignition_revision.common import resolve,write_csv
        from ignition_revision.evaluation import evaluate_fixed
        runs=sorted((resolve(cfg,a.output)/'judges').glob('*/resolved_config.json'),key=lambda p:p.stat().st_mtime)
        root=runs[-1].parent;labels=pd.read_csv(resolve(cfg,cfg['labels']),low_memory=False);rows=[]
        for method in cfg['methods']:
            pred=pd.read_csv(root/method/'predictions.csv')
            for tol in [1,3,5,10]:
                score,_=evaluate_fixed(pred,labels,tolerance=tol)
                rows.append({'method_id':method,'tolerance_frames':tol,**score})
        write_csv(root/'joint_tolerance_metrics.csv',rows)
        print('HMM_COMPARISON_OUTPUT',root)
    return status

if __name__=='__main__':raise SystemExit(main())
