"""Run independent or combined core experiment configurations without changing their parameters."""
import argparse,json,sys
from pathlib import Path
from ignition_revision.common import load_config,write_json
from ignition_revision.cli import execute_config
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',required=True)
    p.add_argument('--only',nargs='+');p.add_argument('--disable',nargs='*',default=[])
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();root=Path(__file__).resolve().parent;suite=json.loads(Path(a.suite).read_text(encoding='utf-8-sig'))
    ids=[r['id'] for r in suite['runs']]
    if len(ids)!=len(set(ids)):p.error('Duplicate run ids')
    unknown=set((a.only or [])+a.disable)-set(ids)
    if unknown:p.error('Unknown run ids: '+','.join(sorted(unknown)))
    records=[]
    for item in suite['runs']:
        if not item.get('enabled',True) or item['id'] in a.disable or (a.only and item['id'] not in a.only):continue
        cfg=load_config(root/item['config']);cfg['output_dir']=str(root/suite.get('output_dir','outputs/suite')/item['id'])
        code=execute_config(item['family'],cfg,a.dry_run,a.resume)
        records.append({'id':item['id'],'family':item['family'],'exit_code':code,'output_dir':cfg['output_dir']})
        write_json(root/suite.get('output_dir','outputs/suite')/'suite_status.json',records)
    if not records:p.error('No enabled selected runs')
    return 1 if any(r['exit_code'] for r in records) else 0
if __name__=='__main__':raise SystemExit(main())
