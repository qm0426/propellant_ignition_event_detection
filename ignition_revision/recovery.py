"""Validated reuse of completed tracking stages after downstream cache failure."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .common import digest,file_hash,resolve,write_json

TRACKING_CONFIG_KEYS=('frames','candidates','tracking','variants','restoration','crop_padding','virtual_tracking_rois','time_advance_policy')


def reuse_tracking_stage(cfg,run,variant,out):
    spec=cfg.get('tracking_recovery')
    if not spec:return None
    try:
        source=Path(spec['source_run_dir']).resolve()
        meta_path=Path(spec['run_meta_file']).resolve()
        if meta_path!=source/'run_meta.json' or file_hash(meta_path)!=spec['run_meta_sha256']:
            raise ValueError('Recovery run metadata differs from selected snapshot')
        meta=json.loads(meta_path.read_text(encoding='utf-8'))
        if file_hash(source/'resolved_config.json')!=meta['outputs'].get('resolved_config.json'):
            raise ValueError('Original tracking configuration checksum mismatch')
        previous=json.loads((source/'resolved_config.json').read_text(encoding='utf-8'))
        if digest({k:previous.get(k) for k in TRACKING_CONFIG_KEYS})!=digest({k:cfg.get(k) for k in TRACKING_CONFIG_KEYS}):
            raise ValueError('Tracking parameters or crop settings changed')
        from . import tracking,detection,common
        for module in (tracking,detection,common):
            path=Path(module.__file__)
            if meta['code_hashes'].get(path.name)!=file_hash(path):raise ValueError('Tracking producer code changed: '+path.name)
        # Current Run re-hashes the actual inputs. Recovery-source audit files
        # are extra inputs; every original input must still match exactly.
        if any(run.inputs.get(path)!=value for path,value in meta['input_hashes'].items()):
            raise ValueError('Original tracking input contents changed')
        paths={}
        for key,filename in [('observations_file','track_observations.csv'),('roi_manifest_file','roi_manifest.csv')]:
            path=Path(spec[key]).resolve();expected=source/variant['id']/filename
            if path!=expected or file_hash(path)!=meta['outputs'].get(str(path.relative_to(source))):
                raise ValueError('Recovery output checksum mismatch: '+filename)
            paths[key]=path
        obs=pd.read_csv(paths['observations_file'],low_memory=False,float_precision='round_trip')
        rois=pd.read_csv(paths['roi_manifest_file'],low_memory=False,float_precision='round_trip')
        keys=['recording_id','track_id','frame_idx']
        if obs.duplicated(keys).any() or rois.duplicated(keys).any():raise ValueError('Duplicate recovered track observations')
        if len(obs)!=len(rois) or set(map(tuple,obs[keys].to_numpy()))!=set(map(tuple,rois[keys].to_numpy())):
            raise ValueError('Recovered ROI and observation cohorts differ')
        write_json(Path(out)/'tracking_recovery_status.json',{'status':'verified_reused',
            'source_run_dir':str(source),'source_run_meta_sha256':spec['run_meta_sha256'],
            'observations':len(obs),'rois':len(rois),'tracking_parameters_and_inputs':'identical',
            'downstream_features_and_judgment':'recomputed using corrected cache identity'})
        print(f'TRACKING_RECOVERY {variant["id"]}: verified {len(obs)} saved observations; association not repeated',flush=True)
        return obs,rois
    except (ValueError,KeyError,OSError) as error:
        write_json(Path(out)/'tracking_recovery_status.json',{'status':'not_reused_recompute_tracking','reason':str(error)})
        print(f'TRACKING_RECOVERY {variant["id"]}: cannot verify saved stage; recomputing ({error})',flush=True)
        return None
