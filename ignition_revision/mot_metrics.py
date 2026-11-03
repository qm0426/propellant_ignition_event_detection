"""Run the actual official TrackEval metric implementations on explicit annotated scopes."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from .common import Blocked, boolean, file_hash, require_columns, resolve, write_csv, write_json
from .detection import iou_matrix


def evaluate_standard_mot(observations,reference,frames,cfg,out):
    settings=cfg.get('standard_mot_evaluation')
    if not settings:raise Blocked('Configure official TrackEval source and fully annotated MOT scope to compute HOTA/IDF1/IDSW')
    root=resolve(cfg,settings.get('source_root'))
    if not (root/'trackeval').is_dir():raise Blocked('Official TrackEval source_root not available')
    if not settings.get('source_revision'):raise Blocked('TrackEval immutable source_revision required')
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    try:import trackeval
    except ImportError as e:raise Blocked(f'TrackEval dependency: {e}') from e
    if root not in Path(trackeval.__file__).resolve().parents:raise ValueError('Imported TrackEval differs from configured source_root')
    require_columns(frames,['annotated'],'MOT frame scope')
    require_columns(reference,['recording_id','frame_idx','ref_object_id','x1','y1','x2','y2'],'MOT reference')
    scope=frames[frames.annotated.map(boolean)]
    if scope.empty:raise Blocked('No explicitly fully annotated MOT frames')
    calculators={'HOTA':trackeval.metrics.HOTA(),'Identity':trackeval.metrics.Identity({'THRESHOLD':settings['iou_threshold'],'PRINT_CONFIG':False}),
                 'CLEAR':trackeval.metrics.CLEAR({'THRESHOLD':settings['iou_threshold'],'PRINT_CONFIG':False})}
    results={name:{} for name in calculators}
    for rec,ff in scope.groupby('recording_id'):
        valid=set(ff.frame_idx.astype(int))
        gt=reference[(reference.recording_id==rec)&reference.frame_idx.isin(valid)]
        pred=observations[(observations.recording_id==rec)&observations.frame_idx.isin(valid)]
        if 'ignore' in gt and gt['ignore'].map(boolean).any():
            raise Blocked('TrackEval wrapper currently requires fully annotated non-ignore frames; use explicit non-ignore scope or official dataset preprocessing')
        if gt.duplicated(['frame_idx','ref_object_id']).any() or pred.duplicated(['frame_idx','track_id']).any():raise ValueError('Duplicate per-frame identities')
        gids={v:i for i,v in enumerate(sorted(gt.ref_object_id.unique(),key=str))};pids={v:i for i,v in enumerate(sorted(pred.track_id.unique(),key=str))}
        data={'num_gt_ids':len(gids),'num_tracker_ids':len(pids),'num_gt_dets':len(gt),'num_tracker_dets':len(pred),
              'num_timesteps':len(ff),'gt_ids':[],'tracker_ids':[],'similarity_scores':[]}
        for fr in sorted(valid):
            g=gt[gt.frame_idx==fr];p=pred[pred.frame_idx==fr]
            data['gt_ids'].append(np.asarray([gids[v] for v in g.ref_object_id],dtype=int))
            data['tracker_ids'].append(np.asarray([pids[v] for v in p.track_id],dtype=int))
            data['similarity_scores'].append(iou_matrix(g,p))
        for name,calc in calculators.items():
            try:results[name][str(rec)]=calc.eval_sequence(data)
            except (AttributeError,TypeError) as e:raise Blocked(f'Installed TrackEval/numpy API incompatibility: {e}; no approximate replacement used') from e
    combined={name:calc.combine_sequences(results[name]) for name,calc in calculators.items()}
    write_json(out/'official_mot_metrics.json',{'per_recording':results,'combined':combined,
        'source_revision':settings['source_revision'],'source_hashes':{str(p.relative_to(root)):file_hash(p) for p in (root/'trackeval'/'metrics').glob('*.py')},
        'scope':scope[['recording_id','frame_idx']].to_dict('records'),'units':'HOTA/IDF1 fractions; IDSW integer'})
    return {'HOTA':float(np.mean(combined['HOTA']['HOTA'])),'IDF1':float(combined['Identity']['IDF1']),
            'IDSW':int(combined['CLEAR']['IDSW']),'fragmentations':int(combined['CLEAR']['Frag'])}
