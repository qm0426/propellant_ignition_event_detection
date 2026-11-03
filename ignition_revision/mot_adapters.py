"""Explicit input-score policies and isolated imports for author MOT trackers."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import types
import numpy as np
import pandas as pd
from .common import Blocked, digest, file_hash, resolve


def author_variant(backend):
    from .common import PROJECT
    if backend=='ocsort':
        root=PROJECT/'revision_external/ocsort/trackers/ocsort_tracker'
        manifest=PROJECT/'revision_external/ocsort/source_manifest.json'
        return {'id':'MOT_OCSORT','backend':backend,'source_root':str(root),
            'source_file':str(root/'ocsort.py'),'source_manifest':str(manifest),
            'source_sha256':'122cc1c27d3321eac0debdf3ba782b76d189e1ac878256b8291ffeb4314fa979',
            'source_revision':'8462e7e729a93ccd3bd995c0a79a890336cb3a0b',
            'score_policy':{'kind':'fixed_pool_constant','constant':1.0},
            'settings':{'det_thresh':0.0,'max_age':5,'min_hits':3,'iou_threshold':.3,
                        'delta_t':3,'asso_func':'iou','inertia':.2,'use_byte':False},
            'comparison_scope':'author OC-SORT on shared cleaned candidates; constant inclusion score; Byte recovery OFF'}
    if backend=='bytetrack':
        root=PROJECT/'revision_external/bytetrack_compat'
        manifest=root/'source_manifest.json'
        # Hash is frozen by the compatibility manifest; never infer native
        # confidence from the old quality column's name.
        expected=json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else {}
        return {'id':'MOT_BYTE','backend':backend,'source_root':str(root),
            'source_file':str(root/'byte_tracker.py'),'source_manifest':str(manifest),
            'source_sha256':expected.get('files',{}).get('byte_tracker.py','missing'),
            'source_revision':'d1bf0191adff59bc8fcfeaa0b33d3d1642552a99 + documented import/NumPy/IoU compatibility',
            'frame_rate':30,'frame_rate_meaning':'author buffer normalization only; effective lost buffer 5 source frames; acquisition 50000 Hz',
            'score_policy':{'kind':'legacy_quality_proxy','transform':'divide_clip','scale':1.3,
                'provenance':'postprocess_detection_txt_modified.compute_final_quality_score; not detector confidence'},
            'settings':{'track_thresh':.5,'track_buffer':5,'match_thresh':.8,'mot20':False},
            'comparison_scope':'ByteTrack QUALITY-SCORE ADAPTED on shared cleaned candidates; no claim of native detector confidence'}
    raise ValueError(backend)


def prepare_scores(candidates, variant):
    """Return tracker inputs and an audit without modifying the candidate pool."""
    kind=variant.get('score_policy',{}).get('kind','detector_confidence')
    method=variant['backend'];policy=variant.get('score_policy',{});settings=variant.get('settings',{})
    raw=pd.to_numeric(candidates.native_score,errors='coerce').to_numpy(float)
    if kind=='fixed_pool_constant':
        if method!='ocsort' or settings.get('use_byte') is not False:
            raise ValueError('Fixed-pool policy requires OC-SORT with use_byte=False')
        constant=float(policy.get('constant',1.))
        if not np.isfinite(constant) or not constant>float(settings['det_thresh']):
            raise ValueError('Constant score must pass det_thresh for EVERY shared candidate')
        score=np.full(len(candidates),constant)
    elif kind=='legacy_quality_proxy':
        if method!='bytetrack' or not candidates.score_semantics.eq('legacy_quality_score').all():
            raise Blocked('Quality adaptation requires ByteTrack and explicitly sourced legacy_quality_score')
        if not np.isfinite(raw).all() or (raw<0).any():raise Blocked('Missing/invalid original quality scores')
        scale=float(policy.get('scale',0))
        if policy.get('transform')!='divide_clip' or not np.isfinite(scale) or scale<=0:
            raise ValueError('Quality proxy requires explicit positive scale and divide_clip')
        score=np.clip(raw/scale,0.,1.)
    elif method=='sort':
        score=np.where(np.isfinite(raw),raw,1.)
    elif kind=='detector_confidence':
        if not candidates.score_semantics.isin(['calibrated_confidence','detector_confidence']).all() or not np.isfinite(raw).all():
            raise Blocked(f'{method} needs detector confidence, or an EXPLICIT fixed-pool/quality adaptation')
        if ((raw<0)|(raw>1)).any():raise ValueError('Native confidence must be between 0 and 1')
        score=raw.copy()
    else:raise ValueError('Unknown score policy: '+kind)
    audit=candidates[['recording_id','frame_idx','candidate_id','native_score','score_semantics']].copy()
    audit=audit.rename(columns={'native_score':'input_score','score_semantics':'input_score_semantics'})
    audit['tracker_score']=score;audit['score_policy']=kind
    if method=='bytetrack':
        threshold=float(settings['track_thresh'])
        if not .1<threshold<.9:raise ValueError('ByteTrack high threshold must be in (0.1,0.9)')
        audit['score_group']=np.select([score>threshold,(score>.1)&(score<threshold)],['high','low'],default='ignored')
        audit['eligible_new_track']=score>=threshold+.1
    else:audit['score_group']='shared_pool'
    return score,audit


def validate_author_source(cfg,variant):
    source=resolve(cfg,variant['source_file'])
    if not source.is_file() or file_hash(source)!=variant['source_sha256']:
        raise Blocked('Author/compatibility source missing or hash mismatch: '+str(source))
    manifest=resolve(cfg,variant['source_manifest'])
    if not manifest.is_file():raise Blocked('Source manifest missing: '+str(manifest))
    meta=json.loads(manifest.read_text(encoding='utf-8'))
    root=resolve(cfg,variant['source_root'])
    for rel,expected in meta['files'].items():
        path=root/rel
        if not path.is_file() or file_hash(path)!=expected:raise Blocked('Source dependency hash mismatch: '+str(path))
    return source,root,meta


def load_author_module(cfg,variant):
    source,root,meta=validate_author_source(cfg,variant)
    name='_revision_author_'+variant['backend']+'_'+digest(meta)[:12]
    package=types.ModuleType(name);package.__path__=[str(root)];sys.modules[name]=package
    # Discard only this isolated author namespace so joint and individual jobs
    # start from fresh class counters and no globally imported yolox package.
    for key in list(sys.modules):
        if key.startswith(name+'.'):del sys.modules[key]
    full=name+'.'+source.stem
    spec=importlib.util.spec_from_file_location(full,source)
    module=importlib.util.module_from_spec(spec);sys.modules[full]=module
    try:spec.loader.exec_module(module)
    except ImportError as e:raise Blocked('Author tracker dependency unavailable: '+str(e)) from e
    return module
