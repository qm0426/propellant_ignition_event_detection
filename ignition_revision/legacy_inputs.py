"""Adapt the final legacy data interfaces. Historical references are not independent GT."""
from __future__ import annotations
import copy
import os
import re
from pathlib import Path
import pandas as pd
from .common import Blocked, digest, file_hash, load_legacy, read_table, resolve, write_csv, write_json

EXTS={'.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp'}
KEYS=('roi_root','annotation_csv','original_reference_tracks_csv','detections_txt_dir','prebirth_image_dir','max_txt','recording_id')

def add_legacy_arguments(p):
    p.add_argument('--roi-root','--tracks-root',dest='roi_root')
    p.add_argument('--annotation-csv','--event-annotation-csv',dest='annotation_csv')
    p.add_argument('--original-reference-tracks-csv')
    p.add_argument('--detections-txt-dir')
    p.add_argument('--prebirth-image-dir','--image-dir',dest='prebirth_image_dir')
    p.add_argument('--max-txt',type=int)
    p.add_argument('--recording-id')
    p.add_argument('--output-dir')
    p.add_argument('--legacy-reproduction',action='store_true')

def merge_legacy_arguments(cfg,args):
    cfg=copy.deepcopy(cfg)
    supplied={k:getattr(args,k,None) for k in KEYS if getattr(args,k,None) is not None}
    if supplied:cfg.setdefault('legacy_inputs',{}).update(supplied)
    for k in KEYS:
        if k in cfg:cfg.setdefault('legacy_inputs',{})[k]=cfg.pop(k)
    for old,new in [('tracks_root','roi_root'),('event_annotation_csv','annotation_csv'),('image_dir','prebirth_image_dir')]:
        if old in cfg:cfg.setdefault('legacy_inputs',{})[new]=cfg.pop(old)
    if getattr(args,'output_dir',None):cfg['output_dir']=args.output_dir
    if getattr(args,'legacy_reproduction',False):cfg['evaluation_mode']='legacy_reproduction'
    return cfg

def track_id(value):
    s=str(value).strip()
    if re.fullmatch(r'[0-9]+(?:\.0+)?',s):return int(float(s))
    m=re.fullmatch(r'track_([0-9]+)',s)
    if m:return int(m[1])
    raise ValueError(f'Invalid legacy track ID: {value!r}')

def effective_labels(annotation,sequences,rec):
    ann=read_table(annotation);ann.columns=[str(c).strip().lower() for c in ann.columns]
    if not {'track_id','frame_id'}<=set(ann):raise ValueError('Legacy adapter requires effective_trace track_id/frame_id; use canonical labels for other annotation schemas')
    values={}
    for _,r in ann.iterrows():
        if pd.isna(r.track_id):continue
        tid=track_id(r.track_id)
        if tid in values:raise ValueError(f'Duplicate annotation track {tid}')
        if pd.isna(r.frame_id) or str(r.frame_id).strip()=='':values[tid]=None
        else:
            n=float(r.frame_id)
            if not n.is_integer() or n<0:raise ValueError(f'Invalid onset index: track {tid}')
            values[tid]=int(n)
    rows=[]
    for tid,frames in sorted(sequences.items()):
        index=values.get(tid)
        if index is not None and index>=len(frames):raise ValueError(f'Onset index out of range: track {tid}, {index}/{len(frames)}')
        rows.append({'recording_id':rec,'track_id':tid,'ref_object_id':f'legacy_track_{tid}',
                     'ref_event_id':f'{rec}:legacy_track_{tid}:first','has_ignition':index is not None,
                     'onset_frame':frames[index] if index is not None else None})
    return pd.DataFrame(rows),[t for t,i in values.items() if i is not None and t not in sequences]

def adapt_legacy_inputs(cfg,family):
    if not cfg.get('legacy_inputs'):return cfg
    cfg=copy.deepcopy(cfg);spec=cfg['legacy_inputs'];rec=str(spec.get('recording_id','legacy_recording'))
    if family not in ('judges','ablations','assignment','restoration'):
        raise Blocked(f'{family} needs canonical inputs: cleaned TXT cannot reconstruct raw prompt groups or detector GT')
    if cfg.get('evaluation_mode')!='legacy_reproduction':raise Blocked('Set --legacy-reproduction for historical labels/references')
    def path(key,required=True):
        if not spec.get(key):
            if required:raise Blocked(f'Missing legacy input: {key}')
            return None
        p=resolve(cfg,spec[key])
        if not p.exists():raise Blocked(f'Missing {key}: {p}')
        return p
    ann=path('annotation_csv',False);sources=[ann] if ann else [];tables={}
    audit={'evaluation_scope':'legacy_reproduction','independent_reference':False,
           'negative_policy':'blank/unlisted tracks are negative (legacy convention)',
           'onset_index':'zero-based per-track sorted position mapped to global frame',
           'prediction_protocol':'revision algorithms; input compatibility does not imply identical legacy results'}
    if family in ('judges','ablations'):
        root=path('roi_root');rows=[];seq={}
        for folder in sorted(root.glob('track_*')):
            if not folder.is_dir():continue
            tid=track_id(folder.name)
            with os.scandir(folder) as entries:
                images=sorted((Path(e.path) for e in entries if e.is_file() and Path(e.name).suffix.lower() in EXTS), key=lambda p:[int(x) if x.isdigit() else x for x in re.split(r'([0-9]+)',p.name)])
            if not images:continue
            frames=[];sources.extend(images)
            for image in images:
                m=re.match(r'^frame_([0-9]+)_',image.name)
                if not m:raise ValueError(f'Cannot infer global frame from {image}; supply canonical rois.csv')
                frame=int(m[1]);frames.append(frame)
                rows.append({'recording_id':rec,'track_id':tid,'frame_idx':frame,'roi_path':str(image)})
            if frames!=sorted(set(frames)):raise ValueError(f'Duplicate/nonmonotonic ROI frames: {folder}')
            seq[tid]=frames
        if not rows:raise Blocked(f'No track ROI images: {root}')
        tables['rois']=pd.DataFrame(rows)
        if ann:
            tables['labels'],excluded=effective_labels(ann,seq,rec)
            audit['positive_annotation_ids_outside_fixed_ROI_cohort']=excluded
        cfg['track_bundle_id']='legacy_roi_pending_content_hash'
        cfg['label_provenance']='legacy effective_trace; algorithm-conditioned fixed ROI cohort; implicit negatives'
    else:
        txtdir=path('detections_txt_dir');image_dir=path('prebirth_image_dir')
        mod=load_legacy('ot_kalman_tracker.py',cfg.get('legacy_root'))
        txts=sorted(txtdir.glob('*.txt'));limit=int(spec.get('max_txt',-1))
        if limit>=0:txts=txts[:limit]
        if not txts:raise Blocked('No selected detection TXT files')
        images=mod.collect_image_paths(image_dir)
        if len(images)<len(txts):raise ValueError('Fewer images than selected TXT files')
        images=images[:len(txts)]
        for i,(t,im) in enumerate(zip(txts,images)):
            if t.stem!=im.stem:raise ValueError(f'TXT/image sorted alignment mismatch at {i}: {t.name} vs {im.name}; use canonical manifests for missing files')
        sources.extend(txts);sources.extend(images)
        detections=mod.load_detections_txt_dir(txtdir,max_txt=limit)
        tables['frames']=pd.DataFrame([{'recording_id':rec,'frame_idx':i,'image_path':str(im.resolve()),'annotated':False} for i,im in enumerate(images)])
        from .detection import BOX_COLUMNS
        tables['candidates']=pd.DataFrame([{'recording_id':rec,'frame_idx':i,'candidate_id':f'{rec}:{i}:{j}',
            **{k:d[k] for k in ('x1','y1','x2','y2')},'prompt_group':'unknown_legacy_merged','native_score':d['score'],
            'score_semantics':'legacy_quality_score'} for i,ds in detections.items() for j,d in enumerate(ds)],columns=BOX_COLUMNS)
        refpath=path('original_reference_tracks_csv',False)
        if ann and not refpath:raise Blocked('effective_trace tracking evaluation requires original_reference_tracks_csv')
        if refpath:
            sources.append(refpath);ref=read_table(refpath)
            need={'track_id','frame_id','x1','y1','x2','y2'}
            if not need<=set(ref):raise ValueError(f'Reference tracks missing {need-set(ref)}')
            if ref.duplicated(['track_id','frame_id']).any():raise ValueError('Duplicate reference track/frame')
            seq={int(t):g.sort_values('frame_id').frame_id.astype(int).tolist() for t,g in ref.groupby('track_id')}
            if ann:
                labels,excluded=effective_labels(ann,seq,rec)
                if excluded:raise ValueError(f'Positive IDs absent from original reference: {excluded[:20]}')
                # Keep all reference events under max-txt; do not hide out-of-scope FNs.
                tables['reference_events']=labels[labels.has_ignition].drop(columns=['track_id','has_ignition'])
            boxes=ref.rename(columns={'frame_id':'frame_idx'}).copy();boxes['recording_id']=rec
            boxes['ref_object_id']=boxes.track_id.map(lambda t:f'legacy_track_{int(t)}')
            boxes['cx']=(boxes.x1+boxes.x2)/2;boxes['cy']=(boxes.y1+boxes.y2)/2
            tables['reference_boxes']=boxes[['recording_id','frame_idx','ref_object_id','x1','y1','x2','y2','cx','cy']]
        cfg.setdefault('evaluation',{})
        for k,v in [('mapping_max_distance',30.),('mapping_min_shared_frames',3),('mapping_min_purity',.8)]:
            if cfg['evaluation'].get(k) is None:cfg['evaluation'][k]=v
        audit['tracking_reference']='old algorithm tracks, NOT independent manual MOT GT'
        audit['mapping_protocol']='revision distance/purity mapping, not identical to legacy coverage/cost mapping'
    # Files already have absolute paths; avoid repeated Windows stat/resolve calls per ROI.
    from concurrent.futures import ThreadPoolExecutor
    print(f'LEGACY_INPUT_CHECK: {len(sources)} source files; hashing contents...',flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes=dict(zip((str(p) for p in sources),pool.map(file_hash,sources)))
    if family in ('judges','ablations'):
        cfg['track_bundle_id']='legacy_roi_'+digest({k:v for k,v in hashes.items() if Path(k).suffix.lower() in EXTS})[:16]
    out=resolve(cfg,cfg.get('output_dir','revision_outputs'))/'_legacy_inputs'/digest([spec,hashes,audit])[:20]
    for name,table in tables.items():write_csv(out/(name+'.csv'),table);cfg[name]=str(out/(name+'.csv'))
    for name in ('labels','reference_boxes','reference_events'):
        if name not in tables:cfg.pop(name,None)
    cfg['legacy_input_provenance']=audit;cfg['legacy_source_hashes']=hashes
    write_json(out/'legacy_input_audit.json',{'source_hashes':hashes,**audit,'tables':{k:len(v) for k,v in tables.items()}})
    cfg['legacy_input_audit']=str(out/'legacy_input_audit.json')
    return cfg
