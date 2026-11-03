"""Labelme preannotation for independent frame-level detector review.

Image sampling never conditions on detection success or positive-event GT. Seeds
are editable proposals, not reference GT. Review must add missed particles.
"""
from __future__ import annotations
import argparse,json,re,math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from .common import boolean,file_hash,read_table,write_csv,write_json,require_columns


def read_pixel_boxes(path):
    boxes=[];invalid=[]
    if not path.is_file():return boxes,invalid
    for lineno,line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(),1):
        fields=line.split()
        if not fields or fields[0].lower()=='label':continue
        if len(fields) not in (5,6):raise ValueError(f'Unsupported pixel TXT at {path}:{lineno}')
        try:x1,y1,x2,y2=map(float,fields[1:5])
        except ValueError as e:raise ValueError(f'Invalid coordinates at {path}:{lineno}') from e
        row={'source_label':fields[0],'x1':x1,'y1':y1,'x2':x2,'y2':y2,'line':lineno}
        if not np.isfinite([x1,y1,x2,y2]).all() or x2<=x1 or y2<=y1:invalid.append(row);continue
        boxes.append(row)
    return boxes,invalid


def export_tasks(cfg):
    out=Path(cfg['output_dir']).resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'Nonempty review directory: {out}')
    root=Path(cfg['image_root']);pattern=re.compile(cfg.get('frame_regex',r'_(?P<frame>\d+)$'))
    if 'frame' not in pattern.groupindex:raise ValueError('frame_regex requires named frame group')
    candidates=[];start=int(cfg.get('start_frame',0));end=cfg.get('end_frame');offset=int(cfg.get('frame_offset',0))
    for path in root.iterdir():
        if path.suffix.lower() not in ('.png','.jpg','.jpeg','.tif','.tiff','.bmp'):continue
        match=pattern.search(path.stem)
        if not match:continue
        f=int(match.group('frame'))+offset
        if f>=start and (end is None or f<=int(end)):candidates.append((f,path))
    candidates.sort(key=lambda v:v[0])
    if len({f for f,p in candidates})!=len(candidates):raise ValueError('Ambiguous images for same physical frame')
    n=int(cfg.get('count',40));rng=np.random.default_rng(int(cfg.get('seed',20261025)))
    if n<1 or len(candidates)<n:raise ValueError('count must be positive and no larger than eligible images')
    sampling=cfg.get('sampling','temporal_stratified')
    if sampling=='temporal_stratified':
        chunks=np.array_split(np.arange(len(candidates)),n)
        chosen=[(int(rng.choice(chunk)),i,len(chunk)) for i,chunk in enumerate(chunks)]
    elif sampling=='uniform_random':
        chosen=[(int(index),0,len(candidates)) for index in sorted(rng.choice(len(candidates),size=n,replace=False).tolist())]
    else:raise ValueError('sampling must be temporal_stratified or uniform_random')
    blank=float(cfg.get('blank_fraction',.25))
    if not 0<=blank<=1:raise ValueError('blank_fraction must be in [0,1]')
    blind=set(rng.choice(n,size=round(n*blank),replace=False).tolist())
    tasks=out/'tasks';tasks.mkdir(parents=True);manifest=[];seed_rows=[];invalid_rows=[]
    for k,(index,stratum,population) in enumerate(chosen):
        f,path=candidates[index];seed_path=Path(cfg['seed_txt_dir'])/(path.stem+'.txt')
        boxes,invalid=read_pixel_boxes(seed_path)
        with Image.open(path) as im:
            if im.mode not in ('L','RGB','RGBA'):raise ValueError('High bit-depth conversion requires an explicit documented mapping')
            w,h=im.size;image_path=tasks/(path.stem+'.png');im.convert('RGB').save(image_path)
        shapes=[]
        for j,b in enumerate(boxes):
            seed_rows.append({'recording_id':cfg['recording_id'],'frame_idx':f,'candidate_id':f'seed:{f}:{j}',**b})
            if k in blind:continue
            x1=max(0.,b['x1']);y1=max(0.,b['y1']);x2=min(float(w),b['x2']);y2=min(float(h),b['y2'])
            if x2<=x1 or y2<=y1:invalid.append({**b,'reason':'outside_image'});continue
            shapes.append({'label':'particle_unknown','points':[[x1,y1],[x2,y2]],'group_id':None,'description':'editable detector proposal; inspect the whole image and add missed targets','shape_type':'rectangle','flags':{}})
        json_path=image_path.with_suffix('.json')
        write_json(json_path,{'version':'5.5.0','flags':{'reviewed':False,'complete':False},'shapes':shapes,'imagePath':image_path.name,'imageData':None,'imageHeight':h,'imageWidth':w})
        invalid_rows.extend({'recording_id':cfg['recording_id'],'frame_idx':f,**b} for b in invalid)
        manifest.append({'recording_id':cfg['recording_id'],'frame_idx':f,'json_path':str(json_path),'task_image':str(image_path),'original_image':str(path.resolve()),'width':w,'height':h,'sampling_stratum':stratum,'eligible_in_stratum':population,'inclusion_probability':n/population if sampling=='uniform_random' else 1/population,'preannotated':k not in blind,'seed_boxes':len(shapes),'seed_txt':str(seed_path.resolve()),'seed_missing':not seed_path.exists(),'source_image_sha256':file_hash(path),'task_image_sha256':file_hash(image_path),'seed_sha256':file_hash(seed_path) if seed_path.is_file() else None})
    write_csv(out/'review_manifest.csv',manifest);write_csv(out/'seed_candidates_snapshot.csv',seed_rows);write_json(out/'invalid_seed_boxes.json',invalid_rows)
    write_json(out/'export_meta.json',{'config':cfg,'eligible_images':len(candidates),'selected_images':n,'blank_images':len(blind),'sampling':sampling,'sampling_note':'Uniform without replacement' if sampling=='uniform_random' else 'One random image per equal-count temporal stratum; independent of detector output','gt_status':'unreviewed; no manual reference labels exported','label_semantics':{'cold':'visually non-burning particle','hot':'visibly luminous/burning particle','particle_unknown':'particle exists but cold/hot state not established','ignore':'unassessable region; not a confirmed missed particle'},'notes':['Preannotation-assisted review is not blind human annotation.','Empty JSON does not establish an empty image.','Do not use annotated overlays from vis/ as raw images.','Object identities here are per-frame only, not MOT ground truth.']})
    print(f'LABELME_TASKS {tasks}: {n} images ({len(blind)} without seed boxes)')
    return out


def import_tasks(review_dir,output_dir,annotator,allow_pending=False):
    root=Path(review_dir);out=Path(output_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    manifest=read_table(root/'review_manifest.csv');rows=[];frames=[];pending=[];sources=[]
    if manifest.duplicated(['recording_id','frame_idx']).any():raise ValueError('Duplicate review frame')
    for r in manifest.itertuples():
        path=Path(r.json_path);task=Path(r.task_image)
        if file_hash(task)!=r.task_image_sha256:raise ValueError('Task image was modified after export')
        data=json.loads(path.read_text(encoding='utf-8-sig'));flags=data.get('flags',{})
        if not boolean(flags.get('reviewed',False)) or not boolean(flags.get('complete',False)):
            pending.append({'recording_id':r.recording_id,'frame_idx':r.frame_idx,'json_path':str(path)});continue
        if file_hash(Path(r.original_image))!=r.source_image_sha256:raise ValueError('Original evaluation image changed after task export')
        if data['imageHeight']!=r.height or data['imageWidth']!=r.width or Path(data['imagePath']).name!=task.name:raise ValueError('JSON image geometry/reference changed')
        for i,shape in enumerate(data.get('shapes',[])):
            if shape.get('shape_type')!='rectangle' or len(shape['points'])!=2:raise ValueError('Use rectangle shapes for detector box reference')
            label=shape['label'];mapping={'cold':'cold','hot':'hot','particle_unknown':'unknown','ignore':'unknown'}
            if label not in mapping:raise ValueError(f'Unknown annotation label {label}')
            a,b=np.asarray(shape['points'],dtype=float);x1,y1=np.minimum(a,b);x2,y2=np.maximum(a,b)
            if not np.isfinite([x1,y1,x2,y2]).all() or not (0<=x1<x2<=r.width and 0<=y1<y2<=r.height):raise ValueError('Invalid reviewed box coordinates')
            rows.append({'recording_id':r.recording_id,'frame_idx':int(r.frame_idx),'ref_object_id':f'{r.recording_id}:frame{int(r.frame_idx)}:object{i}','x1':x1,'y1':y1,'x2':x2,'y2':y2,'cx':(x1+x2)/2,'cy':(y1+y2)/2,'state':mapping[label],'ignore':label=='ignore','size_bin':None,'annotator':annotator,'annotation_source':'human_review_of_preannotation' if boolean(r.preannotated) else 'human_from_blank'})
        frames.append({'recording_id':r.recording_id,'frame_idx':int(r.frame_idx),'image_path':r.original_image,'annotated':True,'sampling_stratum':r.sampling_stratum,'inclusion_probability':r.inclusion_probability,'annotation_source':'assisted' if boolean(r.preannotated) else 'blank'})
        sources.append({'json_path':str(path),'sha256':file_hash(path),'original_image_sha256':r.source_image_sha256})
    if pending and not allow_pending:raise ValueError(f'{len(pending)} images not reviewed+complete; finish them, or explicitly use --allow-pending for a partial review scope')
    if not frames:raise ValueError('No reviewed complete frames')
    write_csv(out/'frames.csv',frames);write_csv(out/'reference_boxes.csv',rows,['recording_id','frame_idx','ref_object_id','x1','y1','x2','y2','cx','cy','state','ignore','size_bin','annotator','annotation_source']);write_csv(out/'pending_frames.csv',pending,['recording_id','frame_idx','json_path'])
    write_json(out/'annotation_provenance.json',{'annotator':annotator,'frames':len(frames),'boxes':len(rows),'pending':len(pending),'source_manifest_sha256':file_hash(root/'review_manifest.csv'),'annotation_files':sources,'scope':'reviewed complete frame box reference; no cross-frame identities','note':'Detector evaluation must use the unchanged detector outputs, not these corrected GT boxes as predictions.'})
    print(f'REVIEWED_REFERENCE {out}: {len(frames)} frames, {len(rows)} boxes')
    return out


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('export');e.add_argument('--config',required=True)
    i=sub.add_parser('import');i.add_argument('--review-dir',required=True);i.add_argument('--output',required=True);i.add_argument('--annotator',required=True);i.add_argument('--allow-pending',action='store_true')
    a=p.parse_args(argv)
    if a.command=='export':export_tasks(json.loads(Path(a.config).read_text(encoding='utf-8-sig')))
    else:import_tasks(a.review_dir,a.output,a.annotator,a.allow_pending)
