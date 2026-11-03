"""Traceable conversion of legacy tracks/event annotation to ROI and full-frame data.

Algorithm boxes and inferred states remain drafts. Frame completeness is a separate
human decision. Never turn an unannotated object or a non-event track into cold GT.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from .common import file_hash, read_table, require_columns, write_csv, write_json

BOX_COLUMNS=['recording_id','frame_idx','box_id','track_id','x1','y1','x2','y2','class_id','review_status','point_type','label_source']


def tid(value):
    m=re.fullmatch(r'(?:track_)?(\d+)',str(value))
    if not m:raise ValueError(f'Invalid legacy track ID: {value}')
    return int(m.group(1))


def integral(values,name):
    v=pd.to_numeric(values,errors='raise')
    if not np.isfinite(v).all() or (v%1!=0).any() or (v<0).any():raise ValueError(f'{name} must contain nonnegative integers')
    return v.astype(int)


def event_labels(annotation,crops,recording,review_book=None):
    gt=read_table(annotation);require_columns(gt,['track_id','frame_id'],'effective_trace annotation')
    gt=gt.copy();gt['track_id']=gt.track_id.map(tid);gt['frame_id']=integral(gt.frame_id,'local onset')
    if gt.track_id.duplicated().any():raise ValueError('Duplicate annotated track')
    out=[];groups={int(k):g.reset_index(drop=True) for k,g in crops.groupby('track_id',sort=False)}
    for r in gt.itertuples():
        g=groups.get(r.track_id)
        if g is None or r.frame_id>=len(g):raise ValueError(f'Unmapped local onset: track {r.track_id}, index {r.frame_id}')
        out.append({'recording_id':recording,'track_id':r.track_id,'has_ignition':True,'onset_frame':int(g.iloc[r.frame_id].frame_id),'onset_local_idx':int(r.frame_id),'label_source':'manual_effective_trace'})
    if review_book:
        book=pd.read_excel(review_book,sheet_name='model_pred')
        track,valid,ignition,onset='\u8f68\u8ff9','\u662f\u5426\u6709\u6548','\u662f\u5426\u70b9\u71c3','\u70b9\u706b\u5e27\u533a\u95f4'
        yes,no='\u662f','\u5426';require_columns(book,[track,valid,ignition,onset],'manual review')
        positives={r['track_id']:r['onset_local_idx'] for r in out}
        for _,r in book.iterrows():
            if r[valid]!=yes:continue
            k=tid(r[track])
            if r[ignition]==yes and pd.notna(r[onset]):
                if k not in positives or float(r[onset])!=positives[k]:raise ValueError('Manual review and effective_trace version conflict')
            if r[ignition]==no:
                if k in positives:raise ValueError('Positive/negative GT conflict')
                if k in groups:out.append({'recording_id':recording,'track_id':k,'has_ignition':False,'onset_frame':None,'onset_local_idx':None,'label_source':'manual_explicit_non_event_state_unknown'})
    return pd.DataFrame(out)


def build_dataset(cfg):
    out=Path(cfg['output_dir']).resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'Refusing to replace dataset: {out}')
    tracks=read_table(cfg['tracks_csv']).copy();require_columns(tracks,['track_id','frame_id','image_name','x1','y1','x2','y2'],'tracks')
    tracks['track_id']=tracks.track_id.map(tid);tracks['frame_id']=integral(tracks.frame_id,'global frames')
    if tracks.duplicated(['track_id','frame_id']).any():raise ValueError('Duplicate track/frame')
    coords=tracks[['x1','y1','x2','y2']].to_numpy(float)
    if not np.isfinite(coords).all() or (tracks.x2<=tracks.x1).any() or (tracks.y2<=tracks.y1).any():raise ValueError('Invalid tracks boxes')
    crops=read_table(cfg['crop_summary_csv']).copy();require_columns(crops,['track_id','frame_id','status'],'crop export summary')
    crops=crops.loc[crops.status=='ok'].copy();crops['track_id']=crops.track_id.map(tid);crops['frame_id']=integral(crops.frame_id,'crop frame')
    if crops.duplicated(['track_id','frame_id']).any():raise ValueError('Duplicate crop/frame')
    # Match the actual per-track sorted ROI sequence used by the legacy loader.
    crops=crops.sort_values(['track_id','frame_id'],kind='stable')
    rec=cfg['recording_id'];labels=event_labels(cfg['annotation'],crops,rec,cfg.get('review_book'))
    known=labels.set_index('track_id').to_dict('index')
    image_root=Path(cfg['image_root']);image_map={}
    for p in image_root.iterdir():
        if p.suffix.lower() in ('.png','.tif','.tiff','.jpg','.jpeg','.bmp'):
            if p.stem in image_map:raise ValueError('Ambiguous image stem')
            image_map[p.stem]=p.resolve()
    observed=tracks.loc[tracks.point_type.eq('detected')] if 'point_type' in tracks else tracks
    frame_names=observed.groupby('frame_id').image_name.agg(lambda x:list(set(Path(str(s)).stem for s in x)))
    if any(len(v)!=1 for v in frame_names):raise ValueError('Observed frame has inconsistent image names')
    if set(tracks.frame_id)-set(frame_names.index):raise ValueError('Frame has no observed image association; provide corrected canonical frame names')
    mismatches=[]
    for r in tracks.itertuples():
        canonical=frame_names.loc[r.frame_id][0]
        if Path(str(r.image_name)).stem!=canonical:mismatches.append({'track_id':r.track_id,'frame_idx':r.frame_id,'stored_image_name':r.image_name,'canonical_observed_image':canonical,'point_type':getattr(r,'point_type','unknown')})
    mismatch_keys={(int(r['track_id']),int(r['frame_idx'])) for r in mismatches}
    labels['onset_image_name_conflict']=[bool(r.has_ignition and (int(r.track_id),int(r.onset_frame)) in mismatch_keys) for r in labels.itertuples()]
    # Retain all mapping evidence, but do not use a timing-conflicted event to infer states or evaluate onset.
    known={k:v for k,v in known.items() if not (v['has_ignition'] and (int(k),int(v['onset_frame'])) in mismatch_keys)}
    frames=list(frame_names.index)
    if cfg.get('frame_range'):frames=[f for f in frames if cfg['frame_range'][0]<=f<=cfg['frame_range'][1]]
    if cfg.get('max_frames'):frames=frames[:int(cfg['max_frames'])]
    if not frames:raise ValueError('No selected frames')
    selected=tracks.loc[tracks.frame_id.isin(frames)].copy()
    split_map={}
    if cfg.get('split_manifest'):
        sm=read_table(cfg['split_manifest']);require_columns(sm,['recording_id','frame_idx','split'],'split manifest')
        if sm.duplicated(['recording_id','frame_idx']).any():raise ValueError('Duplicate split frame')
        split_map={(str(r.recording_id),int(r.frame_idx)):str(r.split) for r in sm.itertuples()}
    frame_rows=[];boxes=[];rois=[];padding=int(cfg.get('roi_padding',8));guard=int(cfg.get('onset_guard_frames',2))
    if padding<0 or guard<0:raise ValueError('padding and guard must be nonnegative')
    out.mkdir(parents=True,exist_ok=True)
    for f,group in selected.groupby('frame_id',sort=True):
        stem=frame_names.loc[f][0]
        if stem not in image_map:raise FileNotFoundError(stem)
        path=image_map[stem]
        with Image.open(path) as im:
            width,height=im.size
            frame_rows.append({'recording_id':rec,'frame_idx':int(f),'image_path':str(path),'width':width,'height':height,'split':split_map.get((rec,int(f)),'unassigned'),'boxes_complete':False,'states_complete':False,'annotation_status':'draft'})
            for r in group.itertuples():
                if r.x1<0 or r.y1<0 or r.x2>width or r.y2>height:raise ValueError(f'Out-of-image box in frame {f}')
                event=known.get(r.track_id);cls=-1;source='unknown_state'
                if event and event['has_ignition']:
                    delta=f-event['onset_frame']
                    if delta < -guard:cls=1;source='pre_onset_cold_candidate'
                    elif delta > guard:cls=2;source='post_onset_hot_candidate_persistence_assumption'
                point=getattr(r,'point_type','detected')
                if point!='detected':cls=-1;source='algorithm_restoration_not_observation'
                boxes.append({'recording_id':rec,'frame_idx':int(f),'box_id':f'{rec}:{int(f)}:{r.track_id}','track_id':r.track_id,'x1':r.x1,'y1':r.y1,'x2':r.x2,'y2':r.y2,'class_id':cls,'review_status':'draft','point_type':point,'label_source':source})
                if event and cfg.get('export_rois',True):
                    if im.mode.startswith(('I','F')):raise ValueError('Specify and perform documented 8-bit image conversion before ROI export')
                    left=max(0,int(np.floor(r.x1))-padding);top=max(0,int(np.floor(r.y1))-padding)
                    right=min(width,int(np.ceil(r.x2))+padding);bottom=min(height,int(np.ceil(r.y2))+padding)
                    crop_path=out/'rois'/f'track_{r.track_id}'/f'frame_{int(f):08d}.png';crop_path.parent.mkdir(parents=True,exist_ok=True)
                    im.crop((left,top,right,bottom)).convert('RGB').save(crop_path)
                    cx=float(getattr(r,'cx',(r.x1+r.x2)/2));cy=float(getattr(r,'cy',(r.y1+r.y2)/2))
                    rois.append({'recording_id':rec,'track_id':r.track_id,'frame_idx':int(f),'roi_path':str(crop_path),'center_roi_x':cx-left,'center_roi_y':cy-top,'crop_left':left,'crop_top':top,'cx':cx,'cy':cy,'x1':r.x1,'y1':r.y1,'x2':r.x2,'y2':r.y2,'point_type':point})
    write_csv(out/'image_name_conflicts.csv',mismatches,['track_id','frame_idx','stored_image_name','canonical_observed_image','point_type'])
    write_csv(out/'frames.csv',frame_rows);write_csv(out/'boxes.csv',boxes,BOX_COLUMNS)
    write_csv(out/'rois.csv',rois,['recording_id','track_id','frame_idx','roi_path','center_roi_x','center_roi_y','crop_left','crop_top','cx','cy','x1','y1','x2','y2','point_type'])
    labels['ref_object_id']=labels.track_id.map(lambda k:f'{rec}:legacy_track_{k}')
    present=set(r['track_id'] for r in rois)
    write_csv(out/'all_event_mapping_audit.csv',labels)
    write_csv(out/'fixed_cohort_labels.csv',labels.loc[labels.track_id.isin(present)&~labels.onset_image_name_conflict])
    sources={k:{'path':str(Path(cfg[k]).resolve()),'sha256':file_hash(Path(cfg[k]))} for k in ('tracks_csv','crop_summary_csv','annotation','review_book','split_manifest') if cfg.get(k)}
    write_json(out/'dataset_meta.json',{'schema_version':1,'config':cfg,'sources':sources,'frames':len(frame_rows),'boxes':len(boxes),'rois':len(rois),'event_labels':len(labels),'onset_image_conflicts_excluded':int(labels.onset_image_name_conflict.sum()),'image_name_conflicts':len(mismatches),'detector_training_ready':False,'scope':'algorithm_box_and_state_drafts; no independently reviewed full-frame boxes','local_onset_mapping':'zero-based successful crop sequence sorted by global frame','unknown_class_id':-1,'class_names':{'1':'cold','2':'hot'},'note':'Review boxes.csv and frames.csv; unknown state is never background. Non-event is not cold. Images with no tracking rows are not exported and must be added when selecting complete evaluation snippets.'})
    print(f'DATASET_DRAFT {out}: {len(frame_rows)} frames, {len(boxes)} boxes, {len(rois)} ROIs')
    return out


def strict_bool(value):
    if isinstance(value,(bool,np.bool_)):return bool(value)
    if str(value).strip().lower() in ('true','1'):return True
    if str(value).strip().lower() in ('false','0'):return False
    raise ValueError(f'Invalid boolean: {value}')


class FullFrameDataset:
    """Torchvision detection data. Empty targets allowed only on explicitly complete frames.

    In draft experiments the caller must opt in AND provide a class for every box.
    Unknown targets are never dropped and treated as background. No image resizing
    here: torchvision handles reversible resize and restores source coordinates.
    """
    def __init__(self,root,split=None,allow_draft=False,class_names=None,allow_ignore=False):
        self.root=Path(root).resolve();self.frames=read_table(self.root/'frames.csv');self.boxes=read_table(self.root/'boxes.csv')
        meta_path=self.root/'dataset_meta.json'
        self.metadata=json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.is_file() else {}
        self.class_names={int(k):v for k,v in (class_names or self.metadata.get('class_names',{'1':'cold','2':'hot'})).items()}
        if set(self.class_names)!=set(range(1,len(self.class_names)+1)):raise ValueError('Foreground class IDs must be contiguous from 1')
        ignore_path=self.root/'ignore_boxes.csv'
        self.ignores=read_table(ignore_path) if ignore_path.is_file() else pd.DataFrame(columns=BOX_COLUMNS)
        require_columns(self.ignores,BOX_COLUMNS,'ignore boxes')
        if len(self.ignores) and not allow_ignore:raise ValueError('Partial labels require model.ignore_regions=true; unknown regions must not become background')
        self.ignore_groups={k:g for k,g in self.ignores.groupby(['recording_id','frame_idx'])}
        require_columns(self.frames,['recording_id','frame_idx','image_path','split','boxes_complete','states_complete','annotation_status'],'frames')
        require_columns(self.boxes,BOX_COLUMNS,'boxes')
        if self.frames.duplicated(['recording_id','frame_idx']).any() or self.boxes.box_id.duplicated().any():raise ValueError('Duplicate frame/box')
        all_keys=set(zip(self.frames.recording_id,self.frames.frame_idx))
        if set(zip(self.ignores.recording_id,self.ignores.frame_idx))-all_keys:raise ValueError('Ignore boxes reference absent frames')
        if set(zip(self.boxes.recording_id,self.boxes.frame_idx))-all_keys:raise ValueError('Boxes reference frames absent from manifest')
        integral(self.frames.frame_idx,'frame_idx')
        if split:self.frames=self.frames.loc[self.frames.split==split].reset_index(drop=True)
        if self.frames.empty:raise ValueError(f'Empty split: {split}')
        self.groups={key:g for key,g in self.boxes.groupby(['recording_id','frame_idx'])};self.allow_draft=allow_draft
        for r in self.frames.itertuples():
            complete=strict_bool(r.boxes_complete) and strict_bool(r.states_complete) and r.annotation_status=='reviewed'
            if not complete and not allow_draft:raise ValueError(f'Frame not fully reviewed: {r.recording_id}/{r.frame_idx}')
            g=self.groups.get((r.recording_id,r.frame_idx),self.boxes.iloc[:0])
            if not g.class_id.isin(self.class_names).all():raise ValueError('Unknown or incompatible foreground class present')
            if not allow_draft and not g.review_status.eq('reviewed').all():raise ValueError('Unreviewed boxes')
            coords=g[['x1','y1','x2','y2']].to_numpy(float)
            if not np.isfinite(coords).all() or (g.x2<=g.x1).any() or (g.y2<=g.y1).any():raise ValueError('Invalid boxes')
    def __len__(self):return len(self.frames)
    def image_path(self,i):
        p=Path(self.frames.iloc[i].image_path);return p if p.is_absolute() else self.root/p
    def __getitem__(self,i):
        import torch
        r=self.frames.iloc[i];g=self.groups.get((r.recording_id,r.frame_idx),self.boxes.iloc[:0])
        with Image.open(self.image_path(i)) as im:
            if im.mode.startswith(('I','F')):raise ValueError('High bit-depth conversion must be documented upstream')
            a=np.asarray(im.convert('RGB')).copy();h,w=a.shape[:2]
        boxes=torch.as_tensor(g[['x1','y1','x2','y2']].to_numpy(float),dtype=torch.float32).reshape(-1,4)
        if len(boxes) and (torch.any(boxes[:,0::2]<0) or torch.any(boxes[:,0::2]>w) or torch.any(boxes[:,1::2]<0) or torch.any(boxes[:,1::2]>h)):raise ValueError('Box outside image')
        ignore=self.ignore_groups.get((r.recording_id,r.frame_idx),self.ignores.iloc[:0])
        ignored=torch.as_tensor(ignore[['x1','y1','x2','y2']].to_numpy(float),dtype=torch.float32).reshape(-1,4)
        if len(ignored) and (not torch.isfinite(ignored).all() or torch.any(ignored[:,2:]<=ignored[:,:2]) or torch.any(ignored[:,0::2]<0) or torch.any(ignored[:,0::2]>w) or torch.any(ignored[:,1::2]<0) or torch.any(ignored[:,1::2]>h)):raise ValueError('Invalid ignore region')
        target={'ignore_boxes':ignored,'boxes':boxes,'labels':torch.as_tensor(g.class_id.to_numpy(int),dtype=torch.int64),'image_id':torch.tensor([i]),'area':(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]),'iscrowd':torch.zeros(len(boxes),dtype=torch.int64)}
        return torch.from_numpy(a).permute(2,0,1).float()/255.,target


def collate_detection(batch):return tuple(zip(*batch))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);a=p.parse_args(argv)
    build_dataset(json.loads(Path(a.config).read_text(encoding='utf-8-sig')))
