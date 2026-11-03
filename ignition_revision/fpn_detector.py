"""Full-frame Faster R-CNN + ResNet18/50 FPN, explicit cold/hot labels.

No network downloads. AP50 is box AP at IoU .5, not COCO AP or ignition-event AP.
"""
from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from .common import Blocked, file_hash, read_table, resolve, write_csv, write_json
from .comparison_data import FullFrameDataset, collate_detection


def build_model(settings):
    import torch
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.models.detection.rpn import AnchorGenerator
    name=settings.get('backbone','resnet18')
    if name not in ('resnet18','resnet50'):raise ValueError('Supported backbones: resnet18, resnet50')
    backbone=resnet_fpn_backbone(backbone_name=name,weights=None,trainable_layers=int(settings.get('trainable_layers',5)))
    if settings.get('backbone_weights'):
        state=torch.load(settings['backbone_weights'],map_location='cpu',weights_only=True)
        if 'state_dict' in state:state=state['state_dict']
        state={k:v for k,v in state.items() if not k.startswith('fc.')}
        missing,unexpected=backbone.body.load_state_dict(state,strict=False)
        if missing or unexpected:raise ValueError(f'Backbone weights mismatch: {missing}, {unexpected}')
    sizes=settings.get('anchor_sizes',[4,8,16,32,64])
    if len(sizes)!=5 or any(float(s)<=0 for s in sizes):raise ValueError('Specify five positive FPN anchor sizes')
    anchors=AnchorGenerator(tuple((int(s),) for s in sizes),((.5,1.,2.),)*5)
    keys=('min_size','max_size','box_detections_per_img','box_score_thresh','box_nms_thresh','rpn_pre_nms_top_n_train','rpn_post_nms_top_n_train','rpn_pre_nms_top_n_test','rpn_post_nms_top_n_test','box_batch_size_per_image')
    kwargs={k:settings[k] for k in keys if k in settings}
    count=int(settings.get('num_classes',3))
    if count<2:raise ValueError('num_classes includes background and must be >=2')
    model=FasterRCNN(backbone,num_classes=count,rpn_anchor_generator=anchors,**kwargs)
    if settings.get('ignore_regions',False):
        from .partial_detection import install_partial_labels
        install_partial_labels(model,settings.get('ignore_ioa',.5))
    if settings.get('keep_original_resolution',False):
        from .partial_detection import IgnoreTransform
        model.transform.__class__=IgnoreTransform;model.transform.keep_original_resolution=True
    return model


def class_ap50(predictions,targets,class_names=None,ignore_ioa=.5):
    from torchvision.ops import box_iou
    import torch
    result={}
    for cls,name in (class_names or {1:'cold',2:'hot'}).items():
        ground={i:t['boxes'][t['labels']==cls].cpu() for i,t in enumerate(targets)}
        total=sum(len(x) for x in ground.values());used={i:set() for i in ground};ranked=[]
        for i,p in enumerate(predictions):
            for box,score,label in zip(p['boxes'].cpu(),p['scores'].cpu(),p['labels'].cpu()):
                if int(label)==cls:ranked.append((float(score),i,box))
        ranked.sort(key=lambda v:-v[0]);tp=[]
        for _,i,box in ranked:
            truth=ground[i];ok=False
            if len(truth):
                overlaps=box_iou(box[None],truth)[0]
                # Highest-IoU unmatched object; already matched GT is not reused.
                available=[j for j in range(len(truth)) if j not in used[i]]
                if available:
                    j=max(available,key=lambda j:float(overlaps[j]))
                    if float(overlaps[j])>=.5:used[i].add(j);ok=True
            if not ok and targets[i].get('ignore_boxes') is not None:
                regions=targets[i]['ignore_boxes'].cpu()
                if len(regions):
                    inter=(torch.minimum(box[None,2:],regions[:,2:])-torch.maximum(box[None,:2],regions[:,:2])).clamp(min=0).prod(dim=1)
                    if (inter/((box[2]-box[0])*(box[3]-box[1])).clamp(min=1e-8)).max()>=ignore_ioa:continue
            tp.append(int(ok))
        if total:
            cum=np.cumsum(tp);rec=cum/total;prec=cum/np.arange(1,len(tp)+1)
            mrec=np.r_[0,rec,1];mp=np.r_[0,prec,0];mp=np.maximum.accumulate(mp[::-1])[::-1]
            changes=np.flatnonzero(mrec[1:]!=mrec[:-1]);ap=float(np.sum((mrec[changes+1]-mrec[changes])*mp[changes+1]))
        else:ap=None
        result[name]={'AP50':ap,'n_gt':total,'n_predictions':len(ranked),'n_scored_predictions':len(tp)}
    present=[r['AP50'] for r in result.values() if r['AP50'] is not None]
    result['mAP50']=float(np.mean(present)) if present else None
    return result


def dataset_signature(dataset):
    # Include the actual image evidence, not only mutable manifest paths.
    return {str(dataset.image_path(i).resolve()):file_hash(dataset.image_path(i)) for i in range(len(dataset))}


def check_split(train,val,cfg):
    th=dataset_signature(train);vh=dataset_signature(val)
    if set(th.values())&set(vh.values()):raise ValueError('Training/validation image content leakage')
    tr=set(train.frames.recording_id.astype(str));va=set(val.frames.recording_id.astype(str));overlap=tr&va
    policy=cfg.get('split_policy','recording_disjoint')
    if policy not in ('recording_disjoint','temporal_blocked_same_recording'):raise ValueError('Unknown split_policy')
    if overlap and policy=='recording_disjoint':raise ValueError('Train/validation recording overlap')
    if overlap:
        gap=int(cfg.get('temporal_purge_frames',0))
        if gap<1:raise ValueError('Temporal split requires positive temporal_purge_frames')
        for rec in overlap:
            a=train.frames.loc[train.frames.recording_id.astype(str)==rec,'frame_idx'].to_numpy(int)
            b=np.sort(val.frames.loc[val.frames.recording_id.astype(str)==rec,'frame_idx'].to_numpy(int))
            pos=np.searchsorted(b,a);nearest=np.minimum(np.abs(a-b[np.clip(pos,0,len(b)-1)]),np.abs(a-b[np.clip(pos-1,0,len(b)-1)]))
            if nearest.min()<=gap:raise ValueError('Temporal purge gap violated')
            def ids(ds):
                frames=set(ds.frames.loc[ds.frames.recording_id.astype(str)==rec,'frame_idx'])
                objects=pd.concat([ds.boxes,ds.ignores],ignore_index=True)
                return set(objects.loc[(objects.recording_id.astype(str)==rec)&objects.frame_idx.isin(frames),'track_id'].dropna().astype(str))
            if ids(train)&ids(val):raise ValueError('Track identity crosses temporal training/validation split; purge crossing tracks/frames')
    return {'train_images':th,'validation_images':vh,'split_policy':policy,'independent_recordings':not bool(overlap),'temporal_purge_frames':cfg.get('temporal_purge_frames')}


def train_fpn(cfg,run):
    import torch
    from torch.utils.data import DataLoader
    seed=int(cfg.get('seed',2026));random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if cfg.get('torch_threads'):torch.set_num_threads(int(cfg['torch_threads']))
    device=torch.device(cfg.get('device','cpu'))
    root=resolve(cfg,cfg['dataset_dir']);allow=bool(cfg.get('allow_draft_labels',False))
    partial=bool(cfg.get('model',{}).get('ignore_regions',False))
    train=FullFrameDataset(root,'train',allow,allow_ignore=partial);val=FullFrameDataset(root,'validation',allow,allow_ignore=partial)
    with run.time('dataset_verification'):splitmeta=check_split(train,val,cfg)
    settings=dict(cfg.get('model',{}));settings.setdefault('num_classes',len(train.class_names)+1)
    if settings['num_classes']!=len(train.class_names)+1 or train.class_names!=val.class_names:raise ValueError('Model and dataset category schema mismatch')
    if settings.get('backbone_weights'):settings['backbone_weights']=str(resolve(cfg,settings['backbone_weights']))
    with run.time('model_initialization',device=str(device)):model=build_model(settings).to(device)
    optim=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=float(cfg.get('learning_rate',.005)),momentum=.9,weight_decay=float(cfg.get('weight_decay',.0005)))
    loader=DataLoader(train,batch_size=int(cfg.get('batch_size',2)),shuffle=True,num_workers=int(cfg.get('num_workers',0)),collate_fn=collate_detection)
    vl=DataLoader(val,batch_size=1,shuffle=False,num_workers=0,collate_fn=collate_detection)
    history=[];best=-1.;epochs=int(cfg.get('epochs',10))
    if epochs<1:raise ValueError('epochs must be positive')
    provenance={'architecture':'FasterRCNN_'+settings.get('backbone','resnet18')+'_FPN','model_settings':{k:v for k,v in settings.items() if k!='backbone_weights'},'classes':{'0':'background',**{str(k):v for k,v in train.class_names.items()}},'partial_labels':partial,'dataset_scope':train.metadata.get('scope'),'allow_draft_labels':allow,'evaluation_scope':'draft_algorithm_labels' if allow else 'reviewed_full_frame_boxes','split':splitmeta,'dataset_files':{n:file_hash(root/n) for n in ('frames.csv','boxes.csv','ignore_boxes.csv','dataset_meta.json') if (root/n).is_file()},'backbone_weights_sha256':file_hash(Path(settings['backbone_weights'])) if settings.get('backbone_weights') else None,'seed':seed,'train_recording_ids':sorted(train.frames.recording_id.astype(str).unique().tolist()),'validation_recording_ids':sorted(val.frames.recording_id.astype(str).unique().tolist())}
    for epoch in range(epochs):
        model.train();losses=[]
        for images,targets in loader:
            images=[im.to(device) for im in images];targets=[{k:v.to(device) for k,v in t.items()} for t in targets]
            with run.time('fpn_train',units=len(images),unit='images',device=str(device)):
                optim.zero_grad(set_to_none=True);parts=model(images,targets);loss=sum(parts.values())
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite detector loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optim.step()
            losses.append(float(loss.detach()))
        model.eval();predictions=[];truth=[]
        with torch.no_grad():
            for images,targets in vl:
                with run.time('fpn_validation',units=len(images),unit='images',device=str(device)):
                    pred=model([im.to(device) for im in images])
                predictions.extend({k:v.cpu() for k,v in p.items()} for p in pred);truth.extend(targets)
        metrics=class_ap50(predictions,truth,train.class_names,float(settings.get('ignore_ioa',.5)));score=metrics['mAP50']
        if score is None:raise ValueError('Validation contains no objects; mAP50 undefined')
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'validation_mAP50':score,**{'validation_'+name+'_AP50':metrics[name]['AP50'] for name in train.class_names.values()}})
        checkpoint={'model':model.state_dict(),'optimizer':optim.state_dict(),'epoch':epoch+1,'provenance':provenance,'validation_metrics':metrics}
        torch.save(checkpoint,run.out/'last.pt')
        if score>best:best=score;torch.save(checkpoint,run.out/'best.pt')
        write_csv(run.out/'training_history.csv',history)
        print(f'epoch {epoch+1}/{epochs} loss={history[-1]["train_loss"]:.5f} val_mAP50={score:.5f}',flush=True)
    write_json(run.out/'training_provenance.json',provenance);run.record('FPN_FASTER_RCNN','completed')


def infer_fpn(cfg,run):
    import torch
    if cfg.get('torch_threads'):torch.set_num_threads(int(cfg['torch_threads']))
    device=torch.device(cfg.get('device','cpu'))
    checkpoint_path=resolve(cfg,cfg['checkpoint']);checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    provenance=checkpoint['provenance'];model=build_model(provenance['model_settings']);model.load_state_dict(checkpoint['model']);model.to(device).eval()
    frames=read_table(resolve(cfg,cfg['frames']))
    if cfg.get('split'):frames=frames.loc[frames.split==cfg['split']]
    if frames.empty:raise ValueError('No inference frames')
    if frames.duplicated(['recording_id','frame_idx']).any():raise ValueError('Duplicate inference frame')
    class_names={int(k):v for k,v in provenance['classes'].items() if int(k)>0}
    seen=provenance['split'];training_hashes=set(seen['train_images'].values());validation_hashes=set(seen['validation_images'].values())
    rows=[];processed_rows=[];predictions=[];targets=[];image_hashes={};root=resolve(cfg,cfg.get('dataset_dir','.'))
    threshold=float(cfg.get('score_threshold',.05))
    if not 0<=threshold<=1:raise ValueError('score_threshold must be in [0,1]')
    evaluation=bool(cfg.get('evaluate',False));dataset=None
    if evaluation and cfg.get('evaluation_split_policy','recording_disjoint')!='recording_disjoint' and not cfg.get('allow_development_evaluation',False):raise ValueError('Use recording-disjoint test data, or explicitly mark same-recording evaluation as development')
    if evaluation:
        # Exact frame key alignment; no row-position assumption.
        dataset=FullFrameDataset(root,cfg.get('split'),bool(cfg.get('allow_draft_labels',False)),allow_ignore=provenance.get('partial_labels',False))
        if len(dataset.ignores):raise ValueError('Complete independent box scoring requires reviewed frames without ignored unknown objects; use evaluate=false for partial-label inference')
        indexes={(r.recording_id,int(r.frame_idx)):i for i,r in dataset.frames.iterrows()}
    for r in frames.itertuples():
        path=Path(r.image_path);path=path if path.is_absolute() else root/path
        sha=file_hash(path);image_hashes[str(path.resolve())]=sha
        if evaluation and (sha in training_hashes or sha in validation_hashes) and not cfg.get('allow_development_evaluation',False):raise ValueError('Formal evaluation image was used for training/validation')
        if evaluation and cfg.get('evaluation_split_policy','recording_disjoint')=='recording_disjoint':
            # Image content guard alone cannot establish independent recording evaluation.
            train_records=set(cfg.get('training_recording_ids',[]))
            # Recover recording IDs from checkpoint added by current writer.
            train_records.update(provenance.get('train_recording_ids',[]));train_records.update(provenance.get('validation_recording_ids',[]))
            if str(r.recording_id) in train_records and not cfg.get('allow_development_evaluation',False):raise ValueError('Evaluation recording overlaps development recordings')
        with Image.open(path) as im:
            if im.mode.startswith(('I','F')):raise ValueError('Provide documented 8-bit image conversion')
            a=np.asarray(im.convert('RGB')).copy()
        processed_rows.append({'recording_id':r.recording_id,'frame_idx':int(r.frame_idx),'image_path':str(path.resolve()),'image_sha256':sha})
        tensor=torch.from_numpy(a).permute(2,0,1).float()/255.
        with torch.no_grad(),run.time('fpn_inference',unit='images',device=str(device)):
            pred={k:v.cpu() for k,v in model([tensor.to(device)])[0].items()}
        # Metrics use all model outputs; output threshold is a separate operating point.
        if evaluation:
            key=(r.recording_id,int(r.frame_idx))
            if key not in indexes:raise ValueError('Evaluation frame missing in complete annotation dataset')
            _,target=dataset[indexes[key]];targets.append(target);predictions.append(pred)
        for j,(box,score,label) in enumerate(zip(pred['boxes'],pred['scores'],pred['labels'])):
            if float(score)<threshold:continue
            x1,y1,x2,y2=map(float,box)
            rows.append({'recording_id':r.recording_id,'frame_idx':int(r.frame_idx),'candidate_id':f'{r.recording_id}:{r.frame_idx}:{j}','x1':x1,'y1':y1,'x2':x2,'y2':y2,'cx':(x1+x2)/2,'cy':(y1+y2)/2,'class_id':int(label),'class_label':class_names[int(label)],'score':float(score),'native_score':float(score),'score_semantics':'model_class_score_not_calibrated','prompt_group':class_names[int(label)],'source':'FPN_FASTER_RCNN'})
    columns=['recording_id','frame_idx','candidate_id','x1','y1','x2','y2','cx','cy','class_id','class_label','score','native_score','score_semantics','prompt_group','source']
    write_csv(run.out/'full_frame_detections.csv',rows,columns)
    write_csv(run.out/'inferred_frames.csv',processed_rows,['recording_id','frame_idx','image_path','image_sha256'])
    if evaluation:
        from .evaluation import detector_metrics,detector_strata
        table=pd.DataFrame(rows,columns=columns)
        gt=dataset.boxes.copy();gt['state']=gt.class_id.map(class_names);gt['ref_object_id']=gt.box_id
        scope=frames[['recording_id','frame_idx']].copy()
        score,per_frame=detector_metrics(table,gt,scope,float(cfg.get('iou_threshold',.5)))
        score.update({'method_id':'FPN_FASTER_RCNN','evaluation_scope':'development' if cfg.get('allow_development_evaluation') else ('draft_labels' if cfg.get('allow_draft_labels') else 'reviewed_test'),'score_threshold':threshold,'iou_threshold':float(cfg.get('iou_threshold',.5))})
        write_csv(run.out/'S03_detector_comparison.csv',[score]);write_csv(run.out/'per_frame_metrics.csv',per_frame)
        strata,members=detector_strata(table,gt,scope,float(cfg.get('iou_threshold',.5)))
        write_csv(run.out/'detector_by_state_size.csv',strata);write_csv(run.out/'reference_detection_members.csv',members)
        metrics=class_ap50(predictions,targets,class_names);metrics['evaluation_scope']='development' if cfg.get('allow_development_evaluation') else ('draft_labels' if cfg.get('allow_draft_labels') else 'reviewed_test');write_json(run.out/'box_metrics_AP50.json',metrics)
    write_json(run.out/'inference_provenance.json',{'checkpoint':str(checkpoint_path),'sha256':file_hash(checkpoint_path),'image_sha256':image_hashes,'export_score_threshold':threshold,'detections_sha256':file_hash(run.out/'full_frame_detections.csv'),'inferred_frames_sha256':file_hash(run.out/'inferred_frames.csv'),'model_provenance':provenance,'note':'Per-frame class-labelled boxes; ignition onset requires track association. AP50 is not event timing accuracy.'})
    run.record('FPN_FASTER_RCNN','completed')
