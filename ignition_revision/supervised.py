"""Supervised state baseline. Labels must be independent per-frame states, not post-onset guesses."""
from __future__ import annotations
import inspect
import random
import numpy as np
from PIL import Image
from .common import Blocked, file_hash, read_table, require_columns, resolve, write_csv, write_json


def dependencies():
    try:
        import torch
        import torchvision
        from torchvision.models import resnet18
        return torch,torchvision,resnet18
    except (ImportError,OSError,RuntimeError) as e:
        raise Blocked(f'Working compatible torch/torchvision installation required: {e}') from e


def preprocessing(spec):
    _,tv,_=dependencies()
    for key in ('resize','mean','std'):
        if key not in spec:raise Blocked(f'Explicit ResNet preprocessing.{key} required')
    if len(spec['mean'])!=3 or len(spec['std'])!=3 or min(spec['std'])<=0:raise ValueError('Invalid RGB normalization')
    return tv.transforms.Compose([tv.transforms.Resize(tuple(spec['resize'])),tv.transforms.ToTensor(),tv.transforms.Normalize(spec['mean'],spec['std'])])


def load_checkpoint(torch,path,device):
    kwargs={'map_location':device}
    if 'weights_only' in inspect.signature(torch.load).parameters:kwargs['weights_only']=True
    return torch.load(path,**kwargs)


def train_state(cfg,run):
    torch,tv,resnet18=dependencies()
    table=read_table(resolve(cfg,cfg.get('state_labels')))
    require_columns(table,['recording_id','ref_object_id','frame_idx','roi_path','state_label','split'],'state training labels')
    if not cfg.get('label_provenance'):raise Blocked('Independent frame-state label_provenance required')
    allowed={'cold','hot','ignore'}
    if set(table.state_label)-allowed:raise ValueError('Use cold/hot/ignore state labels')
    if set(table.split)-{'train','validation','test'}:raise ValueError('Unknown split role')
    rec_sets={role:set(table[table.split==role].recording_id.astype(str)) for role in ('train','validation','test')}
    if any(rec_sets[a]&rec_sets[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):raise ValueError('Recording leakage between training/validation/test')
    image_hashes={role:{file_hash(resolve(cfg,x)) for x in table.loc[table.split==role,'roi_path']} for role in ('train','validation','test')}
    if any(image_hashes[a]&image_hashes[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):raise ValueError('Duplicate source ROI files across data splits, even if recording IDs differ')
    train=table[(table.split=='train')&(table.state_label!='ignore')];valid=table[(table.split=='validation')&(table.state_label!='ignore')]
    if train.empty or valid.empty or set(train.state_label)!={'cold','hot'}:raise Blocked('Need nonempty recording-disjoint train/validation and both training classes')
    settings=cfg.get('training',{})
    for k in ('epochs','batch_size','learning_rate','seed','preprocessing'):
        if k not in settings or settings[k] is None:raise Blocked(f'Explicit training.{k} required')
    seed=int(settings['seed']);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    device=cfg.get('device','cpu');transform=preprocessing(settings['preprocessing'])
    class Dataset(torch.utils.data.Dataset):
        def __init__(self,rows):self.rows=rows.reset_index(drop=True)
        def __len__(self):return len(self.rows)
        def __getitem__(self,index):
            row=self.rows.iloc[index]
            with Image.open(resolve(cfg,row.roi_path)) as im:image=transform(im.convert('RGB'))
            return image,int(row.state_label=='hot')
    dataset=Dataset(train);val=Dataset(valid)
    # Object-balanced sampling prevents long trajectories from determining the training objective.
    sizes=train.groupby(['recording_id','ref_object_id']).size()
    weights=[1/float(sizes.loc[(r.recording_id,r.ref_object_id)]) for _,r in train.iterrows()]
    generator=torch.Generator().manual_seed(seed)
    sampler=torch.utils.data.WeightedRandomSampler(weights,len(weights),replacement=True,generator=generator)
    train_loader=torch.utils.data.DataLoader(dataset,batch_size=settings['batch_size'],sampler=sampler,num_workers=0)
    valid_loader=torch.utils.data.DataLoader(val,batch_size=settings['batch_size'],shuffle=False,num_workers=0)
    with run.time('resnet_model_load',unit='models',device=device):
        model=resnet18(weights=None);model.fc=torch.nn.Linear(model.fc.in_features,2);model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=settings['learning_rate']);loss_fn=torch.nn.CrossEntropyLoss()
    curves=[];best=float('inf')
    for epoch in range(int(settings['epochs'])):
        model.train();total=0.;n=0
        with run.time('resnet_training_epoch',len(train),'state_frames',device):
            for x,y in train_loader:
                x,y=x.to(device),y.to(device);optimizer.zero_grad();loss=loss_fn(model(x),y);loss.backward();optimizer.step()
                total+=float(loss.detach().cpu())*len(y);n+=len(y)
        model.eval();valid_loss=0.;vn=0
        with run.time('resnet_validation_epoch',len(valid),'state_frames',device):
            with torch.no_grad():
                for x,y in valid_loader:
                    x,y=x.to(device),y.to(device);loss=loss_fn(model(x),y);valid_loss+=float(loss.cpu())*len(y);vn+=len(y)
        value=valid_loss/vn;curves.append({'epoch':epoch,'train_loss':total/n,'validation_loss':value})
        if value<best:
            best=value
            torch.save({'state_dict':model.state_dict(),'architecture':'resnet18','class_order':['cold','hot'],
                        'preprocessing':settings['preprocessing'],'train_recordings':sorted(rec_sets['train']),
                        'validation_recordings':sorted(rec_sets['validation']),
                        'train_image_hashes':sorted(image_hashes['train']),'validation_image_hashes':sorted(image_hashes['validation']),'label_sha256':file_hash(resolve(cfg,cfg['state_labels'])),
                        'epoch':epoch,'initialization':'random_no_download','selection':'validation_loss'},run.out/'best_resnet18.pt')
        write_csv(run.out/'training_curve.csv',curves)
    budget=[]
    for role,subset in [('train',train),('validation',valid)]:
        budget.append({'role':role,'recordings':subset.recording_id.nunique(),'objects':len(subset[['recording_id','ref_object_id']].drop_duplicates()),
                       'state_frames':len(subset),'hot_frames':int((subset.state_label=='hot').sum()),'cold_frames':int((subset.state_label=='cold').sum())})
    write_csv(run.out/'supervised_training_budget.csv',budget)
    run.record('J_RESNET18_TRAIN','completed')


_MODEL_CACHE={}

def infer_state_probabilities(df,cfg,settings):
    torch,tv,resnet18=dependencies()
    path=resolve(cfg,settings.get('checkpoint'))
    if not path.is_file():raise Blocked('A separately trained ResNet checkpoint is required')
    device=settings.get('device',cfg.get('device','cpu'));key=(str(path),file_hash(path),device)
    if key not in _MODEL_CACHE:
        ckpt=load_checkpoint(torch,path,device)
        if ckpt.get('architecture')!='resnet18' or ckpt.get('class_order')!=['cold','hot']:raise ValueError('Incompatible state checkpoint')
        model=resnet18(weights=None);model.fc=torch.nn.Linear(model.fc.in_features,2);model.load_state_dict(ckpt['state_dict']);model.to(device).eval()
        _MODEL_CACHE[key]=(model,preprocessing(ckpt['preprocessing']),ckpt)
    model,transform,meta=_MODEL_CACHE[key]
    if not settings.get('allow_development_inference',False):
        used=set(meta['train_recordings'])|set(meta['validation_recordings'])
        if used & set(df.recording_id.astype(str)):raise ValueError('Test inference overlaps training/validation recordings')
        if 'train_image_hashes' not in meta or 'validation_image_hashes' not in meta:raise Blocked('Checkpoint lacks source-image provenance; regenerate using current trainer')
        used_images=set(meta['train_image_hashes'])|set(meta['validation_image_hashes'])
        if any(file_hash(resolve(cfg,p)) in used_images for p in df.roi_path):raise ValueError('Test ROI bytes overlap training/validation, even under renamed recordings')
    values=[]
    with torch.no_grad():
        for _,row in df.iterrows():
            with Image.open(resolve(cfg,row.roi_path)) as im:x=transform(im.convert('RGB')).unsqueeze(0).to(device)
            values.append(float(model(x).softmax(dim=1)[0,1].cpu()))
    return np.asarray(values)
