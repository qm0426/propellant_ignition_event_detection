from __future__ import annotations
import argparse
import copy
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_laplace, maximum_filter, label
from .common import Blocked, boolean, digest, load_legacy, read_table, require_columns, resolve, write_csv, write_json

BOX_COLUMNS=['recording_id','frame_idx','candidate_id','x1','y1','x2','y2','prompt_group','native_score','score_semantics']


def validate_frames(df):
    require_columns(df,['recording_id','frame_idx','image_path'],'frames')
    if df.duplicated(['recording_id','frame_idx']).any():raise ValueError('Duplicate frames')
    if (df.frame_idx<0).any() or (df.frame_idx%1!=0).any():raise ValueError('frame_idx must be 0-based integer')
    return df.sort_values(['recording_id','frame_idx'])


def validate_candidates(df):
    require_columns(df,BOX_COLUMNS,'candidates')
    if df.candidate_id.duplicated().any():raise ValueError('candidate_id must be globally unique')
    if not np.isfinite(df[['x1','y1','x2','y2']].to_numpy(dtype=float)).all():raise ValueError('Nonfinite boxes')
    if ((df.x2<=df.x1)|(df.y2<=df.y1)).any():raise ValueError('Boxes require x2>x1, y2>y1')
    return df


def iou_matrix(a,b):
    aa=a[['x1','y1','x2','y2']].to_numpy(float);bb=b[['x1','y1','x2','y2']].to_numpy(float)
    lo=np.maximum(aa[:,None,:2],bb[None,:,:2]);hi=np.minimum(aa[:,None,2:],bb[None,:,2:])
    inter=np.maximum(hi-lo,0).prod(axis=2)
    area_a=(aa[:,2:]-aa[:,:2]).prod(axis=1);area_b=(bb[:,2:]-bb[:,:2]).prod(axis=1)
    return inter/np.maximum(area_a[:,None]+area_b[None,:]-inter,1e-12)


def classical_candidates(image, config, method):
    import cv2
    gray=np.asarray(image.convert('L'),np.float32)
    polarity=config.get('polarity')
    if polarity not in ('bright','dark'):raise ValueError('Specify classical detector polarity=bright|dark')
    sign=1 if polarity=='bright' else -1
    h,w=gray.shape;boxes=[]
    if method=='log':
        if not config.get('sigmas') or config.get('response_threshold') is None:
            raise Blocked('LoG requires development-selected sigmas and response_threshold')
        # The scale-normalized LoG response still has intensity units. Preserve
        # old explicit/raw configs while the corrected preset declares unit input.
        intensity_scale=config.get('intensity_scale','raw_255')
        if intensity_scale not in ('unit','raw_255'):
            raise ValueError('LoG intensity_scale must be unit|raw_255')
        if intensity_scale=='unit':gray=gray/255.
        sigmas=np.asarray(config['sigmas'],float)
        if (not np.isfinite(sigmas).all() or (sigmas<=0).any() or
                (np.diff(sigmas)<=0).any()):
            raise ValueError('LoG sigmas must be finite, positive and strictly increasing')
        threshold=float(config['response_threshold'])
        nms_iou=float(config.get('nms_iou',.3))
        if not np.isfinite(threshold) or threshold<=0:raise ValueError('LoG response_threshold must be positive')
        if not 0<nms_iou<=1:raise ValueError('LoG nms_iou must be in (0,1]')
        responses=np.stack([-sign*s*s*gaussian_laplace(gray,s) for s in sigmas])
        # Suppress in x, y AND adjacent scale; border scales remain eligible.
        peaks=(responses==maximum_filter(responses,size=(3,3,3),mode='nearest'))&(responses>threshold)
        # One representative per flat peak plateau, deterministic on score/ties.
        components,n_components=label(peaks,structure=np.ones((3,3,3),dtype=np.uint8))
        if n_components:
            coordinates=np.argwhere(peaks)
            component_ids=components[tuple(coordinates.T)]
            scores=responses[tuple(coordinates.T)]
            order=np.lexsort((coordinates[:,2],coordinates[:,1],coordinates[:,0],-scores,component_ids))
            ordered_ids=component_ids[order]
            representatives=order[np.r_[True,ordered_ids[1:]!=ordered_ids[:-1]]]
            for k,y,x in coordinates[representatives]:
                sigma=float(sigmas[k]);radius=float(np.sqrt(2)*sigma)
                boxes.append({'x1':max(0,x-radius),'y1':max(0,y-radius),'x2':min(w,x+radius),'y2':min(h,y+radius),
                              'native_score':float(responses[k,y,x]),
                              'score_semantics':'LoG_scale_normalized_response_'+intensity_scale,'sigma':sigma})
        boxes=log_nms(boxes,nms_iou)
    elif method=='mser':
        for k in ('min_area','max_area'):
            if k not in config:raise Blocked('MSER requires development-selected min_area/max_area')
        mser=cv2.MSER_create();mser.setMinArea(int(config['min_area']));mser.setMaxArea(int(config['max_area']))
        _,rectangles=mser.detectRegions((255-gray if polarity=='bright' else gray).astype(np.uint8))
        for x,y,bw,bh in rectangles:
            boxes.append({'x1':int(x),'y1':int(y),'x2':int(x+bw),'y2':int(y+bh),'native_score':None,'score_semantics':'none'})
    else:raise ValueError(method)
    return boxes


def log_nms(boxes,iou_threshold):
    """Score-ordered, deterministic suppression of overlapping LoG scale boxes."""
    ordered=sorted(boxes,key=lambda b:(-b['native_score'],b['sigma'],b['y1'],b['x1']))
    if not ordered:return []
    coords=np.asarray([[b[k] for k in ('x1','y1','x2','y2')] for b in ordered],float)
    area=(coords[:,2]-coords[:,0])*(coords[:,3]-coords[:,1])
    remaining=np.arange(len(ordered));kept=[]
    while len(remaining):
        i=remaining[0];kept.append(ordered[i]);rest=remaining[1:]
        lo=np.maximum(coords[i,:2],coords[rest,:2]);hi=np.minimum(coords[i,2:],coords[rest,2:])
        intersection=np.maximum(hi-lo,0).prod(axis=1)
        overlap=intersection/np.maximum(area[i]+area[rest]-intersection,1e-12)
        remaining=rest[overlap<=iou_threshold]
    return kept


def clean_candidates(raw, frames, config, run, cfg):
    module=load_legacy('postprocess_detection_txt_modified.py',cfg.get('upstream_root'))
    parser=module.build_arg_parser()
    defaults={a.dest:a.default for a in parser._actions if a.dest!='help'}
    unknown=set(config)-set(defaults)-{'disabled_rules'}
    if unknown:raise ValueError(f'Unknown cleaning settings: {sorted(unknown)}')
    args=argparse.Namespace(**{**defaults,**{k:v for k,v in config.items() if k!='disabled_rules'}})
    disabled=set(config.get('disabled_rules',[]));allowed={'background','geometry','defocus','nested','duplicates','nms','all'}
    if disabled-allowed:raise ValueError(f'Unknown disabled cleaning rules: {disabled-allowed}')
    all_rows=[]
    for _,f in frames.iterrows():
        subset=raw[(raw.recording_id==f.recording_id)&(raw.frame_idx==f.frame_idx)]
        with Image.open(resolve(cfg,f.image_path)) as source:image=source.convert('RGB')
        working=copy.deepcopy(subset.to_dict('records'))
        for i,box in enumerate(working):
            box.update({'_idx':i,'score':None if pd.isna(box.get('native_score')) else box['native_score']})
            # Legacy quality heuristics remain separate from native_score.
            module.enrich_box_metrics(box,image,args)
            if 'all' not in disabled:
                if 'background' not in disabled:module.apply_rule1_background(box,args)
                if 'geometry' not in disabled:module.apply_rule2_geometry(box,*image.size,args)
                if 'defocus' not in disabled:module.apply_optional_defocus_filter(box,args)
        if 'all' not in disabled:
            if 'nested' not in disabled:module.apply_rule3_nested(working,args)
            if 'duplicates' not in disabled and args.enable_duplicate_suppression:module.apply_duplicate_overlap_suppression(working,args)
            if 'nms' not in disabled and args.enable_nms:module.apply_nms(working,args.nms_iou)
        for box in working:
            box['suppression_parent_id']=None  # legacy does not expose its exact suppression parent
            box['lineage_status']='legacy_reason_only' if not box['keep'] else 'retained'
            box['clean_config_hash']=digest(config)
        all_rows.extend(working)
    write_json(run.out/'cleaning_resolved.json',vars(args))
    audit=pd.DataFrame(all_rows) if all_rows else raw.assign(keep=pd.Series(dtype=bool))
    return audit[audit.keep.map(boolean)].copy(),audit


def acquire_candidates(cfg,frames,run):
    mode=cfg.get('detector_mode','cached')
    if mode=='cached':
        return validate_candidates(read_table(resolve(cfg,cfg.get('candidates'))))
    mod=load_legacy('test_locate_anything.py',cfg.get('upstream_root')) if mode in ('responses','live') else None
    if mode=='responses':
        responses=read_table(resolve(cfg,cfg.get('responses')))
        require_columns(responses,['recording_id','frame_idx','prompt_group','prompt','response'],'raw responses')
        rows=[]
        for _,x in responses.iterrows():
            f=frames[(frames.recording_id==x.recording_id)&(frames.frame_idx==x.frame_idx)]
            if len(f)!=1:raise ValueError('Response frame absent from frame manifest')
            with Image.open(resolve(cfg,f.iloc[0].image_path)) as im:size=im.size
            for j,b in enumerate(mod.LocateAnythingWorker.parse_boxes(x.response,*size)):
                rows.append({**b,'recording_id':x.recording_id,'frame_idx':int(x.frame_idx),'candidate_id':digest([x.to_dict(),j])[:24],
                             'prompt_group':x.prompt_group,'prompt':x.prompt,'native_score':None,'score_semantics':'none'})
        return validate_candidates(pd.DataFrame(rows,columns=BOX_COLUMNS+['prompt']))
    if mode!='live':raise ValueError(f'Unknown detector_mode {mode}')
    model=resolve(cfg,cfg.get('model_path'))
    if not model.is_dir() or not cfg.get('model_revision'):raise Blocked('Live VLM requires local model_path and immutable model_revision')
    prompts=cfg.get('prompts',{})
    if set(prompts)!={'cold','hot'} or not all(isinstance(v,list) and v for v in prompts.values()):
        raise Blocked('Configure exact nonempty cold/hot prompt lists')
    with run.time('model_load',unit='models',device=cfg.get('device','cpu')):
        worker=mod.LocateAnythingWorker(model_path=str(model),device=cfg.get('device','cpu'))
    rows=[];responses=[]
    for _,f in frames.iterrows():
        with Image.open(resolve(cfg,f.image_path)) as im:image=im.convert('RGB')
        for group,phrases in prompts.items():
            for phrase in phrases:
                with run.time('vlm_inference',unit='calls',device=cfg.get('device','cpu')):
                    answer=worker.ground_multi(image,phrase,**cfg.get('generation',{}))
                call_id=digest([f.recording_id,int(f.frame_idx),group,phrase,cfg['model_revision']])[:24]
                responses.append({'call_id':call_id,'recording_id':f.recording_id,'frame_idx':int(f.frame_idx),'prompt_group':group,'prompt':phrase,'response':answer})
                for j,b in enumerate(worker.parse_boxes(answer,*image.size)):
                    rows.append({**b,'recording_id':f.recording_id,'frame_idx':int(f.frame_idx),'candidate_id':f'{call_id}:{j}',
                                 'prompt_group':group,'prompt':phrase,'native_score':None,'score_semantics':'none'})
                write_csv(run.out/'raw_responses.csv',responses)
    return validate_candidates(pd.DataFrame(rows,columns=BOX_COLUMNS+['prompt']))


def run_detection(cfg,run):
    from .evaluation import detector_metrics,detector_strata
    frames=validate_frames(read_table(resolve(cfg,cfg.get('frames'))))
    with run.time('candidate_acquisition',len(frames),'frames',cache_mode='cached_candidates' if cfg.get('detector_mode','cached')=='cached' else 'cold'):
        raw=acquire_candidates(cfg,frames,run)
    write_csv(run.out/'raw_candidates.csv',raw)
    tables=[]
    available=set(cfg.get('available_prompt_groups',[])) or set(raw.prompt_group)
    selected=cfg.get('selected_method_ids')
    for group in ('cold','hot','both'):
        if selected is not None and not any(f'DET_{group.upper()}_{state}' in selected for state in ('raw','cleaned')):continue
        subset=raw if group=='both' else raw[raw.prompt_group==group]
        if (group!='both' and group not in available) or (group=='both' and not {'cold','hot'}<=available):
            run.record('DET_'+group.upper(),'blocked','No actual cached prompt group; cannot infer missing prompt calls')
            continue
        for state in ('raw','cleaned'):
            method='DET_'+group.upper()+'_'+state
            if selected is not None and method not in selected:continue
            try:
                with run.time(method,len(frames),'frames',cache_mode='cached_candidates'):
                    candidates,audit=clean_candidates(subset,frames,cfg.get('cleaning',{}),run,cfg) if state=='cleaned' else (subset,subset)
                out=run.out/method;write_csv(out/'candidates.csv',candidates);write_csv(out/'cleaning_audit.csv',audit)
                if cfg.get('reference_boxes'):
                    scope=frames[frames.annotated.map(boolean)] if 'annotated' in frames else frames.iloc[:0]
                    if scope.empty:raise Blocked('Explicit annotated=true frames required for independent detection scoring')
                    metrics,per_frame=detector_metrics(candidates,read_table(resolve(cfg,cfg['reference_boxes'])),scope,cfg['evaluation']['iou_threshold'])
                    tables.append({'method_id':method,**metrics});write_csv(out/'per_frame_metrics.csv',per_frame)
                    strata,members=detector_strata(candidates,read_table(resolve(cfg,cfg['reference_boxes'])),scope,cfg['evaluation']['iou_threshold'])
                    write_csv(out/'detector_by_state_size.csv',strata);write_csv(out/'reference_detection_members.csv',members)
                run.record(method,'completed' if cfg.get('reference_boxes') else 'prediction_only','Independent frame boxes not supplied' if not cfg.get('reference_boxes') else None)
                if cfg.get('downstream'):run_frozen_downstream(cfg,run,method,out/'candidates.csv')
            except Blocked as e:run.record(method,'blocked',str(e))
    for method,settings in cfg.get('classical',{}).items():
        if selected is not None and 'DET_'+method not in selected:continue
        try:
            rows=[]
            with run.time('DET_'+method,len(frames),'frames'):
                from concurrent.futures import ThreadPoolExecutor
                def detect_frame(f):
                    with Image.open(resolve(cfg,f.image_path)) as image:
                        return f,classical_candidates(image,settings,method)
                workers=int(cfg.get('classical_workers',1))
                if workers<1:raise ValueError('classical_workers must be positive')
                with ThreadPoolExecutor(max_workers=workers) as executor:
                  for number,(f,boxes) in enumerate(executor.map(detect_frame,frames.itertuples(index=False)),1):
                    for j,b in enumerate(boxes):rows.append({**b,'recording_id':f.recording_id,'frame_idx':int(f.frame_idx),'candidate_id':f'{method}:{f.recording_id}:{f.frame_idx}:{j}','prompt_group':'classical'})
                    if number%10==0 or number==len(frames):
                        print(f'CLASSICAL_PROGRESS DET_{method}: {number}/{len(frames)} frames; {len(rows)} candidates',flush=True)
            cand=pd.DataFrame(rows,columns=BOX_COLUMNS+['sigma'])
            write_csv(run.out/('DET_'+method)/'candidates.csv',cand)
            if cfg.get('reference_boxes'):
                scope=frames[frames.annotated.map(boolean)] if 'annotated' in frames else frames.iloc[:0]
                if scope.empty:raise Blocked('No independently annotated frame scope')
                metrics,per=detector_metrics(cand,read_table(resolve(cfg,cfg['reference_boxes'])),scope,cfg['evaluation']['iou_threshold'])
                tables.append({'method_id':'DET_'+method,**metrics});write_csv(run.out/('DET_'+method)/'per_frame_metrics.csv',per)
                strata,members=detector_strata(cand,read_table(resolve(cfg,cfg['reference_boxes'])),scope,cfg['evaluation']['iou_threshold'])
                write_csv(run.out/('DET_'+method)/'detector_by_state_size.csv',strata);write_csv(run.out/('DET_'+method)/'reference_detection_members.csv',members)
            run.record('DET_'+method,'completed' if cfg.get('reference_boxes') else 'prediction_only')
            if cfg.get('downstream'):run_frozen_downstream(cfg,run,'DET_'+method,run.out/('DET_'+method)/'candidates.csv')
        except Blocked as e:run.record('DET_'+method,'blocked',str(e))
    write_csv(run.out/'S03_detector_comparison.csv',tables)


def run_frozen_downstream(cfg,parent,method,candidates_path):
    """Same declared tracker/restoration/judge config for every detector pool; independent event GT."""
    from .common import Run
    from .experiments import run_tracking
    local=copy.deepcopy(cfg);local.pop('downstream',None);local.update(copy.deepcopy(cfg['downstream']))
    local['candidates']=str(candidates_path.resolve());local['output_dir']=str(parent.out/'downstream')
    local['variants']=[{'id':method+'_EVENT','backend':cfg['downstream'].get('backend','ot'),
                       'rescue':cfg['downstream'].get('rescue',True),'restoration':cfg['downstream'].get('restoration',{'gap':True,'prebirth':True})}]
    child=Run(local,'assignment')
    try:
        run_tracking(local,child);child.finish()
    except Exception as e:
        child.finish(e);raise
    parent.record(method+'_EVENT',child.meta['status'],str(child.out))
    table=child.out/'S04_assignment_and_mot.csv'
    if table.exists():
        try:data=pd.read_csv(table)
        except pd.errors.EmptyDataError:return
        destination=parent.out/'detector_to_event.csv'
        if destination.exists():data=pd.concat([pd.read_csv(destination),data],ignore_index=True)
        write_csv(destination,data)
