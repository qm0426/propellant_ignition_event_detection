"""Resumable image -> LocateAnything -> original cleaning -> tracker -> judges.

GT is optional and is only read by explicit evaluation after prediction.
The detector may run in its own existing conda environment via subprocess.
"""
from pathlib import Path
from contextlib import contextmanager
import argparse,copy,importlib.util,json,re,subprocess,sys,time,traceback
import numpy as np
import pandas as pd
from PIL import Image
from .common import digest,file_hash,load_config,write_json,write_csv,resolve

STAGES=['detect','clean','track','judge']

class PipelineRun:
    def __init__(self,out):self.out=Path(out);self.inputs={};self.statuses=[];self.timings=[]
    @contextmanager
    def time(self,stage,units=1,unit='tracks',**kwargs):
        start=time.perf_counter()
        try:yield
        finally:self.timings.append(dict(stage=stage,total_s=time.perf_counter()-start,N_units=units,unit=unit))
    def record(self,method,status,reason=None):self.statuses.append(dict(method_id=method,status=status,reason=reason))

def frame_manifest(cfg):
    root=resolve(cfg,cfg['image_dir']);rx=re.compile(cfg.get('frame_regex',r'_(?P<frame>\d+)$'))
    if 'frame' not in rx.groupindex:raise ValueError('frame_regex needs named group frame')
    extensions=cfg.get('image_extensions',['.png','.jpg','.jpeg','.tif','.tiff','.bmp'])
    if isinstance(extensions,str):extensions=extensions.split(',')
    extensions={'.'+s.strip().lower().lstrip('.') for s in extensions if s.strip()}
    supported={'.png','.jpg','.jpeg','.tif','.tiff','.bmp'}
    if not extensions or extensions-supported:raise ValueError('image_extensions must contain supported image suffixes')
    rows=[]
    selected={}
    for path in root.iterdir():
        if not path.is_file() or path.suffix.lower() not in extensions:continue
        m=rx.search(path.stem)
        if not m:raise ValueError('Image name does not match frame_regex: '+path.name)
        frame=int(m['frame'])+int(cfg.get('frame_offset',0))
        if frame<cfg.get('start_frame',0) or (cfg.get('end_frame') is not None and frame>cfg['end_frame']):continue
        if frame<0:raise ValueError(f'Negative physical frame {frame}: {path.name}; check frame_offset/start_frame')
        if frame in selected:
            raise ValueError(f'Duplicate physical frame {frame}: {selected[frame].name} and {path.name}. '
                             'Set image_extensions to select the original image format; no files were deleted.')
        selected[frame]=path
    for frame,path in sorted(selected.items()):
        with Image.open(path) as im:
            if im.mode not in ('RGB','RGBA','L'):raise ValueError('Supply explicit 8-bit conversion for '+str(path))
            width,height=im.size
        rows.append(dict(recording_id=cfg['recording_id'],frame_idx=frame,image_path=str(path.resolve()),width=width,height=height,image_sha256=file_hash(path)))
    df=pd.DataFrame(rows)
    if df.empty:raise ValueError('No selected images')
    df=df.sort_values('frame_idx').reset_index(drop=True)
    if df.frame_idx.duplicated().any() or df.frame_idx.min()<0:raise ValueError('Duplicate or negative physical frames')
    if cfg.get('require_contiguous_frames',True) and np.any(np.diff(df.frame_idx)!=1):raise ValueError('Missing images in selected physical frame range')
    if len(df[['width','height']].drop_duplicates())!=1:raise ValueError('A recording must have fixed dimensions')
    return df

def detector_worker(job):
    """Only this subprocess imports transformers/torch and the user's detector."""
    import torch
    spec=importlib.util.spec_from_file_location('_pressure_locate',job['detector_script'])
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    settings=job['detector'];frames=pd.read_csv(job['frames']);out=Path(job['directory']);out.mkdir(parents=True,exist_ok=True)
    worker=None
    for i,f in enumerate(frames.itertuples(),1):
        target=out/(Path(f.image_path).stem+'.json')
        if target.exists():
            data=json.loads(target.read_text(encoding='utf-8'))
            if data['image_sha256']!=f.image_sha256 or data['job_key']!=job['key']:raise ValueError('Per-frame detector checkpoint mismatch')
            continue
        if worker is None:worker=mod.LocateAnythingWorker(model_path=settings['model'],device=settings.get('device','cuda'),dtype=torch.bfloat16)
        seed=int(settings.get('seed',2026))+int(f.frame_idx);torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        prompts=settings.get('prompts') or mod.DEFAULT_TWO_STATE_PROMPTS
        with Image.open(f.image_path) as im:
            boxes=mod.detect_boxes_for_image(worker,im.convert('RGB'),two_state=True,class_prompts=prompts,
                single_prompt='',gen_kwargs=settings.get('generation',{}),nms_iou=settings.get('nms_iou',.5))
        write_json(target,{'image_sha256':f.image_sha256,'job_key':job['key'],'seed':seed,'boxes':boxes})
        mod.save_boxes_txt(boxes,str(target.with_suffix('.txt')))
        print(f'DETECT {i}/{len(frames)} frame={f.frame_idx} boxes={len(boxes)}',flush=True)

def acquire(cfg,frames,out,key):
    from .labelme_review import read_pixel_boxes
    from .detection import BOX_COLUMNS,validate_candidates
    det=cfg.get('detector',{});mode=det.get('mode','live');rows=[];invalid_rows=[]
    if mode=='live':
        script=resolve(cfg,det['script']);local_model=Path(det['model'])
        if not local_model.is_dir():raise ValueError('Use an existing local model snapshot directory; no implicit download')
        job={'detector_script':str(script),'detector':det,'frames':str(out/'frames.csv'),'directory':str(out/'01_detection/per_frame'),'key':key}
        jobfile=out/'01_detection/worker_job.json';write_json(jobfile,job)
        command=[det.get('python',sys.executable),str(Path(__file__).resolve().parents[1]/'run_pressure_pipeline.py'),'--worker',str(jobfile)]
        subprocess.run(command,check=True,cwd=cfg['project_root'])
    elif mode!='cached_txt':raise ValueError('detector.mode must be live or cached_txt')
    for f in frames.itertuples():
        if mode=='live':
            source=out/'01_detection/per_frame'/(Path(f.image_path).stem+'.json')
            boxes=json.loads(source.read_text(encoding='utf-8'))['boxes']
        else:
            source=resolve(cfg,det['txt_dir'])/(Path(f.image_path).stem+'.txt')
            if not source.is_file():raise FileNotFoundError('Missing detection file (an empty file is valid): '+str(source))
            boxes,invalid=read_pixel_boxes(source)
            invalid_rows.extend({'frame_idx':int(f.frame_idx),'source':str(source),**b} for b in invalid)
        for j,b in enumerate(boxes):
            coords=np.array([b[k] for k in ('x1','y1','x2','y2')],float)
            if not np.isfinite(coords).all() or coords[2]<=coords[0] or coords[3]<=coords[1]:
                invalid_rows.append({'frame_idx':int(f.frame_idx),'source':str(source),**b});continue
            label=b.get('label',b.get('source_label','unknown'))
            rows.append({**b,'recording_id':f.recording_id,'frame_idx':int(f.frame_idx),'candidate_id':f'{f.recording_id}:{f.frame_idx}:{j}',
                'prompt_group':'cold' if 'unburned' in label else 'hot' if 'burning' in label else 'unknown',
                'native_score':None,'score_semantics':'none'})
    write_json(out/'01_detection/invalid_boxes.json',invalid_rows)
    return validate_candidates(pd.DataFrame(rows,columns=list(dict.fromkeys(BOX_COLUMNS+['label','prompt','source_label','line']))))

def run_pipeline(cfg,start='detect',stop='judge',resume=False,dry_run=False):
    if cfg.get('methods')!={'GMM_RTV':{'method':'gmm','rtv':True,'candidate_policy':'first_terminal'}}:
        raise ValueError('This entry point runs only the original GMM_RTV method; use run_hmm_comparison.py for experiments')
    if cfg.get('tracker',{}).get('backend','ot')!='ot':raise ValueError('The original-method entry point requires OT association')
    cfg=copy.deepcopy(cfg);out=resolve(cfg,cfg['output_dir']);frames=frame_manifest(cfg)
    legacy=resolve(cfg,cfg['legacy_root']);upstream=resolve(cfg,cfg['upstream_root'])
    sources=[upstream/'test_locate_anything.py',upstream/'postprocess_detection_txt_modified.py',legacy/'ot_kalman_tracker.py',legacy/'ignition_detection_gmm_rtv_new.py']
    hashes={str(p):file_hash(p) for p in sources}
    hashes.update({str(p):file_hash(p) for p in Path(__file__).parent.glob('*.py')})
    external={}
    if cfg.get('detector',{}).get('mode')=='cached_txt':
        for f in frames.itertuples():
            p=resolve(cfg,cfg['detector']['txt_dir'])/(Path(f.image_path).stem+'.txt');external[str(p)]=file_hash(p)
    else:
        model=Path(cfg['detector']['model'])
        if not model.is_dir():raise ValueError('Configure existing local detector.model snapshot')
        external={str(p):file_hash(p) for p in sorted(model.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    key=digest([cfg,frames.to_dict('records'),hashes,external])
    if STAGES.index(start)>STAGES.index(stop):raise ValueError('start-from must precede stop-after')
    if dry_run:
        print(json.dumps({'frames':len(frames),'recording_id':cfg['recording_id'],'stages':STAGES[STAGES.index(start):STAGES.index(stop)+1],
            'output':str(out),'sources':hashes,'pipeline_key':key,'GT_required':False},ensure_ascii=False,indent=2));return
    out.mkdir(parents=True,exist_ok=True);lock=out/'pipeline_config.json'
    if lock.exists():
        if json.loads(lock.read_text(encoding='utf-8'))['key']!=key:raise ValueError('Inputs/config/code changed; choose a new output_dir')
        if not resume:raise FileExistsError('Output exists; pass --resume or use a new output_dir')
    else:
        write_json(lock,{'key':key,'config':cfg,'source_sha256':hashes,'external_sha256':external})
        import shutil
        for i,p in enumerate(sources):
            dest=out/'source_snapshot'/f'{i}_{p.name}';dest.parent.mkdir(exist_ok=True);shutil.copy2(p,dest)
        write_csv(out/'frames.csv',frames)
    statefile=out/'pipeline_status.json';state=json.loads(statefile.read_text(encoding='utf-8')) if statefile.exists() else {}
    run=PipelineRun(out);cfg['virtual_tracking_rois']=cfg.get('virtual_tracking_rois',True)
    cfg.setdefault('features',{'difference_policy':'contiguous_zero_boundary','background_correction':True})
    destinations={'detect':out/'01_detection/raw_candidates.csv','clean':out/'02_cleaning/candidates.csv','track':out/'03_tracking/roi_manifest.csv','judge':out/'04_judgment/events.csv'}
    for stage in STAGES[:STAGES.index(stop)+1]:
        entry=state.get(stage,{})
        if entry.get('status')=='completed':
            if not all((out/p).is_file() and file_hash(out/p)==h for p,h in entry['outputs'].items()):raise ValueError('Stage output changed: '+stage)
            print('VERIFIED_SKIP',stage,flush=True);continue
        if STAGES.index(stage)<STAGES.index(start):raise ValueError('Required earlier stage is not complete: '+stage)
        directory=destinations[stage].parent;directory.mkdir(parents=True,exist_ok=True);run.out=directory
        try:
            print('START',stage,flush=True)
            with run.time(stage,len(frames),'frames'):
                if stage=='detect':write_csv(destinations[stage],acquire(cfg,frames,out,key))
                elif stage=='clean':
                    from .detection import clean_candidates
                    raw=pd.read_csv(destinations['detect'],low_memory=False)
                    cleaned,audit=clean_candidates(raw,frames,cfg.get('cleaning',{}),run,cfg)
                    write_csv(destinations[stage],cleaned);write_csv(directory/'cleaning_audit.csv',audit)
                elif stage=='track':
                    from .tracking import run_custom,crop_observations
                    from .experiments import export_mot
                    variant={**cfg.get('tracker',{}),'id':'PRESSURE_TRACKS'}
                    if variant.get('backend','ot') not in ('ot','hungarian'):raise ValueError('Choose ot or hungarian')
                    obs,trace=run_custom(pd.read_csv(destinations['clean'],low_memory=False),frames,cfg,variant,run)
                    write_csv(directory/'track_observations.csv',obs);write_csv(directory/'association_trace.csv',trace);export_mot(obs,directory)
                    rois=crop_observations(obs,frames,cfg,directory);write_csv(destinations[stage],rois)
                    compat=obs.rename(columns={'frame_idx':'frame_id'}).copy()
                    lookup=dict(zip(frames.frame_idx,frames.image_path))
                    compat['image_name']=[Path(lookup[f]).stem for f in compat.frame_id]
                    write_csv(out/'tracks.csv',compat)
                else:
                    from .judges import extract_track_features
                    from .experiments import predict_feature_bundle,evaluate_tracking_observations
                    rois=pd.read_csv(destinations['track'],low_memory=False)
                    if rois.empty:
                        write_csv(destinations[stage],[],['recording_id','track_id','method_id','has_ignition','onset_frame','confirmation_frame'])
                    else:
                        bundle,base,mod=extract_track_features(rois,cfg,run)
                        predictions=predict_feature_bundle(bundle,mod,base,cfg['methods'],cfg,run)
                        if set(predictions)!=set(cfg['methods']):raise RuntimeError('Some judges failed; inspect method error.json')
                        write_csv(destinations[stage],pd.concat(predictions.values(),ignore_index=True))
                        # Both reference files are necessary. Prediction works without any GT.
                        if cfg.get('reference_boxes') or cfg.get('reference_events'):
                            if not (cfg.get('reference_boxes') and cfg.get('reference_events')):raise ValueError('Evaluation requires both reference_boxes and reference_events')
                            obs=pd.read_csv(out/'03_tracking/track_observations.csv',low_memory=False)
                            metrics=[]
                            for method,pred in predictions.items():
                                target=directory/method;target.mkdir(exist_ok=True)
                                score,_,reason=evaluate_tracking_observations(obs,pred,{**cfg,'frames':str(out/'frames.csv')},run,target)
                                if reason:raise ValueError(reason)
                                metrics.append({'method_id':method,**score})
                            write_csv(directory/'event_metrics.csv',metrics)
            outputs={str(p.relative_to(out)):file_hash(p) for p in directory.rglob('*') if p.is_file()}
            if stage=='track':outputs['tracks.csv']=file_hash(out/'tracks.csv')
            state[stage]={'status':'completed','outputs':outputs};write_json(statefile,state)
        except Exception as e:
            state[stage]={'status':'failed','error':str(e)};write_json(statefile,state);raise
        finally:
            previous=pd.read_csv(out/'runtime_summary.csv').to_dict('records') if (out/'runtime_summary.csv').exists() else []
            write_csv(out/'runtime_summary.csv',previous+run.timings);run.timings=[]
    print('OUTPUT',out,flush=True)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config');p.add_argument('--worker')
    p.add_argument('--start-from',choices=STAGES,default='detect');p.add_argument('--stop-after',choices=STAGES,default='judge')
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args(argv)
    if a.worker:detector_worker(json.loads(Path(a.worker).read_text(encoding='utf-8')));return 0
    if not a.config:p.error('--config is required')
    run_pipeline(load_config(a.config),a.start_from,a.stop_after,a.resume,a.dry_run);return 0
