from __future__ import annotations
import argparse
import ast
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'review_v2_inputs/historical_implicit_negatives'
DEFAULT_OUT=ROOT/'revision_outputs/prompt_sensitivity_20261030'
GROUPS={'cold':'unburned_aluminum_droplet','hot':'burning_aluminum_droplet'}


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)
def csv(path,data,columns=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    frame=data if isinstance(data,pd.DataFrame) else pd.DataFrame(data,columns=columns)
    frame.to_csv(path,index=False,encoding='utf-8-sig')


def current_prompts(source):
    tree=ast.parse(Path(source).read_text(encoding='utf-8-sig'))
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEFAULT_TWO_STATE_PROMPTS' for t in node.targets):
            d=ast.literal_eval(node.value)
            return {group:d[label] for group,label in GROUPS.items()}
    raise ValueError('No DEFAULT_TWO_STATE_PROMPTS in source')


def default_sets(source):
    return {
        'A':{'description':'Current script default wording, newly rerun baseline; historical cache wording not inferred',
             'prompts':current_prompts(source)},
        'B':{'description':'Attribute-preserving synonym/order paraphrase 1', 'prompts':{
            'cold':['very small dark circular particles','small circular black droplets',
                    'small round gray particles without a glow','separate compact dark specks',
                    'small spherical particles that do not glow'],
            'hot':['compact circular droplets with an orange glow','small separate particles emitting a bright glow',
                   'small luminous bright spherical droplets','very small faint dots glowing orange',
                   'small orange particles that have just ignited','small circular incandescent specks',
                   'small faintly glowing orange dots']}},
        'C':{'description':'Attribute-preserving synonym/order paraphrase 2', 'prompts':{
            'cold':['tiny round particles with a dark appearance','small droplets with a black circular appearance',
                    'small round gray particles with no visible glow','compact dark specks separated from others',
                    'small particles with a spherical shape and no glow'],
            'hot':['compact round droplets emitting orange light','small isolated particles with a bright glow',
                   'small spherical droplets that glow brightly','tiny orange dots with a faint glow',
                   'small newly ignited particles with an orange appearance','small incandescent specks with a round shape',
                   'small orange dots with a dim glow']}}
    }


def validate_sets(sets):
    if 'A' not in sets or len(sets)<2:raise ValueError('At least A and one alternate prompt set required')
    budget=None
    for name,spec in sets.items():
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):raise ValueError('Unsafe prompt set name')
        prompts=spec['prompts']
        if set(prompts)!=set(GROUPS):raise ValueError('Every set needs cold and hot lists')
        counts=[]
        for group in GROUPS:
            values=prompts[group]
            if not isinstance(values,list) or not values or any(not isinstance(v,str) or not v.strip() for v in values):raise ValueError('Nonempty prompt strings required')
            if len(set(values))!=len(values):raise ValueError('Duplicate phrase within one pool')
            counts.append(len(values))
        if budget is None:budget=counts
        if counts!=budget:raise ValueError('Prompt counts must match across sets, separately for cold and hot')


def prepare(output,inputs=DATA,seeds=(2026,2027,2028),limit=None,prompt_file=None):
    output=Path(output).resolve();inputs=Path(inputs).resolve()
    if output.exists():raise FileExistsError('Use a new output root; prepare never overwrites a frozen protocol')
    if not seeds or len(set(seeds))!=len(seeds) or any(type(s) is not int or not 0<=s<2**32 for s in seeds):raise ValueError('Distinct integer seeds in [0,2**32) required')
    frames=pd.read_csv(inputs/'detector_frames.csv',low_memory=False).sort_values(['recording_id','frame_idx']).reset_index(drop=True)
    gt=pd.read_csv(inputs/'detector_boxes.csv',low_memory=False)
    from ignition_revision.detection import validate_frames
    validate_frames(frames)
    if frames.empty:raise ValueError('No frames')
    if limit is not None and limit<1:raise ValueError('Positive limit required')
    if 'annotated' in frames:
        from ignition_revision.common import boolean
        if not frames.annotated.map(boolean).all():raise ValueError('All selected frames must have explicit annotation')
    source_frames=len(frames)
    # A pilot samples evenly through the recording rather than choosing only the
    # first few frames. Its scope remains explicitly distinct from the full run.
    if limit and limit<len(frames):frames=frames.iloc[np.linspace(0,len(frames)-1,limit).round().astype(int)].reset_index(drop=True)
    gt=gt.merge(frames[['recording_id','frame_idx']],on=['recording_id','frame_idx'],how='inner',validate='many_to_one')
    if gt.duplicated(['recording_id','frame_idx','ref_object_id']).any():raise ValueError('Duplicate reference box IDs')
    source=ROOT/'test_locate_anything.py'
    sets=read(prompt_file) if prompt_file else default_sets(source);validate_sets(sets)
    for _,r in frames.iterrows():
        path=Path(r.image_path)
        if not path.is_absolute():path=ROOT/path
        if not path.is_file():raise FileNotFoundError(path)
        frames.loc[_, 'image_path']=str(path.resolve())
    output.mkdir(parents=True);snapshot=output/'source_snapshot';snapshot.mkdir()
    for name in ['test_locate_anything.py','postprocess_detection_txt_modified.py']:shutil.copy2(ROOT/name,snapshot/name)
    csv(output/'frames.csv',frames);csv(output/'boxes.csv',gt);write(output/'prompt_sets.json',sets)
    cleaning_cfg=ROOT/'revision_outputs/review_v2_full/resolved_cases/DET_BOTH_CLEANED.json'
    cleaning=read(cleaning_cfg).get('cleaning',{}) if cleaning_cfg.exists() else {'disabled_rules':[]}
    per_frame=sum(len(x) for x in sets['A']['prompts'].values())
    dependencies=[Path(__file__),ROOT/'ignition_revision/detection.py',ROOT/'ignition_revision/evaluation.py',ROOT/'ignition_revision/common.py']
    protocol={'version':1,'created_at':datetime.now(timezone.utc).isoformat(),'input_root':str(inputs),
        'frames':len(frames),'reference_boxes':len(gt),'source_frame_count':source_frames,
        'scope':'full_300_frame_detection_cohort' if len(frames)==300 and source_frames==300 and len(gt)==4273 else 'pilot_or_custom_cohort',
        'seeds':list(seeds),'sets':list(sets),'phrases_per_frame':per_frame,
        'expected_calls':len(frames)*len(sets)*len(seeds)*per_frame,'iou_threshold':.5,
        'cleaning':cleaning,'baseline':'A newly rerun with current script prompts; not assumed to reproduce historical prompts',
        'generation':{'generation_mode':'hybrid','max_new_tokens':2048,'temperature':.2,'verbose':False},
        'generation_worker_fixed_settings':{'do_sample':True,'top_p':.9,'repetition_penalty':1.1},
        'paired_seed_policy':'SHA256(base_seed,recording_id,physical_frame,prompt_group,slot), independent of set wording',
        'preclean_merge':'Concatenate per-phrase outputs without early class-aware NMS, identically for all sets; new baseline, not historical-cache reproduction',
        'primary_condition':'both_cleaned','conditions':['cold_raw','hot_raw','both_raw','both_cleaned'],
        'invalid_boxes':'raw: count malformed/degenerate boxes as FP; cleaned: rejected before cleaning, audited',
        'frozen_files':{str(p.resolve()):sha(p) for p in [output/'frames.csv',output/'boxes.csv',output/'prompt_sets.json',*snapshot.iterdir(),*dependencies]},
        'source_hashes':{str(inputs/n):sha(inputs/n) for n in ['detector_frames.csv','detector_boxes.csv']},
        'image_hashes':{str(Path(v)):sha(v) for v in frames.image_path},
        'parameter_design':'retrospective_prespecified_wording_diagnostic_not_independent_tuning'}
    write(output/'protocol.json',protocol);(output/'protocol.sha256').write_text(sha(output/'protocol.json'),encoding='ascii')
    write(output/'inference_status.json',{'status':'prepared','completed_calls':0,'expected_calls':protocol['expected_calls']})
    print(f'PROMPT_PROTOCOL {output}\n{len(frames)} frames / {len(gt)} GT boxes / {len(sets)} sets / {len(seeds)} seeds / {protocol["expected_calls"]} model calls',flush=True)
    return output


def load_protocol(root,check_images=True):
    root=Path(root).resolve()
    if sha(root/'protocol.json')!=(root/'protocol.sha256').read_text(encoding='ascii').strip():raise ValueError('Protocol changed; prepare a new root')
    p=read(root/'protocol.json')
    for name,value in p['frozen_files'].items():
        if not Path(name).is_file() or sha(name)!=value:raise ValueError('Frozen input/code changed: '+name)
    for name,value in p['source_hashes'].items():
        if not Path(name).is_file() or sha(name)!=value:raise ValueError('Original detector input changed: '+name)
    if check_images:
        for name,value in p['image_hashes'].items():
            if sha(name)!=value:raise ValueError('Image content changed: '+name)
    return root,p,pd.read_csv(root/'frames.csv',low_memory=False),pd.read_csv(root/'boxes.csv',low_memory=False),read(root/'prompt_sets.json')


def paired_seed(base,rec,frame,group,slot):return int(digest([int(base),str(rec),int(frame),group,int(slot)])[:8],16)
def schedule(frames,sets,seeds,selected=None):
    for name,spec in sets.items():
        if selected and name not in selected:continue
        for seed in seeds:
            for f in frames.itertuples(index=False):
                for group in GROUPS:
                    for slot,prompt in enumerate(spec['prompts'][group]):
                        key={'set':name,'seed':int(seed),'recording_id':str(f.recording_id),'frame_idx':int(f.frame_idx),'group':group,'slot':slot}
                        yield key,str(f.image_path),prompt
def call_path(root,key):return root/'calls'/key['set']/str(key['seed'])/(digest(key)[:24]+'.json')
def sealed_write(path,payload):write(path,{'payload':payload,'sha256':digest(payload)})
def sealed_read(path,protocol_hash,key,model_hash):
    envelope=read(path);d=envelope['payload']
    if digest(d)!=envelope['sha256'] or d['protocol_hash']!=protocol_hash or d['key']!=key or d['model_hash']!=model_hash:raise ValueError('Invalid/changed inference checkpoint: '+str(path))
    return d


def locate_model(value):
    path=Path(value).expanduser()
    if path.is_dir():return path.resolve()
    from huggingface_hub import snapshot_download
    try:return Path(snapshot_download(repo_id=value,local_files_only=True)).resolve()
    except Exception as ex:raise ValueError('No local model cache. Supply --model-path with the existing LocateAnything snapshot directory; no automatic download was attempted.') from ex


def model_fingerprint(path):
    files=[p for p in path.rglob('*') if p.is_file() and p.suffix not in ('.pyc','.pyo') and
           '__pycache__' not in p.relative_to(path).parts and not any(x.startswith('.') for x in p.relative_to(path).parts)]
    if not files:raise ValueError('Empty model directory')
    return {'resolved_model_path':str(path),'files':{str(f.relative_to(path)):sha(f) for f in sorted(files)}}


def doctor(model='nvidia/LocateAnything-3B',device='cuda'):
    import importlib.metadata,importlib.util,sys
    packages=['torch','transformers','huggingface_hub','numpy','pandas','scipy','cv2','PIL']
    found={name:importlib.util.find_spec(name) is not None for name in packages}
    if not all(found.values()):
        raise ValueError('Missing packages in '+sys.executable+': '+','.join(k for k,v in found.items() if not v))
    import torch
    from ignition_revision.common import load_legacy
    load_legacy('test_locate_anything.py',ROOT)  # Import only; no model loaded.
    path=locate_model(model)
    weights=list(path.glob('*.safetensors'))+list(path.glob('pytorch_model*.bin'))
    ready=bool(weights) and (torch.cuda.is_available() if device.startswith('cuda') else True)
    result={'python':sys.executable,'torch':torch.__version__,'transformers':importlib.metadata.version('transformers'),
        'cuda_available':torch.cuda.is_available(),'requested_device':device,'local_model':str(path),
        'weight_files':len(weights),'ready_without_loading_weights':ready,
        'note':'Checks local imports/cache only; does not execute model loading or generation.'}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not ready:raise ValueError('Environment/model not ready for requested device; choose a CUDA-enabled Python and complete local model cache')
    return result


def seed_all(seed,torch):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends,'cudnn'):torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def infer_loop(root,p,frames,sets,worker,model_meta,selected=None,seed_fn=None,sync=None,max_calls=None):
    protocol_hash=sha(root/'protocol.json');model_hash=digest(model_meta)
    if (root/'model_provenance.json').exists():
        if read(root/'model_provenance.json')!=model_meta:raise ValueError('Cannot mix model/device/runtime provenance in one root')
    else:write(root/'model_provenance.json',model_meta)
    done=0;new=0;began=time.perf_counter();last_image_path=None;image=None
    selected_expected=sum(1 for _ in schedule(frames,sets,p['seeds'],selected))
    try:
        for key,path,prompt in schedule(frames,sets,p['seeds'],selected):
            dest=call_path(root,key)
            if dest.exists():sealed_read(dest,protocol_hash,key,model_hash);done+=1;continue
            if max_calls is not None and new>=max_calls:break
            if image is None or last_image_path!=path:
                with Image.open(path) as im:image=im.convert('RGB')
                last_image_path=path
            seed=paired_seed(key['seed'],key['recording_id'],key['frame_idx'],key['group'],key['slot'])
            if seed_fn:seed_fn(seed)
            if sync:sync()
            start=time.perf_counter()
            answer=worker.ground_multi(image,prompt,**p['generation'])
            if sync:sync()
            elapsed=time.perf_counter()-start
            if not isinstance(answer,str):raise TypeError('Expected text from LocateAnything, got '+type(answer).__name__)
            boxes=worker.parse_boxes(answer,*image.size)
            payload={'protocol_hash':protocol_hash,'model_hash':model_hash,'key':key,'paired_call_seed':seed,
                'prompt':prompt,'full_question':f'Locate all the instances that match the following description: {prompt}.',
                'response':answer,'boxes':boxes,'image_size':list(image.size),'elapsed_s':elapsed}
            sealed_write(dest,payload);done+=1;new+=1
            if new%10==0 or done==selected_expected:
                status={'status':'running','completed_selected_calls':done,'selected_expected_calls':selected_expected,
                        'expected_full_calls':p['expected_calls'],'new_calls_this_invocation':new,'wall_s':time.perf_counter()-began,'last_call':key}
                write(root/'inference_status.json',status)
                print(f'PROMPT_INFERENCE {done}/{selected_expected} selected calls; new={new}; {key["set"]} seed={key["seed"]} frame={key["frame_idx"]} {key["group"]}/{key["slot"]}',flush=True)
    except BaseException as ex:
        write(root/'inference_status.json',{'status':'interrupted_or_failed','completed_selected_calls':done,
            'expected_full_calls':p['expected_calls'],'reason':str(ex),'completed_checkpoints_preserved':True})
        raise
    status={'status':'selected_calls_complete' if done==selected_expected else 'budget_paused',
        'completed_selected_calls':done,'selected_expected_calls':selected_expected,'expected_full_calls':p['expected_calls'],
        'new_calls_this_invocation':new,'wall_s':time.perf_counter()-began}
    write(root/'inference_status.json',status);print(json.dumps(status),flush=True)
    return status


def infer(root,model='nvidia/LocateAnything-3B',device='cuda',selected=None,max_calls=None,dtype='auto'):
    root,p,frames,gt,sets=load_protocol(root)
    if selected and not set(selected)<=set(sets):raise ValueError('Unknown selected set')
    import torch,importlib.metadata
    from ignition_revision.common import load_legacy
    if device.startswith('cuda') and not torch.cuda.is_available():raise ValueError('CUDA is unavailable in this Python environment; select a CUDA-enabled environment or explicitly use --device cpu')
    if dtype=='auto':dtype=('bfloat16' if torch.cuda.is_bf16_supported() else 'float16') if device.startswith('cuda') else 'float32'
    actual_dtype={'bfloat16':torch.bfloat16,'float16':torch.float16,'float32':torch.float32}[dtype]
    path=locate_model(model);print('Hashing local model files: '+str(path),flush=True)
    meta={**model_fingerprint(path),'device':device,'dtype':dtype,
        'versions':{k:importlib.metadata.version(k) for k in ['torch','transformers','numpy','Pillow']}}
    if (root/'model_provenance.json').exists() and read(root/'model_provenance.json')!=meta:raise ValueError('Model/device/runtime changed; use a new root')
    module=load_legacy('test_locate_anything.py',root/'source_snapshot')
    worker=module.LocateAnythingWorker(model_path=str(path),device=device,dtype=actual_dtype)
    worker.model.eval();torch.use_deterministic_algorithms(True,warn_only=True)
    sync=(lambda:torch.cuda.synchronize(device)) if device.startswith('cuda') else None
    return infer_loop(root,p,frames,sets,worker,meta,selected,lambda seed:seed_all(seed,torch),sync,max_calls)


def raw_candidates(calls):
    from ignition_revision.detection import BOX_COLUMNS
    rows=[];invalid=[];responses=[]
    for d in calls:
        key=d['key'];w,h=d['image_size'];responses.append({**key,'prompt':d['prompt'],'full_question':d['full_question'],
            'paired_call_seed':d['paired_call_seed'],'response':d['response'],'elapsed_s':d['elapsed_s']})
        for j,b in enumerate(d['boxes']):
            try:coords=np.asarray([b[k] for k in ['x1','y1','x2','y2']],float)
            except (KeyError,ValueError,TypeError):coords=np.full(4,np.nan)
            clipped=np.maximum([0,0,0,0],np.minimum([w,h,w,h],coords))
            if not np.isfinite(coords).all() or clipped[2]<=clipped[0] or clipped[3]<=clipped[1]:
                invalid.append({**key,'box_index':j,'reason':'nonfinite_or_degenerate_box'});continue
            rows.append({'recording_id':key['recording_id'],'frame_idx':key['frame_idx'],
                'candidate_id':digest([key,j])[:28],**dict(zip(['x1','y1','x2','y2'],clipped)),
                'prompt_group':key['group'],'label':GROUPS[key['group']],'prompt':d['prompt'],
                'phrase_slot':key['slot'],'native_score':None,'score_semantics':'none',
                'coordinates_clipped':bool(np.any(clipped!=coords))})
        malformed=max(0,d['response'].count('<box>')-len(d['boxes']))
        for j in range(malformed):invalid.append({**key,'box_index':j,'reason':'unparsed_box_tag'})
    return pd.DataFrame(rows,columns=BOX_COLUMNS+['label','prompt','phrase_slot','coordinates_clipped']),invalid,responses


def reference_members(candidates,gt,frames,iou):
    from ignition_revision.detection import iou_matrix
    from ignition_revision.evaluation import optimal_pairs
    from ignition_revision.common import boolean
    rows=[]
    for f in frames.itertuples(index=False):
        a=candidates[(candidates.recording_id==f.recording_id)&(candidates.frame_idx==f.frame_idx)]
        b=gt[(gt.recording_id==f.recording_id)&(gt.frame_idx==f.frame_idx)]
        if 'ignore' in b:b=b[~b['ignore'].map(boolean)]
        overlap=iou_matrix(b,a);matched={i for i,j in optimal_pairs(overlap>=iou,1-overlap)}
        for i,r in enumerate(b.itertuples(index=False)):
            rows.append({'recording_id':f.recording_id,'frame_idx':int(f.frame_idx),'ref_object_id':r.ref_object_id,
                'detected':i in matched,'state':getattr(r,'state','unknown'),'size_bin':getattr(r,'size_bin','unknown')})
    return pd.DataFrame(rows)


def score(root,allow_partial=False):
    root,p,frames,gt,sets=load_protocol(root)
    if not (root/'model_provenance.json').exists():raise ValueError('No model inference checkpoints; run infer first')
    model_meta=read(root/'model_provenance.json');model_hash=digest(model_meta);ph=sha(root/'protocol.json')
    cells={};missing=[];verified_calls=0
    for name in sets:
        for seed in p['seeds']:
            calls=[];absent=[]
            for key,path,prompt in schedule(frames,{name:sets[name]},[seed]):
                checkpoint=call_path(root,key)
                if not checkpoint.exists():absent.append(key);continue
                d=sealed_read(checkpoint,ph,key,model_hash)
                if d['prompt']!=prompt:raise ValueError('Checkpoint wording mismatch')
                calls.append(d);verified_calls+=1
            if absent:missing.append({'set':name,'seed':seed,'missing_calls':len(absent),'first_missing':absent[0]})
            else:cells[(name,seed)]=calls
    if missing and not allow_partial:raise ValueError(f'Incomplete inference: {missing}. Resume infer; or use score --allow-partial (scores only complete set/seed cells).')
    if not cells:raise ValueError('No complete set/seed cell; partial-frame cohorts are never scored silently')
    from ignition_revision.detection import clean_candidates
    from ignition_revision.evaluation import detector_metrics,counts_metrics
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f');report=root/'reports'/stamp;report.mkdir(parents=True)
    metrics=[];members=[];per_frames=[]
    for (name,seed),calls in cells.items():
        cell=report/name/str(seed);cell.mkdir(parents=True);raw,invalid,responses=raw_candidates(calls)
        csv(cell/'raw_candidates.csv',raw);csv(cell/'invalid_outputs.csv',invalid,columns=['set','seed','recording_id','frame_idx','group','slot','box_index','reason'])
        csv(cell/'raw_responses.csv',responses)
        cfg={'upstream_root':str(root/'source_snapshot'),'project_root':str(ROOT)}
        if len(raw):cleaned,audit=clean_candidates(raw,frames,p['cleaning'],SimpleNamespace(out=cell),cfg)
        else:cleaned=raw.copy();audit=raw.copy()
        csv(cell/'cleaning_audit.csv',audit);csv(cell/'cleaned_candidates.csv',cleaned)
        for condition in p['conditions']:
            pool=cleaned if condition=='both_cleaned' else raw
            group=condition.split('_')[0]
            if group in GROUPS:pool=pool[pool.prompt_group.eq(group)]
            m,per=detector_metrics(pool,gt,frames,p['iou_threshold'])
            bad=[x for x in invalid if group=='both' or x['group']==group] if condition!='both_cleaned' else []
            if bad:
                m=counts_metrics(m['TP'],m['FP']+len(bad),m['FN'])
                bad_count=Counter((x['recording_id'],x['frame_idx']) for x in bad)
                per=[{**r,**counts_metrics(r['TP'],r['FP']+bad_count[(r['recording_id'],r['frame_idx'])],r['FN'])} for r in per]
            metrics.append({'set':name,'seed':seed,'condition':condition,'N_frames':len(frames),
                'candidate_boxes':len(pool),'invalid_raw_counted_FP':len(bad),**m})
            mem=reference_members(pool,gt,frames,p['iou_threshold']);mem['set']=name;mem['seed']=seed;mem['condition']=condition;members.append(mem)
            per_frames.extend({'set':name,'seed':seed,'condition':condition,**r} for r in per)
        print(f'PROMPT_SCORE {name} seed={seed}: {len(raw)} valid raw candidates; {len(cleaned)} cleaned',flush=True)
    metric=pd.DataFrame(metrics);member=pd.concat(members,ignore_index=True)
    csv(report/'metrics_by_seed.csv',metric);csv(report/'reference_members.csv',member);csv(report/'per_frame_metrics.csv',per_frames)
    summaries=[]
    for (name,condition),g in metric.groupby(['set','condition']):
        row={'set':name,'condition':condition,'N_seeds':len(g)}
        for col in ['precision','recall','f1','TP','FP','FN']:
            row.update({col+'_mean':g[col].mean(),col+'_std':g[col].std(ddof=0),col+'_min':g[col].min(),col+'_max':g[col].max()})
        summaries.append(row)
    csv(report/'summary_across_seeds.csv',summaries)
    paired=[];changes=[];keys=['recording_id','frame_idx','ref_object_id']
    for name in sets:
        if name=='A':continue
        for seed in p['seeds']:
            if ('A',seed) not in cells or (name,seed) not in cells:continue
            for condition in p['conditions']:
                a=member[(member['set']=='A')&(member.seed==seed)&(member.condition==condition)]
                b=member[(member['set']==name)&(member.seed==seed)&(member.condition==condition)]
                merged=a[keys+['detected']].merge(b[keys+['detected']],on=keys,validate='one_to_one',suffixes=('_A','_variant'))
                if len(merged)!=len(a) or len(a)!=len(b):raise ValueError('Reference cohort mismatch')
                aa=merged.detected_A;bb=merged.detected_variant
                paired.append({'baseline':'A','variant':name,'seed':seed,'condition':condition,'GT_boxes':len(merged),
                    'both_detected':int((aa&bb).sum()),'A_detected_variant_missed':int((aa&~bb).sum()),
                    'A_missed_variant_detected':int((~aa&bb).sum()),'both_missed':int((~aa&~bb).sum())})
                changed=merged[aa!=bb].copy();changed['variant']=name;changed['seed']=seed;changed['condition']=condition;changes.append(changed)
    csv(report/'paired_gt_detection_changes.csv',paired,columns=['baseline','variant','seed','condition','GT_boxes','both_detected','A_detected_variant_missed','A_missed_variant_detected','both_missed'])
    csv(report/'changed_gt_members.csv',pd.concat(changes,ignore_index=True) if changes else pd.DataFrame(columns=keys+['detected_A','detected_variant','variant','seed','condition']))
    strata=member.groupby(['set','seed','condition','size_bin'],dropna=False).detected.agg(['size','sum']).reset_index().rename(columns={'size':'GT_boxes','sum':'TP'})
    strata['recall']=strata.TP/strata.GT_boxes;csv(report/'recall_by_size.csv',strata)
    manifest={'status':'partial' if missing else 'completed','scope':p['scope'],'protocol_hash':ph,'model_hash':model_hash,
        'complete_cells':len(cells),'expected_cells':len(sets)*len(p['seeds']),'verified_calls':verified_calls,'missing_cells':missing,
        'primary_condition':'both_cleaned','input_GT_files_unchanged':True,
        'no_native_detector_scores_or_AP':True,'outputs':{str(f.relative_to(report)):sha(f) for f in report.rglob('*.csv')}}
    write(report/'report_manifest.json',manifest);write(root/'latest_report.json',{'report_dir':str(report),'status':manifest['status']})
    print('PROMPT_REPORT '+str(report),flush=True)
    return report


def status(root):
    root,p,frames,gt,sets=load_protocol(root,check_images=False)
    model_hash=digest(read(root/'model_provenance.json')) if (root/'model_provenance.json').exists() else None
    done=0
    for key,path,prompt in schedule(frames,sets,p['seeds']):
        f=call_path(root,key)
        if f.exists():sealed_read(f,sha(root/'protocol.json'),key,model_hash);done+=1
    result={'scope':p['scope'],'frames':len(frames),'GT_boxes':len(gt),'sets':list(sets),'seeds':p['seeds'],
        'expected_calls':p['expected_calls'],'verified_completed_calls':done,'remaining_calls':p['expected_calls']-done}
    print(json.dumps(result,ensure_ascii=False,indent=2));return result


def main(argv=None):
    parser=argparse.ArgumentParser(description='Frozen-cohort paired-seed prompt wording validation, resumable per call')
    sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--output',type=Path,default=DEFAULT_OUT);prep.add_argument('--inputs',type=Path,default=DATA)
    prep.add_argument('--seeds',default='2026,2027,2028');prep.add_argument('--limit-frames',type=int);prep.add_argument('--prompt-sets',type=Path)
    for name in ['infer','all']:
        p=sub.add_parser(name);p.add_argument('--root',type=Path,default=DEFAULT_OUT);p.add_argument('--model-path',default='nvidia/LocateAnything-3B')
        p.add_argument('--device',default='cuda');p.add_argument('--sets',help='Comma-separated subset of A,B,C; saved cells remain reusable')
        p.add_argument('--dtype',choices=['auto','bfloat16','float16','float32'],default='auto')
        p.add_argument('--max-new-calls',type=int,help='Stop after this many NEW calls, preserving all checkpoints')
    sc=sub.add_parser('score');sc.add_argument('--root',type=Path,default=DEFAULT_OUT);sc.add_argument('--allow-partial',action='store_true')
    st=sub.add_parser('status');st.add_argument('--root',type=Path,default=DEFAULT_OUT)
    dc=sub.add_parser('doctor');dc.add_argument('--model-path',default='nvidia/LocateAnything-3B');dc.add_argument('--device',default='cuda')
    args=parser.parse_args(argv)
    if args.command=='prepare':prepare(args.output,args.inputs,tuple(int(x) for x in args.seeds.split(',')),args.limit_frames,args.prompt_sets)
    elif args.command=='score':score(args.root,args.allow_partial)
    elif args.command=='status':status(args.root)
    elif args.command=='doctor':doctor(args.model_path,args.device)
    else:
        if args.max_new_calls is not None and args.max_new_calls<1:parser.error('--max-new-calls must be positive')
        if args.command=='all' and not args.root.exists():prepare(args.root)
        selected=[x.strip() for x in args.sets.split(',')] if args.sets else None
        state=infer(args.root,args.model_path,args.device,selected,args.max_new_calls,args.dtype)
        if args.command=='all':
            if state['status']=='budget_paused':print('Budget pause: resume all with the same root; no partial result scored as full.')
            else:score(args.root,allow_partial=bool(selected))
