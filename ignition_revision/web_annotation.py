"""Local full-frame annotation server. Original evidence is read-only."""
from __future__ import annotations
import argparse,csv,copy,hashlib,json,mimetypes,re,secrets,threading,uuid,shutil
from datetime import datetime
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,unquote,parse_qs

NS=('original_20260706','application_T3_20260719')
FLAGS=('reviewed','complete','identity_checked')
class Conflict(ValueError):pass

def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write_json(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def integer(v,minimum=0,maximum=7999):
    if isinstance(v,bool) or not isinstance(v,(int,float)) or int(v)!=v or not minimum<=v<=maximum:raise ValueError('编号或帧号超出允许范围')
    return int(v)
def validate(doc,confirmed=False):
    if doc.get('imageWidth')!=1280 or doc.get('imageHeight')!=896:raise ValueError('图像尺寸不匹配')
    shapes=doc.get('shapes',[])
    if not isinstance(shapes,list) or len(shapes)>2000:raise ValueError('框数量无效')
    seen=set();keys=set()
    for s in shapes:
        if s.get('shape_type')!='rectangle' or s.get('label') not in ('particle','cold','hot'):raise ValueError('只接受 particle/cold/hot 矩形')
        k=s.get('group_id')
        if k is not None:
            integer(k,1,2147483647)
            if k in seen:raise ValueError(f'本帧出现重复人工 ID：{k}，请先拆分或删除重复框')
            seen.add(k)
        elif confirmed:raise ValueError('完整确认前必须为每个框分配人工 ID')
        pts=s.get('points');
        if not isinstance(pts,list) or len(pts)!=2 or any(not isinstance(p,list) or len(p)!=2 for p in pts):raise ValueError('矩形坐标格式错误')
        try:x1,x2=sorted([float(pts[0][0]),float(pts[1][0])]);y1,y2=sorted([float(pts[0][1]),float(pts[1][1])])
        except (ValueError,TypeError):raise ValueError('坐标不是数字')
        if not (0<=x1<x2<=1280 and 0<=y1<y2<=896):raise ValueError('矩形为空或超出原图范围')
        s['points']=[[x1,y1],[x2,y2]]
        if confirmed and s.get('flags',{}).get('identity_uncertain'):raise ValueError('仍有未裁决身份')
        key=s.get('_web_key')
        if not isinstance(key,str) or not key or key in keys:raise ValueError('框的内部标识重复或缺失')
        keys.add(key)
    flags=doc.get('flags',{})
    if any(type(flags.get(k,False)) is not bool for k in FLAGS):raise ValueError('确认状态格式无效')
    if confirmed and not all(flags.get(k,False) for k in FLAGS):raise ValueError('逐帧检查、完整性及身份确认尚未完成')

class AnnotationStore:
    def __init__(self,root):
        self.root=Path(root).resolve();self.work=self.root/'11_web_annotations';self.lock=threading.RLock()
        self.catalog=self._js(self.root/'09_viewer/catalog.js');self.boxes={n:self._js(self.root/f'09_viewer/{n}_boxes.js') for n in NS}
        kit=self.root/'06_existing_annotation_tasks/mot_identity_review_20261025';self.kit=kit
        self.clip_rows={}
        if (kit/'frame_manifest.csv').exists():
            with (kit/'frame_manifest.csv').open(encoding='utf-8-sig',newline='') as f:
                for r in csv.DictReader(f):
                    if r['annotation_required'].lower()=='true':self.clip_rows.setdefault(r['clip_id'],{})[int(r['frame_idx'])]=r
        self.scopes=[{'id':n,'label':'原 GT 轨迹复核' if i==0 else '580 应用候选复核','start':0,'end':7999,'kind':'track'} for i,n in enumerate(NS)]
        self.scopes +=[{'id':c,'label':f'MOT {c}：{min(rows)}–{max(rows)}','start':min(rows),'end':max(rows),'kind':'mot'} for c,rows in sorted(self.clip_rows.items())]
    @staticmethod
    def _js(p):return json.loads(p.read_text(encoding='utf-8').split('=',1)[1].rstrip().removesuffix(';'))
    def scope(self,s):
        for r in self.scopes:
            if r['id']==s:return r
        raise ValueError('未知标注空间')
    def path(self,s,f):
        spec=self.scope(s);f=integer(f)
        if not spec['start']<=f<=spec['end']:raise ValueError('帧不在当前标注范围内')
        return self.work/'frames'/s/f'frame_{f:08d}.json'
    def load(self,s,f):
        p=self.path(s,f)
        if p.exists():return read_json(p)
        shapes=[]
        if s in self.clip_rows:
            d=read_json(self.clip_rows[s][f]['json_path'])
            for i,sh in enumerate(d.get('shapes',[])):
                sh['_web_key']=f'seed:{i}';sh['_source_track_id']=sh.get('group_id')
        else:
            for b in self.boxes[s].get(str(f),[]):
                if b[5]!='detected':continue
                shapes.append({'label':'particle','group_id':int(b[0]),'shape_type':'rectangle','points':[[b[1],b[2]],[b[3],b[4]]],'flags':{},'_web_key':f'track:{int(b[0])}','_source_track_id':int(b[0]),'description':'Algorithm proposal; physical identity requires human review'})
            d={'version':'5.5.0','flags':dict.fromkeys(FLAGS,False),'shapes':shapes,'imageData':None,'imageHeight':896,'imageWidth':1280}
        d.update({'imagePath':f'frame_{f:08d}.png','_web_revision':0,'_scope':s,'_frame_idx':f})
        return d
    def _commit(self,s,changes,action):
        """Validate everything before mutation; keep recoverable pre-edit snapshots."""
        op=uuid.uuid4().hex;folder=self.work/'history'/op;before=[]
        for f,doc in changes:
            validate(doc,all(doc.get('flags',{}).get(k,False) for k in FLAGS))
            p=self.path(s,f);old=self.load(s,f);before.append((f,p,old,p.exists()))
        for f,p,old,existed in before:write_json(folder/f'{f:08d}.json',{'existed':existed,'document':old})
        applied=[]
        try:
            for (f,doc),(_,p,old,_) in zip(changes,before):
                doc['_web_revision']=int(old['_web_revision'])+1;doc['_updated_at']=datetime.now().isoformat(timespec='seconds')
                write_json(p,doc);applied.append((p,old))
        except Exception:
            for f,p,old,existed in before:
                if existed:write_json(p,old)
                elif p.exists():p.unlink()
            raise
        write_json(folder/'operation.json',{'id':op,'scope':s,'action':action,'frames':[f for f,_ in changes],'revisions':{str(f):doc['_web_revision'] for f,doc in changes},'undone':False})
        return op
    def save(self,s,f,doc,revision):
        with self.lock:
            current=self.load(s,f)
            if current['_web_revision']!=revision:raise Conflict('该帧已在其他页面修改。请重新载入，避免覆盖新答案。')
            d=copy.deepcopy(doc);d.update({'_scope':s,'_frame_idx':f,'imagePath':current['imagePath'],'imageWidth':1280,'imageHeight':896,'imageData':None})
            self._commit(s,[(f,d)],'save_frame');return self.load(s,f)
    def batch(self,s,start,end,rules):
        with self.lock:
            start=integer(start);end=integer(end)
            if end<start:raise ValueError('结束帧必须不早于开始帧')
            self.path(s,start);self.path(s,end);mapping={}
            for r in rules:
                a=integer(r['from'],1,2147483647);b=integer(r['to'],1,2147483647)
                if a in mapping:raise ValueError('同一个旧 ID 有多条改写规则')
                mapping[a]=b
            if not mapping:raise ValueError('没有 ID 规则')
            changes=[];count=0
            for f in range(start,end+1):
                d=self.load(s,f);n=0
                for sh in d['shapes']:
                    old=sh.get('group_id')
                    if old in mapping and mapping[old]!=old:sh['group_id']=mapping[old];n+=1
                if n:d['flags']['identity_checked']=False;validate(d);changes.append((f,d));count+=n
            op=self._commit(s,changes,'batch_ids') if changes else None
            return {'operation_id':op,'frames':len(changes),'boxes':count}
    def undo_batch(self,op):
        if not re.fullmatch('[0-9a-f]{32}',op):raise ValueError('无效操作编号')
        with self.lock:
            folder=self.work/'history'/op;meta=read_json(folder/'operation.json')
            if meta['action']!='batch_ids' or meta['undone']:raise ValueError('该批量操作不可再次撤销')
            changes=[]
            for f in meta['frames']:
                if self.load(meta['scope'],f)['_web_revision']!=meta['revisions'][str(f)]:raise Conflict('批量改 ID 后已有其他修改，不能整批撤销。请按帧核查。')
                changes.append((f,read_json(folder/f'{f:08d}.json')['document']))
            self._commit(meta['scope'],changes,'undo_batch');meta['undone']=True;write_json(folder/'operation.json',meta)
            return {'frames':len(changes)}
    def save_event(self,s,track_id,data):
        if s not in NS:raise ValueError('请在轨迹复核模式填写事件')
        integer(track_id,1,2147483647)
        if not any(c['namespace']==s and c['track_id']==track_id for c in self.catalog):raise ValueError('未知轨迹')
        if data.get('has_ignition') not in ('yes','no','uncertain',''):raise ValueError('事件结论无效')
        for key in ('onset_frame','onset_earliest','onset_latest'):
            if data.get(key) is not None:integer(data[key])
        a=data.get('onset_earliest');b=data.get('onset_latest');onset=data.get('onset_frame')
        if a is not None and b is not None and a>b:raise ValueError('起点不确定区间顺序错误')
        if onset is not None and ((a is not None and onset<a) or (b is not None and onset>b)):raise ValueError('起点不在填写的不确定区间内')
        if data.get('has_ignition')=='no' and onset is not None:raise ValueError('无事件时请清空点火帧')
        if data.get('status')=='confirmed' and (not data.get('annotator','').strip() or data.get('has_ignition') in ('','uncertain') or (data['has_ignition']=='yes' and onset is None)):raise ValueError('确认事件前需填写标注人、结论及可判定起点；不能确定时保留待定')
        p=self.work/'events'/s/f'{track_id}.json'
        with self.lock:
            current=read_json(p) if p.exists() else {'revision':0}
            if data.get('revision',0)!=current['revision']:raise Conflict('事件记录已被其他页面修改，请重新载入')
            if p.exists():write_json(self.work/'event_history'/uuid.uuid4().hex/(s+f'_{track_id}.json'),current)
            d={**data,'namespace':s,'original_track_id':track_id,'revision':current['revision']+1};write_json(p,d);return d
    def export(self,s,start,end,annotator,mot=False):
        with self.lock:
            start=integer(start);end=integer(end);self.path(s,start);self.path(s,end)
            if end<start:raise ValueError('帧范围错误')
            docs=[(f,self.load(s,f)) for f in range(start,end+1)]
            if mot:
                if not annotator.strip():raise ValueError('请填写标注人')
                for f,d in docs:
                    try:validate(d,True)
                    except ValueError as e:raise ValueError(f'第 {f} 帧：{e}') from e
            name=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6];out=self.work/'exports'/name
            seq=f'MCFF0004_N001__web_{s}_{start}_{end}';frames=[];pred=[]
            original=NS[0] if s in self.clip_rows else s
            source=self.root/('03_original_tracks/tracks_all_2800.csv' if original==NS[0] else '05_application580/source/tracks_long.csv')
            for f,d in docs:
                path=out/'tasks'/f'frame_{f:08d}.json';write_json(path,d)
                image_source=self.root/f'02_full_frames/view_png/frame_{f:08d}.png';image=path.with_suffix('.png');shutil.copy2(image_source,image)
                raw=self.root/f'02_full_frames/original_tif/MCFF0004-N001_{f:08d}.tif'
                frames.append({'clip_id':'web_clip','recording_id':seq,'source_recording_id':'MCFF0004_N001','frame_idx':f,'sequence_frame':f-start+1,'annotation_required':True,'image_path':str(image),'json_path':str(path),'original_image':str(raw),'width':1280,'height':896,'task_image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),'source_image_sha256':hashlib.sha256(raw.read_bytes()).hexdigest()})
            self._csv(out/'frame_manifest.csv',frames)
            self._csv(out/'clip_catalog.csv',[{'clip_id':'web_clip','selection_kind':'human_selected','start_frame':start,'end_frame':end,'source_recording_id':'MCFF0004_N001','mot_sequence_id':seq}])
            with source.open(encoding='utf-8-sig',newline='') as stream:
                for r in csv.DictReader(stream):
                    if start<=int(r['frame_id'])<=end:pred.append({**r,'frame_idx':int(r['frame_id']),'recording_id':seq,'source_recording_id':'MCFF0004_N001'})
            self._csv(out/'original_tracker_predictions.csv',pred,['track_id','frame_id','frame_idx','recording_id','x1','y1','x2','y2'])
            snap=out/'original_tracker_predictions.csv'
            write_json(out/'export_meta.json',{'config':{'source_fps':50000},'prediction_snapshot_sha256':hashlib.sha256(snap.read_bytes()).hexdigest(),'source_namespace':original,'gt_status':'human reviewed' if mot else 'draft; not evaluation ready'})
            # Event decisions remain distinct from trajectory identity reference.
            events=[read_json(p) for p in sorted((self.work/'events'/original).glob('*.json'))]
            self._csv(out/'event_reviews.csv',events,['namespace','original_track_id','has_ignition','onset_frame','status'])
            segments=[]
            for f,d in docs:
                for sh in d['shapes']:
                    segments.append({'namespace':original,'global_frame_id':f,'original_track_id':sh.get('_source_track_id'),'human_particle_id':sh.get('group_id'),'x1':sh['points'][0][0],'y1':sh['points'][0][1],'x2':sh['points'][1][0],'y2':sh['points'][1][1],'frame_confirmed':all(d.get('flags',{}).get(k,False) for k in FLAGS)})
            self._csv(out/'identity_frame_mapping.csv',segments,['namespace','global_frame_id','original_track_id','human_particle_id','x1','y1','x2','y2','frame_confirmed'])
            if mot:
                from .mot_review import import_mot_review
                import_mot_review(out,out/'mot_reference',annotator)
            write_json(out/'web_export.json',{'scope':s,'start':start,'end':end,'annotator':annotator,'mot_ready':mot,'note':'Human selected continuous scope. Not a representative random sample unless separately designed. Labelme images are copied alongside JSON; MOT manifests contain absolute source paths, rebase if relocating.'})
            return {'path':str(out),'mot_ready':mot,'reference_dir':str(out/'mot_reference') if mot else None}
    @staticmethod
    def _csv(path,rows,empty_columns=None):
        path.parent.mkdir(parents=True,exist_ok=True);columns=list(dict.fromkeys(k for r in rows for k in r)) or empty_columns or ['status']
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)

def serve(root,port=8765,open_browser=False):
    store=AnnotationStore(root);token=secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,data,status=200):
            b=json.dumps(data,ensure_ascii=False,allow_nan=False).encode();self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(b)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(b)
        def do_GET(self):
            try:
                if self.headers.get('Host') not in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'):return self.reply({'error':'只允许本机地址访问'},403)
                url=urlparse(self.path);q=parse_qs(url.query)
                if url.path=='/api/bootstrap':return self.reply({'application':'gt_annotation_v1','token':token,'scopes':store.scopes,'catalog':store.catalog,'output':str(store.work)})
                if url.path=='/api/frame':
                    s=q['scope'][0];f=int(q['frame'][0]);return self.reply({'document':store.load(s,f),'proposals':store.boxes[NS[0] if s in store.clip_rows else s].get(str(f),[])})
                if url.path=='/api/event':
                    s=q['scope'][0];store.scope(s);k=integer(int(q['track'][0]),1,2147483647);p=store.work/'events'/s/f'{k}.json'
                    return self.reply(read_json(p) if p.exists() else {'revision':0})
                rel=unquote(url.path).lstrip('/') or '标注端.html';p=(store.root/rel).resolve()
                if not p.is_relative_to(store.root) or not p.is_file():return self.reply({'error':'文件不存在'},404)
                b=p.read_bytes();self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(p.name)[0] or 'application/octet-stream');self.send_header('Content-Length',str(len(b)));self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(b)
            except Exception as e:self.reply({'error':str(e)},400)
        def do_POST(self):
            try:
                origin=self.headers.get('Origin');host=self.headers.get('Host')
                if host not in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}') or self.headers.get('X-Annotation-Token')!=token or origin not in (None,'http://'+host):return self.reply({'error':'请求来源无效，请从本机标注页操作'},403)
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=4_000_000:raise ValueError('请求体过大或为空')
                d=json.loads(self.rfile.read(n));path=urlparse(self.path).path
                if path=='/api/save':r=store.save(d['scope'],integer(d['frame']),d['document'],d['revision'])
                elif path=='/api/batch':r=store.batch(d['scope'],d['start'],d['end'],d['rules'])
                elif path=='/api/undo-batch':r=store.undo_batch(d['operation_id'])
                elif path=='/api/event':r=store.save_event(d['scope'],d['track_id'],d['event'])
                elif path=='/api/export':r=store.export(d['scope'],d['start'],d['end'],d.get('annotator',''),bool(d.get('mot',False)))
                else:return self.reply({'error':'未知接口'},404)
                self.reply(r)
            except Conflict as e:self.reply({'error':str(e)},409)
            except Exception as e:self.reply({'error':str(e)},400)
    try:server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    except OSError:
        # A second launcher should reopen this annotation service, not spawn a duplicate.
        import urllib.request
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/bootstrap',timeout=2) as response:existing=json.load(response)
            if existing.get('output')!=str(store.work):raise ValueError('端口被其他程序占用，请使用 --port 8766')
        except Exception as e:raise RuntimeError(f'端口 {port} 无法使用，请使用 --port 8766') from e
        print(f'标注服务已运行：http://127.0.0.1:{port}/',flush=True)
        if open_browser:
            import webbrowser;webbrowser.open(f'http://127.0.0.1:{port}/')
        return
    print(f'标注端：http://127.0.0.1:{server.server_port}/  保存目录：{store.work}',flush=True)
    if open_browser:
        import webbrowser;webbrowser.open(f'http://127.0.0.1:{server.server_port}/')
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]/'GT_review_materials_20261025');p.add_argument('--port',type=int,default=8765);p.add_argument('--open',action='store_true');a=p.parse_args();serve(a.root,a.port,a.open)

if __name__=='__main__':main()
