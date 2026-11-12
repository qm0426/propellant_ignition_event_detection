from .common import write_json

def export_mot(obs,out):
    for rec,df in obs.groupby('recording_id'):
        path=out/'motchallenge'/str(rec)/'data.txt';path.parent.mkdir(parents=True,exist_ok=True)
        lines=[]
        for _,x in df.iterrows():lines.append(f'{int(x.frame_idx)+1},{int(x.track_id)},{x.x1},{x.y1},{x.x2-x.x1},{x.y2-x.y1},1,-1,-1,-1')
        path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(out/'motchallenge'/'coordinate_protocol.json',{'internal_frame':'0-based global','mot_frame':'global frame + 1','bbox':'top-left xywh pixels','confidence':'output inclusion sentinel, not detector native confidence'})
