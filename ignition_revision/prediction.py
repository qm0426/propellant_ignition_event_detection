from pathlib import Path
import warnings
import pandas as pd
from .common import Blocked,write_csv,write_json
from .judges import judge_track

def predict_feature_bundle(bundle,mod,base,methods,cfg,run,output_root=None):
    output_root=Path(output_root or run.out);outputs={}
    for method_id,settings in methods.items():
        rows=[];audits=[];diagnostics=[]
        try:
            for progress,((rec,tid),df) in enumerate(bundle.items(),1):
                with run.time('judge_'+method_id,unit='tracks',device=settings.get('device','cpu')):
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter('always')
                        result,candidates,trace=judge_track(df,method_id,settings,mod,base,cfg)
                    trace['warnings']=[{'category':type(w.message).__name__,'message':str(w.message)} for w in captured]
                rows.append({'recording_id':rec,'track_id':tid,**result})
                audits.extend({'recording_id':rec,'track_id':tid,'method_id':method_id,**c} for c in candidates)
                diagnostics.append({'recording_id':rec,'track_id':tid,'trace':trace})
                if len(bundle)>250 and (progress%250==0 or progress==len(bundle)):
                    print(f'JUDGE_PROGRESS {method_id}: {progress}/{len(bundle)} tracks',flush=True)
            table=pd.DataFrame(rows,columns=['recording_id','track_id','method_id','has_ignition','onset_frame','confirmation_frame','status','feature_schema_hash','candidate_count','future_context','onset_cx','onset_cy','onset_x1','onset_y1','onset_x2','onset_y2','combustion_frame'])
            out=output_root/method_id
            write_csv(out/'predictions.csv',table);write_json(out/'candidate_audit.json',audits);write_json(out/'diagnostics.json',diagnostics)
            write_json(out/'method_config.json',settings)
            outputs[method_id]=table
        except Blocked as e:run.record(method_id,'blocked',str(e))
        except Exception as e:
            run.record(method_id,'failed',f'{type(e).__name__}: {e}')
            write_json(output_root/method_id/'error.json',{'exception':type(e).__name__,'message':str(e)})
    return outputs
