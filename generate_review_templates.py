"""Export portable review-method plans, without guessing data or freezing a protocol."""
import copy,itertools,json
from pathlib import Path
from ignition_revision.common import PROJECT,write_json
from ignition_revision.review_v2 import registry
from ignition_revision.experiments import decision_variants

def relative_sources(x):
    if isinstance(x,dict):return {k:relative_sources(v) for k,v in x.items()}
    if isinstance(x,list):return [relative_sources(v) for v in x]
    if isinstance(x,str):
        try:return Path(x).relative_to(PROJECT).as_posix()
        except ValueError:return x
    return x

def main():
    root=PROJECT/'revision_configs';jobs=registry();runs=[]
    def base(family):
        c=json.loads((root/(family+'.template.json')).read_text(encoding='utf-8'))
        c['seed']=2026;c['gmm_config']={'random_state':2026}
        c['evaluation'].update(mapping_max_distance=30,mapping_min_shared_frames=3,mapping_min_purity=.8,tolerances=[1,3,5,10])
        c['protocol_status']='draft_not_frozen'
        return c
    def save(name,family,c):
        write_json(root/(name+'.review.template.json'),relative_sources(c))
        runs.append({'id':name,'family':family,'config':'revision_configs/'+name+'.review.template.json','enabled':True})
    c=base('judges');c['methods']={k:copy.deepcopy(v['settings']) for k,v in jobs.items() if v['group'] in ('judges','rtv_pairs')}
    save('criteria','judges',c)
    c=base('ablations');c['methods']=decision_variants({'background_factor':{'kind':'independent_columns','columns':['bg_med','bg_mad']}})
    save('decision_features','judges',c)
    c=base('assignment');c['variants']=[v['settings'] for v in jobs.values() if v['group'] in ('assignment','native_mot')]
    save('assignment_native','assignment',c)
    c=base('restoration');c['variants']=[]
    for backend in ('ot','hungarian'):
        for r,g,p in itertools.product((False,True),repeat=3):
            c['variants'].append({'id':f'{backend}_R{int(r)}G{int(g)}P{int(p)}','backend':backend,'rescue':r,'restoration':{'gap':g,'prebirth':p}})
    save('factorial','restoration',c)
    dog=copy.deepcopy(c);dog['judge']=copy.deepcopy(jobs['DOG_PAPER_ADAPTED_RTV']['settings'])
    save('factorial_dog','restoration',dog)
    c=base('judges');c['methods']={k:v['settings'] for k,v in jobs.items() if v['group']=='robustness' and v['family']=='judges'}
    save('sensitivity','judges',c)
    for name,job in jobs.items():
        if job['group'] not in ('detection','cleaning') or job['family']!='detection':continue
        c=base('detection');c['frames']='revision_inputs/detector_frames.csv'
        c['reference_boxes']='revision_inputs/detector_boxes.csv';c['candidates']='revision_inputs/detector_candidates.csv'
        c['available_prompt_groups']=['cold','hot'];c['selected_method_ids']=[job['settings']['selection']]
        c['cleaning']={'disabled_rules':job['settings'].get('disabled_rules',[])}
        if 'enable_defocus_filter' in job['settings']:c['cleaning']['enable_defocus_filter']=job['settings']['enable_defocus_filter']
        c['classical']={'log':{'polarity':'dark','intensity_scale':'unit','sigmas':[2.,3.,4.,6.,8.,12.,16.,24.,32.,48.],'response_threshold':.02,'nms_iou':.3},'mser':{'polarity':'dark','min_area':4,'max_area':4096}}
        save(name,'detection',c)
    write_json(root/'review_suite.template.json',{'output_dir':'outputs/review_suite','runs':runs})
    write_json(root/'review_method_index.json',[{'id':r['id'],'family':r['family'],'config':r['config']} for r in runs])
    print('Generated criterion, decision/feature, matched/native tracker, full factorial and sensitivity templates.')
if __name__=='__main__':main()
