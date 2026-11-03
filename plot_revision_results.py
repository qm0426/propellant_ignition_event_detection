#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manuscript-style figures for registered revision results (S03/S04/S05/S06/S08).
No cross-run merging, missing-value imputation, or legacy-column renaming.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import textwrap
import numpy as np
import pandas as pd
from ignition_revision.common import load_legacy, write_csv, write_json

TABLES={'fpn-events':'S06_judge_comparison.csv','fpn-pipeline':'S06_judge_comparison.csv','infer-fpn':'S03_detector_comparison.csv','detection':'S03_detector_comparison.csv','assignment':'S04_assignment_and_mot.csv',
        'restoration':'S05_restoration_events.csv','judges':'S06_judge_comparison.csv','ablations':'S08_decision_feature_ablation.csv'}
LABELS={
 'FPN_IGNITION_SHARED_TRACKS':'FPN ignition + shared tracks',
 'SAS_PAPER_ADAPTED':'SAS (ROI adapted)','DOG_PAPER_ADAPTED':'DoG (known-center adapted)',
 'CNF_ROI_STAGE_ADAPTED':'C&F ROI stage (adapted)','FPN_FASTER_RCNN':'Faster R-CNN + FPN',
 'J_GRAY_LEGACY':'Gray threshold','J_CHANGE_LEGACY':'Change score','J_CHANGE_LEGACY_RTV':'Change score + RTV',
 'J_GMM':'GMM w/o RTV','J_GMM_RTV':'Proposed GMM-RTV','J_PRIOR9':'Prior work [9]',
 'J_PELT':'PELT','J_PELT_RTV':'PELT + RTV','J_SAS_ADAPTED':'SAS (adapted)','J_SAS_ADAPTED_RTV':'SAS + RTV',
 'J_DOG_IGNITION':'DoG (adapted)','J_DOG_IGNITION_RTV':'DoG + RTV','J_RESNET18_ADAPTED':'ResNet18 (adapted)','J_RESNET18_ADAPTED_RTV':'ResNet18 + RTV',
 'ASSIGN_H_R0':'Hungarian / Rescue off','ASSIGN_H_R1':'Hungarian / Rescue on','ASSIGN_O_R0':'OT / Rescue off','ASSIGN_O_R1':'OT / Rescue on',
 'REST_T0':'T0 Center-NN','REST_T1':'T1 OT-Kalman','REST_T2':'T2 + Rescue','REST_T2B':'T2b + Gap filling','REST_FULL':'T3 Full tracker',
 'REST_NO_RESCUE':'w/o Rescue','REST_NO_GAP':'w/o Gap filling','REST_NO_PREBIRTH':'w/o Prebirth',
 'DEC_FULL':'Proposed GMM-RTV','DEC_NO_GEOM':'w/o shape precheck','DEC_NO_INITIALHOT':'w/o initial-hot precheck',
 'DEC_NO_SMOOTH':'w/o run smoothing','DEC_NO_PRIORCOLD':'w/o prior-cold guard','DEC_NO_RTV':'w/o RTV',
 'FEAT_NO_DELTA':'w/o difference features','FEAT_NO_COLOR':'w/o color features','FEAT_NO_MORPH':'w/o morphology features',
 'FEAT_FULL_PLUS_BG':'Full19 + background (21D)','FEAT_NO_BG':'Full19 (19D)',
 'FEAT_GRAY':'Gray (2D)','FEAT_GRAY_BG':'Gray + background (4D)',
 'GEOM_0_BG_0':'Geometry off / background off','GEOM_0_BG_1':'Geometry off / background on',
 'GEOM_1_BG_0':'Geometry on / background off','GEOM_1_BG_1':'Geometry on / background on',
 'MOT_SORT':'SORT (author implementation)',
 'MOT_BYTE':'ByteTrack (quality-score adapted)','MOT_OCSORT':'OC-SORT (fixed pool; Byte off)',
 'SAS_PAPER_ADAPTED_RTV':'SAS (ROI adapted) + RTV','DOG_PAPER_ADAPTED_RTV':'DoG (adapted) + RTV',
 'CNF_ROI_STAGE_ADAPTED_RTV':'C&F ROI stage (adapted) + RTV',
 'DET_COLD_raw':'Cold prompt / raw','DET_COLD_cleaned':'Cold prompt / cleaned','DET_HOT_raw':'Hot prompt / raw',
 'DET_HOT_cleaned':'Hot prompt / cleaned','DET_BOTH_raw':'Joint prompts / raw','DET_BOTH_cleaned':'Joint prompts / cleaned',
 'DET_log':'LoG','DET_mser':'MSER'}
for suite_id,internal_id in {'DET_COLD_RAW':'DET_COLD_raw','DET_HOT_RAW':'DET_HOT_raw',
                            'DET_BOTH_RAW':'DET_BOTH_raw','DET_BOTH_CLEANED':'DET_BOTH_cleaned',
                            'DET_LOG':'DET_log','DET_MSER':'DET_mser'}.items():
    LABELS[suite_id]=LABELS[internal_id]
for rule in ('BACKGROUND','GEOMETRY','DEFOCUS','NESTED','DUPLICATES','NMS','ALL'):
    LABELS['CLEAN_NO_'+rule]='Cleaning w/o '+rule.lower()
LABELS['CLEAN_WITH_DEFOCUS']='Cleaning with defocus filter'
ORDER=list(LABELS)
COLORS=['#4C78A8','#F58518','#54A24B','#D55E00','#B279A2','#72B7B2']
GROUPS=('performance','counts','onset','effects','observability','mot','rtv','restoration','runtime','overview')


def read_csv(path):
    if not path.is_file():return pd.DataFrame()
    try:return pd.read_csv(path)
    except pd.errors.EmptyDataError:return pd.DataFrame()


def ordered(df):
    if df.empty:return df
    if 'method_id' not in df:raise ValueError('Expected method_id in metric table')
    if df.method_id.duplicated().any():raise ValueError('Duplicate method IDs: do not supply merged cross-run tables')
    ids=list(df.method_id.astype(str));order=[m for m in ORDER if m in ids]+[m for m in ids if m not in ORDER]
    return df.assign(method_id=df.method_id.astype(str)).set_index('method_id').loc[order].reset_index()


def labels(ids):return [textwrap.fill(LABELS.get(str(x),str(x).replace('_',' ')),32) for x in ids]


def numeric(df,cols):
    return df.reindex(columns=cols).apply(pd.to_numeric,errors='coerce')


def effect_table(df,baseline):
    """Positive is better: full-minus-removed for scores; removed-minus-full for FPR."""
    if df.empty or baseline not in set(df.method_id):return pd.DataFrame()
    for col in ('cohort_hash','track_bundle_id','evaluation_scope'):
        if col in df and df[col].dropna().nunique()>1:raise ValueError(f'Cannot compare differing {col}')
    base=df[df.method_id==baseline].iloc[0];rows=[]
    for _,row in df.iterrows():
        if row.method_id==baseline:continue
        item={'method_id':row.method_id}
        for col in ('precision','recall','f1','false_positive_rate'):
            a=pd.to_numeric(base.get(col),errors='coerce');v=pd.to_numeric(row.get(col),errors='coerce')
            item[col]=(v-a if col=='false_positive_rate' else a-v)*100
        rows.append(item)
    return pd.DataFrame(rows)


def runtime_table(df):
    """Sum seconds and units, then divide. Never average row-level ms/unit."""
    if df.empty:return df
    required={'stage','device','unit','cache_mode','total_s','N_units'}
    if not required<=set(df):raise ValueError(f'Runtime missing {required-set(df)}')
    work=df.copy()
    for c in ('total_s','N_units'):work[c]=pd.to_numeric(work[c],errors='raise')
    if (work[['total_s','N_units']]<0).any().any():raise ValueError('Negative runtime/units')
    result=work.groupby(['stage','device','unit','cache_mode'],dropna=False,sort=False)[['total_s','N_units']].sum().reset_index()
    result['weighted_ms_per_unit']=1000*result.total_s/result.N_units.where(result.N_units>0)
    return result


def bars(ax,df,cols,names,xlabel,rate=False):
    vals=numeric(df,cols);n=len(df);width=.76/max(1,len(cols));y=np.arange(n)
    for j,(col,name) in enumerate(zip(cols,names)):
        v=vals[col].to_numpy(float);pos=y-.38+width*(j+.5);valid=np.isfinite(v)
        ax.barh(pos[valid],v[valid],height=width*.91,color=COLORS[j%len(COLORS)],label=name,zorder=3)
        for yy in pos[~valid]:ax.text(.008,yy,'NA',transform=ax.get_yaxis_transform(),va='center',fontsize=7,color='#777777')
    ax.set_yticks(y,labels(df.method_id));ax.set_ylim(n-.5,-.5);ax.set_xlabel(xlabel)
    ax.grid(axis='x',alpha=.2,zorder=0);ax.set_axisbelow(True)
    if rate:ax.set_xlim(0,1.035)
    elif not np.isfinite(vals.to_numpy()).any() or not np.any(np.nan_to_num(vals.to_numpy())>0):ax.set_xlim(0,1)
    else:ax.set_xlim(left=0)
    from matplotlib.patches import Patch
    handles=[Patch(facecolor=COLORS[j%len(COLORS)],label=name) for j,name in enumerate(names)]
    ax.legend(handles=handles,frameon=False,loc='lower left',bbox_to_anchor=(0,1.01),ncol=min(4,len(cols)),borderaxespad=0)


class Plotter:
    def __init__(self,args,run,meta):
        self.args=args;self.run=run;self.meta=meta;self.out=args.output/meta['family']/meta['run_id']
        self.files=[];self.skipped=[];self.sources={};self.captions=[]
        self.cfg=json.loads((run/'resolved_config.json').read_text(encoding='utf-8')) if (run/'resolved_config.json').is_file() else {}
        self.style=load_legacy('manuscript_main_figures_mm.py');self.style.configure_matplotlib()
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size':args.font_size,'axes.labelsize':args.font_size,'xtick.labelsize':args.font_size-1,
            'ytick.labelsize':args.font_size-1,'legend.fontsize':args.font_size-2,'pdf.fonttype':42,'ps.fonttype':42})
        self.plt=plt
    def table(self,name):
        p=self.run/name;df=read_csv(p)
        if p.is_file():self.sources[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
        return df
    def new(self,n=1,rows=5):
        return self.plt.subplots(1,n,figsize=(8.2*n,max(3.4,.52*rows+1.65)),squeeze=False,layout='constrained')
    def save(self,fig,name,caption,data=None):
        self.out.mkdir(parents=True,exist_ok=True)
        scope=self.cfg.get('evaluation_mode','configured reference scope (see run metadata)')
        note=' | '.join(x for x in [self.args.label,f"Run {self.meta['run_id']}",scope,'PARTIAL' if self.meta['status']=='partial' else ''] if x)
        if not self.args.clean_figures:fig.supxlabel(note,fontsize=7,color='#666666')
        if self.meta.get('synthetic'):fig.text(.5,.5,'SYNTHETIC VALIDATION ONLY',ha='center',va='center',fontsize=22,color='#D55E00',alpha=.23,rotation=20)
        for ax in fig.axes:ax.set_title('')
        for ext in self.args.formats.split(','):
            path=self.out/(name+'.'+ext);fig.savefig(path,dpi=self.args.dpi,bbox_inches='tight',pad_inches=.06);self.files.append(str(path.resolve()))
        self.plt.close(fig)
        if data is not None:write_csv(self.out/(name+'_data.csv'),data)
        self.captions.append(f'### {name}\n\n{caption}\n\nSource run: `{self.run.resolve()}`. Scope: {scope}.\n')
    def skip(self,name,reason):self.skipped.append({'figure':name,'reason':reason})
    def metric_figures(self,df):
        groups=self.args.groups
        if df.empty:self.skip('metrics','No nonempty scored summary table');return
        if 'performance' in groups:
            fig,axs=self.new(rows=len(df));bars(axs[0,0],df,['precision','recall','f1'],['Precision','Recall','F1'],'Detection / ignition score',True)
            self.save(fig,'01_performance','Precision, recall and F1 in the run-defined evaluation cohort. NA denotes undefined/unavailable scores, not zero.',df)
        if 'counts' in groups:
            cols=[c for c in ('TP','FP','FN','TN') if c in df and pd.to_numeric(df[c],errors='coerce').notna().any()]
            if cols:
                fig,axs=self.new(rows=len(df));bars(axs[0,0],df,cols,cols,'Number of events / fixed-cohort tracks')
                self.save(fig,'02_confusion_counts','TP/FP/FN/TN retain the original evaluator denominators. Event-matching evaluation may have no TN.',df[['method_id',*cols]])
        if 'onset' in groups:self.onset(df)
        if 'observability' in groups:
            cols=['roi_pre_onset_coverage','detector_pre_onset_coverage','roi_window_completeness','detector_window_completeness','observable_transition']
            if any(c in df and pd.to_numeric(df[c],errors='coerce').notna().any() for c in cols):
                fig,axs=self.new(rows=len(df));ax=axs[0,0];x=np.arange(len(df))
                for j,col in enumerate(cols):
                    if col in df:ax.plot(x,pd.to_numeric(df[col],errors='coerce'),marker=['o','s','D','^','v'][j],color=COLORS[j],label=col.replace('_',' ').capitalize())
                ax.set_xticks(x,labels(df.method_id),rotation=25,ha='right');ax.set_ylim(0,1.035);ax.set_ylabel('Coverage / observability');ax.grid(alpha=.2)
                ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0,1.01),ncol=2)
                self.save(fig,'04_observability','ROI support and detector support are plotted separately. Fixed reference-event window denominators follow the current evaluator.',df[['method_id',*[c for c in cols if c in df]]])
            else:self.skip('04_observability','No observability fields')
        if 'mot' in groups:
            cols=[c for c in ('HOTA','IDF1') if c in df and pd.to_numeric(df[c],errors='coerce').notna().any()]
            counts=[c for c in ('IDSW','fragmentations') if c in df and pd.to_numeric(df[c],errors='coerce').notna().any()]
            if cols or counts:
                fig,axs=self.new(n=int(bool(cols))+int(bool(counts)),rows=len(df));j=0
                if cols:bars(axs[0,j],df,cols,cols,'Standard MOT score',True);j+=1
                if counts:bars(axs[0,j],df,counts,counts,'Identity switches / fragmentations')
                self.save(fig,'05_standard_mot','Official TrackEval HOTA/IDF1 are fractions; IDSW and fragmentations are counts. Algorithm-reference diagnostic switches are not substituted for standard IDSW.',df[['method_id',*cols,*counts]])
            else:self.skip('05_standard_mot','Standard MOT unavailable; old algorithm tracks are not independent MOT GT')
        if 'effects' in groups:
            baseline=self.args.baseline or {'restoration':'REST_FULL','ablations':'DEC_FULL'}.get(self.meta['family'])
            if baseline:
                effects=effect_table(df,baseline)
                for name,subset in [('06_module_effects',effects[~effects.method_id.str.startswith('FEAT_')] if not effects.empty else effects),('07_feature_effects',effects[effects.method_id.str.startswith('FEAT_')] if not effects.empty else effects)]:
                    if subset.empty:self.skip(name,'No selected full-versus-ablation pairs');continue
                    fig,axs=self.new(rows=len(subset));table=subset.rename(columns={'method_id':'Method','precision':'Precision (pp)','recall':'Recall (pp)','f1':'F1 (pp)','false_positive_rate':'FPR reduction (pp)'}).copy();table['Method']=labels(subset.method_id)
                    self.style.annotated_heatmap(axs[0,0],table,'',value_format='+.2f',normalize_columns_for_color=True,colorbar_label='Column-normalized effect (+ better)',annotation_fontsize=self.args.font_size-2,tick_fontsize=self.args.font_size-1)
                    self.save(fig,name,f'Full system ({baseline}) versus each alternative: full minus alternative for P/R/F1, alternative minus full for FPR, in percentage points. Color normalized within each column; printed values remain raw. MAE is excluded because method-specific TP sets differ.',subset)
    def onset(self,df):
        name='S07_common_tp.csv' if (self.run/'S07_common_tp.csv').exists() else 'common_tp.csv'
        common=self.table(name)
        if common.empty:self.skip('03_common_tp_onset','Common-TP summary absent');return
        common=ordered(common)
        if 'cohort_hash' in common and common.cohort_hash.dropna().nunique()>1:raise ValueError('Mixed Common-TP cohorts')
        if 'common_tp_N' in common and common.common_tp_N.nunique()>1:raise ValueError('Inconsistent Common-TP cohort size')
        fig,axs=self.new(n=2,rows=len(common))
        bars(axs[0,0],common,['onset_mae','onset_median_ae','onset_p95_ae'],['MAE','Median AE','95th-percentile AE'],'Common-TP onset error (frames)')
        bars(axs[0,1],common,['acc_pm1','acc_pm3'],['Within ±1 frame','Within ±3 frames'],'Common-TP onset accuracy',True)
        n=common.common_tp_N.iloc[0] if 'common_tp_N' in common else 'unknown'
        self.save(fig,'03_common_tp_onset',f'Onset errors and accuracies use the shared true-positive reference-event cohort, N={n}. An empty intersection is shown as NA; it is not perfect zero error.',common)
    def changes(self):
        if 'rtv' in self.args.groups:
            df=self.table('S09_rtv_member_changes.csv')
            if not df.empty:
                summary=df.groupby(['method_before','method_after','before','after'],dropna=False).size().reset_index(name='N')
                pairs=summary[['method_before','method_after']].drop_duplicates();fig,axs=self.new(rows=len(pairs));ax=axs[0,0];ys=np.arange(len(pairs));left=np.zeros(len(pairs))
                transitions=sorted(set(zip(summary.before,summary.after)))
                for j,(before,after) in enumerate(transitions):
                    vals=[]
                    for _,p in pairs.iterrows():vals.append(int(summary[(summary.method_before==p.method_before)&(summary.method_after==p.method_after)&(summary.before==before)&(summary.after==after)].N.sum()))
                    ax.barh(ys,vals,left=left,color=COLORS[j%len(COLORS)],label=f'{before} → {after}');left+=vals
                ax.set_yticks(ys,[LABELS.get(x,x) for x in pairs.method_before]);ax.invert_yaxis();ax.set_xlabel('Paired fixed-cohort tracks');ax.legend(frameon=False,bbox_to_anchor=(0,1.01),loc='lower left',ncol=3)
                self.save(fig,'08_rtv_member_changes','Exact before/after RTV classification changes. TN→TN is retained to show the paired cohort denominator.',summary)
            else:self.skip('08_rtv_member_changes','No RTV paired-member results')
        if 'restoration' in self.args.groups:
            df=self.table('restoration_paired_changes.csv')
            if not df.empty:
                df=df[df.baseline!=df.full];summary=df.groupby(['baseline','change']).size().unstack(fill_value=0).reset_index().rename(columns={'baseline':'method_id'})
                if not summary.empty:
                    for key in ('new_tp','lost_tp'):
                        if key not in summary:summary[key]=0
                    fig,axs=self.new(rows=len(summary));bars(axs[0,0],summary,['new_tp','lost_tp'],['New TP from full restoration','Lost TP in full restoration'],'Number of paired reference events')
                    self.save(fig,'09_restoration_paired_tp','New/lost TP counts compare the full tracker with each baseline on reference-event identities; unchanged events are not plotted.',summary)
            fp=self.table('restoration_fp_changes.csv')
            if not fp.empty:
                write_csv(self.out/'restoration_fp_pairing_audit.csv',fp)
                paired=fp[fp['change'].notna()] if 'change' in fp else pd.DataFrame()
                if not paired.empty:
                    paired=paired[paired.baseline!='REST_FULL']
                    summary=paired.groupby(['baseline','change']).size().unstack(fill_value=0).reset_index().rename(columns={'baseline':'method_id'})
                    if not summary.empty:
                        for key in ('new_fp','removed_fp','unchanged_fp'):
                            if key not in summary:summary[key]=0
                        fig,axs=self.new(rows=len(summary));bars(axs[0,0],summary,['removed_fp','new_fp','unchanged_fp'],['Removed FP','New FP','Unchanged FP'],'Number of paired false-positive events')
                        self.save(fig,'09b_restoration_paired_fp','Actual FP correspondences under the configured time/distance pairing rules. Aggregate FP differences are not interpreted as event correspondences.',summary)
                else:self.skip('09b_restoration_paired_fp','FP pairing was not configured; only audit exported')
    def timing(self):
        df=self.table('runtime_summary.csv');summary=runtime_table(df)
        if not summary.empty:
            summary=summary.assign(method_id=summary.stage.astype(str)+' ['+summary.device.astype(str)+', '+summary.unit.astype(str)+']')
            fig,axs=self.new(rows=len(summary));bars(axs[0,0],summary,['total_s'],['Stage total'],'Measured stage time (s)')
            self.save(fig,'10_runtime_stage_totals','Stage times summed by stage, device, unit and cache mode. Nested feature_bundle includes raw_features: stage totals must not be summed into an end-to-end total.',summary)
            fig,axs=self.new(rows=len(summary));bars(axs[0,0],summary,['weighted_ms_per_unit'],['Time per declared unit'],'Weighted time (ms / declared unit)')
            self.save(fig,'11_runtime_per_unit','sum(total_s) / sum(N_units), multiplied by 1000. Unit is explicitly retained for every row; tracks, frames and ROI frames are not interchangeable.',summary)
        repeats=self.table('runtime_repetitions.csv')
        if not repeats.empty:
            vals=pd.to_numeric(repeats.elapsed_s,errors='raise');fig,axs=self.new(rows=4);ax=axs[0,0]
            ax.plot(repeats['repeat'],vals,'o-',color=COLORS[0]);ax.axhline(vals.mean(),color='#D55E00',ls='--',label=f'Mean = {vals.mean():.3f} s');ax.set_xlabel('Measurement repetition');ax.set_ylabel('Elapsed time (s)');ax.set_xticks(repeats['repeat']);ax.grid(alpha=.2);ax.legend(frameon=False)
            self.save(fig,'12_runtime_repetitions',f'Actual measurement repetitions only, N={len(vals)}. No invented confidence interval. Input hashing and source snapshots are outside this elapsed clock; warm cache/model reuse follow the benchmark configuration.',repeats)
    def overview(self,df):
        if df.empty:return
        fig,axs=self.new(n=2,rows=len(df));bars(axs[0,0],df,['precision','recall','f1'],['Precision','Recall','F1'],'Score',True)
        common=self.table('S07_common_tp.csv' if (self.run/'S07_common_tp.csv').exists() else 'common_tp.csv')
        if not common.empty:bars(axs[0,1],ordered(common),['onset_mae'],['Common-TP MAE'],'Common-TP onset error (frames)')
        else:axs[0,1].text(.5,.5,'Common-TP onset data unavailable',transform=axs[0,1].transAxes,ha='center');axs[0,1].set_axis_off()
        self.save(fig,'00_performance_onset_overview','A compact per-run comparison of classification/detection performance and common-TP onset localization. No cross-run cohort pooling.')
    def run_all(self):
        self.out.mkdir(parents=True,exist_ok=True)
        df=ordered(self.table(TABLES.get(self.meta['family'],'_no_metrics.csv')))
        self.metric_figures(df)
        if 'overview' in self.args.groups:self.overview(df)
        self.changes()
        if 'runtime' in self.args.groups:self.timing()
        statuses=self.table('experiment_status.csv')
        if not statuses.empty:write_csv(self.out/'experiment_status.csv',statuses)
        (self.out/'figure_captions.md').write_text('\n'.join(self.captions),encoding='utf-8')
        for name in ('run_meta.json','resolved_config.json'):
            path=self.run/name
            if path.is_file():self.sources[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        return {'run_id':self.meta['run_id'],'family':self.meta['family'],'status':self.meta['status'],'synthetic':bool(self.meta.get('synthetic')),'evaluation_scope':self.cfg.get('evaluation_mode','configured_reference'),
                'source_run':str(self.run.resolve()),'source_hashes':self.sources,'figures':self.files,'skipped':self.skipped}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True,help='One registered run or a parent directory containing runs')
    p.add_argument('--output','--output-dir',type=Path,required=True)
    p.add_argument('--families',help='Comma-separated families; default all')
    p.add_argument('--run-id',action='append',default=[],help='Repeat to select exact run IDs')
    p.add_argument('--groups',default=','.join(GROUPS));p.add_argument('--formats',default='png,pdf')
    p.add_argument('--dpi',type=int,default=400);p.add_argument('--font-size',type=float,default=14)
    p.add_argument('--include-synthetic',action='store_true');p.add_argument('--include-partial',action='store_true')
    p.add_argument('--baseline',help='Optional baseline for signed within-run effects')
    p.add_argument('--label',default='',help='Extra provenance footer, e.g. INTERFACE VALIDATION ONLY')
    p.add_argument('--clean-figures',action='store_true',help='Omit provenance footer; retained in captions/manifest; synthetic watermark remains')
    args=p.parse_args(argv);args.groups=set(args.groups.split(','))
    if args.groups-set(GROUPS):p.error('Unknown figure groups')
    if not set(args.formats.split(','))<={'png','pdf','svg','tif','tiff'}:p.error('Supported formats: png,pdf,svg,tif,tiff')
    if args.font_size<6 or args.dpi<72:p.error('font-size >=6 and dpi >=72 required')
    if not args.runs.is_dir():p.error('Input run directory does not exist')
    if args.output.exists():p.error('Output directory exists; use a new directory to preserve figures')
    families=set(args.families.split(',')) if args.families else None
    entries=[];excluded=[]
    paths=sorted(args.runs.rglob('run_meta.json'))
    for path in paths:
        meta=json.loads(path.read_text(encoding='utf-8'))
        if args.run_id and meta['run_id'] not in args.run_id:continue
        if families and meta['family'] not in families:continue
        if meta.get('synthetic') and not args.include_synthetic:excluded.append({'path':str(path),'reason':'synthetic'});continue
        if meta['status']!='completed' and not (args.include_partial and meta['status']=='partial'):excluded.append({'path':str(path),'reason':meta['status']});continue
        entries.append((path.parent,meta))
    if args.run_id and set(args.run_id)-{m['run_id'] for _,m in entries}:p.error('Requested run ID missing or excluded by filters')
    if not entries:p.error('No eligible registered runs; synthetic/partial runs require explicit switches')
    keys=[(m['family'],m['run_id']) for _,m in entries]
    if len(keys)!=len(set(keys)):p.error('Duplicate run IDs in input tree; select a narrower input root')
    args.output.mkdir(parents=True);results=[]
    for run,meta in entries:
        result=Plotter(args,run,meta).run_all();results.append(result);print(meta['family'],meta['run_id'],len(result['figures']),'files',flush=True)
    write_json(args.output/'plot_manifest.json',{'style_source':'latest_experiment_scripts_20261018/manuscript_main_figures_mm.py','style':'Times New Roman, inward ticks, white background, caption-only panels, legacy heatmap encoding','plot_settings':vars(args),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'style_sha256':hashlib.sha256(Path(load_legacy('manuscript_main_figures_mm.py').__file__).read_bytes()).hexdigest(),'policy':'Per-run separation; NA preserved; Common-TP onset only; no cross-cohort pooling','runs':results,'excluded_runs':excluded})
    print(args.output.resolve())

if __name__=='__main__':main()
