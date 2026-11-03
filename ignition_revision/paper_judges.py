"""Fixed-ROI adaptations. All thresholds are development parameters, not fitted on GT."""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from .common import Blocked, resolve


def gray(row,cfg):
    if all(k in row for k in ('crop_left','crop_top','crop_right_exclusive','crop_bottom_exclusive')):
        from .roi_io import read_roi
        return np.asarray(Image.fromarray(read_roi(row,cfg)).convert('L'),dtype=np.float64)/255.
    with Image.open(resolve(cfg,row.roi_path)) as im:
        # Explicit 8-bit luminance; high bit-depth sources must be converted upstream.
        if im.mode.startswith(('I','F')):raise Blocked('Supply a documented 8-bit ROI conversion for high-bit-depth images')
        return np.asarray(im.convert('L'),dtype=np.float64)/255.


def center(row,shape):
    x=row.get('center_roi_x',np.nan);y=row.get('center_roi_y',np.nan)
    if not np.isfinite(x) or not np.isfinite(y):
        raise Blocked('Known particle center requires center_roi_x/center_roi_y; geometric ROI center is not silently substituted')
    if not (0<=x<shape[1] and 0<=y<shape[0]):raise ValueError('Particle center outside ROI')
    return float(x),float(y)


def consecutive(mask,frames,n):
    if n<1:raise ValueError('min_run must be positive')
    result=[];start=None
    for i,ok in enumerate(mask):
        if not ok or (i and frames[i]!=frames[i-1]+1):
            if start is not None and i-start>=n:result.append(start)
            start=None
        if ok and start is None:start=i
    if start is not None and len(mask)-start>=n:result.append(start)
    return result


def normalized(a,s):
    mode=s['normalization']
    if mode=='unit':return a
    if mode=='background_ratio':
        # A development-selected fixed background avoids using the manual onset to select cold frames.
        b=float(s['background_level_unit'])
        if not 0<b<=1:raise ValueError('background_level_unit must be in (0,1]')
        return a/b
    raise ValueError('normalization must be unit or background_ratio')


def particle_mask(row,a,s,cfg):
    mode=s['mask_mode']
    if mode=='provided':
        if not isinstance(row.get('mask_path'),str):raise Blocked('Missing independent particle mask_path')
        with Image.open(resolve(cfg,row.mask_path)) as im:m=np.asarray(im.convert('L'))>0
        if m.shape!=a.shape:raise ValueError('Mask and ROI size differ')
    elif mode=='center_component':
        # Declared automatic segmentation adaptation, not SAM or the original paper masks.
        x,y=center(row,a.shape);signal=normalized(a,s)
        m=(signal>=float(s['mask_threshold'])) if s.get('mask_polarity','bright')=='bright' else (signal<=float(s['mask_threshold']))
        _,labels=cv2.connectedComponents(m.astype('uint8'),connectivity=8)
        k=labels[int(round(min(y,a.shape[0]-1))),int(round(min(x,a.shape[1]-1)))]
        m=labels==k if k else np.zeros_like(m)
    else:raise ValueError('mask_mode must be provided or center_component')
    return m


def stage_fit(signal,frames,s):
    """Offline continuous low plateau / linear rise / high plateau, bounded grid search.

    y = baseline + amplitude * clip((t-a)/(b-a),0,1). The two linear
    coefficients are solved analytically. Breakpoints are physical frame indices.
    The BIC comparison includes a constant and an unrestricted straight line.
    """
    values=np.asarray(signal,float);frames=np.asarray(frames,int)
    minima=[int(s[k]) for k in ('min_low','min_rise','min_high')]
    if min(minima)<2:raise ValueError('Stage segments require at least two samples')
    if float(s['min_amplitude'])<=0 or float(s['min_slope_per_frame'])<=0 or float(s['max_rmse'])<=0 or float(s['min_bic_improvement'])<0:raise ValueError('Require positive stage amplitude/slope/RMSE limits and nonnegative BIC improvement')
    candidates=[];fits=[]
    cuts=np.r_[0,np.flatnonzero(np.diff(frames)!=1)+1,len(frames)]
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        y=values[lo:hi];t=frames[lo:hi].astype(float);n=len(y)
        if n<sum(minima):continue
        if not np.isfinite(y).all():
            fits.append({'start':int(lo),'reason':'missing_mask_signal'});continue
        eps=max(1e-12,float(np.var(y))*1e-10)
        def bic(sse,k):return n*np.log(max(float(sse)/n,eps))+k*np.log(n)
        null_bic=bic(np.sum((y-y.mean())**2),1)
        coef=np.polyfit(t-t[0],y,1);line_bic=bic(np.sum((y-np.polyval(coef,t-t[0]))**2),2)
        sy=np.r_[0.,np.cumsum(y)];siy=np.r_[0.,np.cumsum(np.arange(n)*y)];sumy=float(sy[-1]);sumyy=float(y@y)
        best=None;step=int(s.get('boundary_step',1))
        if step<1:raise ValueError('boundary_step must be positive')
        for a in range(minima[0],n-minima[1]-minima[2]+1,step):
            for b in range(a+minima[1],n-minima[2]+1,step):
                # Contiguous frames make the clipped ramp's sufficient statistics analytic.
                # O(1) per breakpoint pair instead of constructing an O(n) fitted vector.
                m=b-a;sumz=(m-1)/2.+n-b
                sumz2=(m-1)*(2*m-1)/(6.*m)+n-b
                sumzy=(siy[b]-siy[a]-a*(sy[b]-sy[a]))/m+sy[n]-sy[b]
                den=sumz2-sumz*sumz/n
                amp=float((sumzy-sumz*sumy/n)/den);base=float((sumy-amp*sumz)/n)
                slope=amp/(t[b]-t[a])
                if amp<float(s['min_amplitude']) or slope<float(s['min_slope_per_frame']):continue
                sse=max(0.,sumyy-base*sumy-amp*sumzy);score=bic(sse,4)
                if best is None or score<best['bic']:
                    best={'onset_idx':int(lo+a),'combustion_idx':int(lo+b),'onset_frame':int(t[a]),'combustion_frame':int(t[b]),'baseline':base,'amplitude':amp,'slope':slope,'bic':float(score),'rmse':float(np.sqrt(sse/n))}
        if best is None:
            fits.append({'start':int(lo),'reason':'no_positive_three_stage_fit'});continue
        best.update({'null_bic':float(null_bic),'linear_bic':float(line_bic),'bic_improvement':float(min(null_bic,line_bic)-best['bic'])})
        accepted=best['bic_improvement']>=float(s['min_bic_improvement']) and best['rmse']<=float(s['max_rmse'])
        best['accepted']=bool(accepted);best['decision_available_idx']=int(hi-1);fits.append(best)
        if accepted:candidates.append(best['onset_idx'])
    return candidates,{'signal':values.tolist(),'fits':fits,'future_context':True,'confirmation_indices':{str(f['onset_idx']):f['decision_available_idx'] for f in fits if f.get('accepted')}}


def propose_paper(df,method,s,cfg):
    frames=df.frame_idx.to_numpy(int)
    required={'sas_paper':['normalization','pixel_threshold','mean_threshold','area_unit','area_threshold'],'dog_paper':['sigmas','sigma_ratio','polarity','response_threshold'],'cnf_roi_stage':['min_low','min_rise','min_high','min_amplitude','min_slope_per_frame','min_bic_improvement','max_rmse']}[method]
    if method=='sas_paper' and s.get('normalization')=='background_ratio':required.append('background_level_unit')
    if method=='sas_paper' and s.get('area_unit')=='mm2':required.append('pixels_per_mm')
    if method=='cnf_roi_stage' and not s.get('signal_column'):
        required.append('mask_mode')
        if s.get('mask_mode')=='center_component':required.append('mask_threshold')
        if s.get('normalization')=='background_ratio':required.append('background_level_unit')
    for key in required:
        if s.get(key) is None:raise Blocked(f'{method}: configure development-selected {key}')
    if method=='cnf_roi_stage' and s.get('signal_column'):
        return stage_fit(df[s['signal_column']].to_numpy(float),frames,s)
    signals=[];areas=[];scales=[]
    for _,row in df.iterrows():
        a=gray(row,cfg)
        if method=='sas_paper':
            z=normalized(a,s);signals.append(float(z.mean()))
            binary=(z>=float(s['pixel_threshold'])).astype('uint8')
            count,_,stats,_=cv2.connectedComponentsWithStats(binary,connectivity=8)
            area=float(stats[1:,cv2.CC_STAT_AREA].max()) if count>1 else 0.
            unit=s['area_unit']
            if unit=='mm2':
                ppm=float(s['pixels_per_mm'])
                if ppm<=0:raise ValueError('pixels_per_mm must be positive')
                area/=ppm**2
            elif unit!='px2':raise ValueError('area_unit must be mm2 or px2')
            areas.append(area)
        elif method=='dog_paper':
            x,y=center(row,a.shape);k=float(s['sigma_ratio'])
            if k<=1 or s['polarity'] not in ('bright','dark'):raise ValueError('Invalid DoG scale ratio or polarity')
            best=(-np.inf,None)
            for sigma in s['sigmas']:
                if sigma<=0:raise ValueError('DoG sigma must be positive')
                # Sign is stated explicitly: positive response for a bright blob.
                dog=(cv2.GaussianBlur(a,(0,0),sigma)-cv2.GaussianBlur(a,(0,0),k*sigma))/(k-1)
                if s['polarity']=='dark':dog=-dog
                v=float(cv2.getRectSubPix(dog.astype('float32'),(1,1),(x,y))[0,0])
                if v>best[0]:best=(v,float(sigma))
            if best[1] is None:raise ValueError('sigmas cannot be empty')
            signals.append(best[0]);scales.append(best[1])
        elif method=='cnf_roi_stage':
            mask=particle_mask(row,a,s,cfg)
            signals.append(float(np.median(a[mask])) if mask.any() else float('nan'))
            areas.append(int(mask.sum()))
        else:raise ValueError(method)
    if method=='cnf_roi_stage':
        indices,trace=stage_fit(signals,frames,s);trace['mask_area_px']=areas;trace['mask_mode']=s['mask_mode'];return indices,trace
    if method=='sas_paper':mask=(np.array(signals)>=s['mean_threshold'])&(np.array(areas)>=s['area_threshold'])
    else:mask=np.array(signals)>=s['response_threshold']
    trace={'signal':signals,'largest_connected_area':areas,'selected_sigma':scales,'future_context':False,'polarity':s.get('polarity'),'normalization':s.get('normalization','unit')}
    return consecutive(mask,frames,int(s.get('min_run',1))),trace
