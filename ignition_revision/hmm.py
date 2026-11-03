"""Offline two-state Gaussian HMM, fixed cold->hot transitions, exact EM.

With start=(1,0), A=[[p,1-p],[0,1]], every allowed path is all-cold
or one cold prefix followed by one hot suffix. Enumerating these N paths
is an exact forward/backward calculation, not an unconstrained change score.
No ground-truth labels are read. Missing physical frames split sequences.
"""
import numpy as np
from scipy.special import logsumexp
from .common import digest, require_columns

DEFAULTS={'cold_stay':.95,'max_iter':100,'tol':1e-5,'variance_floor':.05,
          'initial_fraction':.2,'min_bic_gain':0.,'min_thermal_increase':.25}


def path_posterior(emissions,cold_stay):
    """Return log path weights for k=1..N, k=N denotes no transition."""
    n=len(emissions)
    cold=np.r_[0.,np.cumsum(emissions[:,0])]
    hot=np.r_[0.,np.cumsum(emissions[:,1])]
    k=np.arange(1,n+1)
    scores=cold[k]+hot[n]-hot[k]+(k-1)*np.log(cold_stay)
    scores[:-1]+=np.log1p(-cold_stay)
    ll=float(logsumexp(scores));weights=np.exp(scores-ll)
    gamma_hot=np.r_[0.,np.cumsum(weights[:-1])]
    return scores,ll,np.column_stack([1-gamma_hot,gamma_hot])


def emissions(x,means,variances):
    return -.5*(np.log(2*np.pi*variances)[None,:,:]+(x[:,None,:]-means[None,:,:])**2/variances[None,:,:]).sum(axis=2)


def fit_segment(x,settings):
    s={**DEFAULTS,**settings};n,d=x.shape
    p=float(s['cold_stay']);floor=float(s['variance_floor']);fraction=float(s['initial_fraction'])
    if not 0<p<1 or floor<=0 or not 0<fraction<=.5 or int(s['max_iter'])<1 or float(s['tol'])<=0:
        raise ValueError('Invalid HMM transition, variance, initialization or convergence settings')
    count=max(1,int(np.ceil(n*fraction)))
    means=np.stack([x[:count].mean(0),x[-count:].mean(0)])
    variances=np.tile(np.maximum(x.var(0),floor),(2,1));previous=-np.inf
    for iteration in range(int(s['max_iter'])):
        scores,ll,gamma=path_posterior(emissions(x,means,variances),p)
        if np.isfinite(previous) and abs(ll-previous)<=s['tol']*(1+abs(previous)):break
        previous=ll;mass=gamma.sum(0)
        if np.any(mass<1e-8):break
        means=(gamma.T@x)/mass[:,None]
        variances=np.maximum(np.stack([(gamma[:,j,None]*(x-means[j])**2).sum(0)/mass[j] for j in (0,1)]),floor)
    scores,ll,gamma=path_posterior(emissions(x,means,variances),p)
    null_var=np.maximum(x.var(0),floor)
    null_ll=float(emissions(x,x.mean(0)[None,:],null_var[None,:]).sum())
    # Fixed transition/start parameters: null 2D, HMM 4D free emission parameters.
    bic_gain=2*(ll-null_ll)-2*d*np.log(n)
    k=int(np.argmax(scores))+1
    return k,{'N':n,'iterations':iteration+1,'log_likelihood':ll,'null_log_likelihood':null_ll,
              'bic_gain':float(bic_gain),'means':means.tolist(),'variances':variances.tolist(),
              'hot_posterior':gamma[:,1].tolist(),'map_transition_sample':k if k<n else None}


def hmm_candidates(df,mod,config,settings):
    s={**DEFAULTS,**settings.get('hmm',{})}
    columns=settings.get('feature_columns',list(mod.BASE_GMM_FEATURES)+(list(mod.DIFF_GMM_FEATURES) if config.use_diff_features else []))
    require_columns(df,columns,'HMM features')
    thermal=[i for i,c in enumerate(columns) if c in mod.THERMAL_FEATURE_NAMES]
    if not columns or not thermal:raise ValueError('HMM requires named thermal ordering features')
    x=df[columns].to_numpy(float)
    if not np.isfinite(x).all():raise ValueError('Nonfinite HMM input')
    med=np.median(x,0);mad=np.median(abs(x-med),0)+1e-6;x=np.clip((x-med)/mad,-6,6)
    frames=df.frame_idx.to_numpy(int)
    if len(frames)>1 and np.any(np.diff(frames)<=0):raise ValueError('Sorted unique frames required')
    starts=np.r_[0,np.where(np.diff(frames)!=1)[0]+1];stops=np.r_[starts[1:],len(df)]
    candidates=[];segments=[];ranks=np.zeros(len(df),int)
    for lo,hi in zip(starts,stops):
        if hi-lo<config.min_cold_run+config.min_hot_run:
            segments.append({'start':int(lo),'stop':int(hi),'reason':'insufficient_contiguous_support'});continue
        k,fit=fit_segment(x[lo:hi],s);fit.update(start=int(lo),stop=int(hi))
        means=np.asarray(fit['means']);increase=float(np.mean(means[1,thermal]-means[0,thermal]))
        fit['thermal_increase']=increase
        accepted=(config.min_cold_run<=k<=hi-lo-config.min_hot_run and
                  fit['bic_gain']>=s['min_bic_gain'] and increase>=s['min_thermal_increase'])
        fit['accepted']=bool(accepted);segments.append(fit)
        if k<hi-lo:ranks[lo+k:hi]=1
        if accepted and (not config.require_all_prior_cold or not np.any(ranks[:lo+k-config.min_cold_run])):
            candidates.append(int(lo+k))
    return candidates,{'feature_columns':columns,'feature_schema_hash':digest(columns),'settings':s,
        'segments':segments,'raw_rank':ranks.tolist(),'future_context':True,
        'state_model':'two-state diagonal-Gaussian HMM; fixed start cold; cold->hot only',
        'normalization':'whole-track median/MAD clip6; same as GMM',
        'persistence':'MAP cold/hot run lengths checked explicitly; not implied by stay probability',
        'confirmation_indices':{str(i):int(i+config.min_hot_run-1) for i in candidates}}
