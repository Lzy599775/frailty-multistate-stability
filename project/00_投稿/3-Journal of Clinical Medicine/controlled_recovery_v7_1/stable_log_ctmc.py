"""Diagnostic prototype: CTMC uniformization + log-domain products + derivatives.

No optimizer, clipping, probability floor, resampling, or numerical fallback.
The diagonal of the short-time Markov matrix is evaluated using its analytic
row-sum identity; it is not a repair or renormalization of a failed exponential.
"""
import numpy as np
from numba import njit

TRANS = np.array([(r,j) for r in range(3) for j in range(4) if r!=j],dtype=np.int64)
SERIES_RELATIVE_TAIL = 1e-18
MAX_SERIES_TERMS = 256
MAX_SQUARINGS = 1000


@njit(cache=False)
def logadd(a,b):
    if a == -np.inf:return b
    if b == -np.inf:return a
    if a < b:a,b=b,a
    return a+np.log1p(np.exp(b-a))


@njit(cache=False)
def log_eye():
    out=np.full((4,4),-np.inf)
    for i in range(4):out[i,i]=0.
    return out


@njit(cache=False)
def log_product(a,b):
    out=np.full((4,4),-np.inf)
    for i in range(4):
        for j in range(4):
            v=-np.inf
            for k in range(4):v=logadd(v,a[i,k]+b[k,j])
            out[i,j]=v
    return out


@njit(cache=False)
def log_square_score(lp,score,need_gradient):
    out=log_product(lp,lp)
    tangent=np.zeros((4,4,9))
    if need_gradient:
        for i in range(4):
            for j in range(4):
                if out[i,j] == -np.inf:continue
                for k in range(4):
                    term=lp[i,k]+lp[k,j]
                    if term == -np.inf:continue
                    weight=np.exp(term-out[i,j])
                    for z in range(9):tangent[i,j,z]+=weight*(score[i,k,z]+score[k,j,z])
    return out,tangent


@njit(cache=False)
def segment_log_probability(rates_dt,need_gradient=True):
    """d log(P_ij) / d eta_k with eta=log(unclipped transition intensity)."""
    if not np.all(np.isfinite(rates_dt)) or np.any(rates_dt<=0):
        raise ValueError('Prototype requires finite strictly positive rate*duration; no fallback')
    exits=np.zeros(3)
    for k in range(9):exits[TRANS[k,0]]+=rates_dt[k]
    maximum=np.max(exits)
    if not np.isfinite(maximum):raise ValueError('Nonfinite total departure rate')
    exponent=int(np.ceil(np.log2(maximum)))
    nu=2.**exponent
    if not np.isfinite(nu) or nu<=0:raise ValueError('Uniformization scale not representable')
    scale=max(0,exponent+1)
    if scale>MAX_SQUARINGS:raise ValueError('Prototype squaring resource limit exceeded')
    mu=nu/(2.**scale)
    b=np.eye(4)
    db=np.zeros((9,4,4))
    for k in range(9):
        r,j=TRANS[k]
        value=rates_dt[k]/nu
        b[r,j]=value
        db[k,r,j]=value
        db[k,r,r]=-value
    for r in range(3):b[r,r]=1.-exits[r]/nu
    if np.any(b<0):raise ValueError('Uniformization jump matrix has a negative entry; no clipping')
    term=np.eye(4)
    dterm=np.zeros((9,4,4))
    coefficient=np.exp(-mu)
    probability=coefficient*term
    derivative=np.zeros((9,4,4))
    norm_b=np.max(np.sum(np.abs(b),axis=1))
    g=mu*max(1.,norm_b)
    scalar_bound=np.exp(-mu)
    converged=False
    terms=0
    for order in range(1,MAX_SERIES_TERMS+1):
        next_term=term@b
        if need_gradient:
            next_d=np.zeros((9,4,4))
            for z in range(9):
                next_d[z]=dterm[z]@b+term@db[z]
            dterm=next_d
        term=next_term
        coefficient*=mu/order
        probability+=coefficient*term
        if need_gradient:derivative+=coefficient*dterm
        scalar_bound*=g/order
        min_off=np.inf
        for r in range(3):
            for j in range(4):
                if r!=j:min_off=min(min_off,probability[r,j])
        terms=order
        if order+2>g and order>=3:
            tail=scalar_bound*g/(order+1)/(1.-g/(order+2))
            if min_off>0 and tail <= SERIES_RELATIVE_TAIL*min_off:
                converged=True
                break
    if not converged:raise ValueError('Uniformization tail criterion not met; no retry or looser tolerance')
    lp=np.full((4,4),-np.inf)
    score=np.zeros((4,4,9))
    for r in range(3):
        off=0.
        for j in range(4):
            if r==j:continue
            p=probability[r,j]
            if p<=0 or not np.isfinite(p):raise ValueError('Short-time off-diagonal probability not representable')
            off+=p
            lp[r,j]=np.log(p)
            if need_gradient:
                for z in range(9):score[r,j,z]=derivative[z,r,j]/p
        if not 0<=off<1:raise ValueError('Short-time row identity outside its valid domain')
        lp[r,r]=np.log1p(-off)
        if need_gradient:
            for z in range(9):
                ds=0.
                for j in range(4):
                    if r!=j:ds+=derivative[z,r,j]
                score[r,r,z]=-ds/(1.-off)
    lp[3,3]=0.
    for _ in range(scale):lp,score=log_square_score(lp,score,need_gradient)
    if np.any(np.isnan(lp)) or np.any(~np.isfinite(score)):
        raise ValueError('Nonfinite log probability or score in prototype')
    return lp,score,terms,scale


@njit(cache=False)
def data_evaluate(theta,x,dt,bands,si,sj,weights,need_gradient=True):
    beta=theta.reshape((9,x.shape[1]))
    scores=np.zeros((len(x),9,x.shape[1]))
    all_lp=np.zeros((len(x),4,4))
    observed_logp=np.zeros(len(x))
    counts=np.zeros((len(x),2),dtype=np.int64)
    value=0.
    compensation=0.
    for n in range(len(x)):
        ns=np.sum(dt[n]>0)
        designs=np.zeros((ns,x.shape[1]))
        segment_lp=np.zeros((ns,4,4))
        segment_score=np.zeros((ns,4,4,9))
        prefix=np.zeros((ns+1,4,4));prefix[0]=log_eye()
        suffix=np.zeros((ns+1,4,4));suffix[ns]=log_eye()
        for s in range(ns):
            designs[s]=x[n]
            if bands[n,s]>0:designs[s,bands[n,s]]=1.
            rates=np.exp(beta@designs[s])
            lp,tangent,terms,squarings=segment_log_probability(rates*dt[n,s],need_gradient)
            segment_lp[s]=lp;segment_score[s]=tangent
            prefix[s+1]=log_product(prefix[s],lp)
            counts[n,0]=max(counts[n,0],terms)
            counts[n,1]=max(counts[n,1],squarings)
        all_lp[n]=prefix[ns]
        observed=prefix[ns,si[n],sj[n]]
        if not np.isfinite(observed):raise ValueError('Observed log probability not finite')
        observed_logp[n]=observed
        contribution=-weights[n]*observed
        corrected=contribution-compensation
        next_value=value+corrected
        compensation=(next_value-value)-corrected
        value=next_value
        if need_gradient:
            for s in range(ns-1,-1,-1):suffix[s]=log_product(segment_lp[s],suffix[s+1])
            for s in range(ns):
                derivative=np.zeros(9)
                for a in range(4):
                    for b in range(4):
                        logterm=prefix[s,si[n],a]+segment_lp[s,a,b]+suffix[s+1,b,sj[n]]
                        if logterm == -np.inf:continue
                        conditional_weight=np.exp(logterm-observed)
                        for k in range(9):derivative[k]+=conditional_weight*segment_score[s,a,b,k]
                for k in range(9):scores[n,k]-=weights[n]*derivative[k]*designs[s]
    if not np.isfinite(value) or not np.all(np.isfinite(scores)):
        raise ValueError('Nonfinite likelihood or score; no fallback')
    return value,np.sum(scores,axis=0).ravel(),scores.reshape((len(x),theta.size)),all_lp,observed_logp,counts


def evaluate(theta,args,strength,need_gradient=True):
    value,gradient,scores,lp,observed,counts=data_evaluate(theta,*args,need_gradient)
    center=np.zeros_like(theta);center[::args[0].shape[1]]=np.log(.1)
    precision=np.full_like(theta,strength);precision[::args[0].shape[1]]*=.25
    delta=theta-center
    penalty=.5*np.dot(precision*delta,delta)
    return value+penalty,gradient+precision*delta,scores,lp,observed,counts
