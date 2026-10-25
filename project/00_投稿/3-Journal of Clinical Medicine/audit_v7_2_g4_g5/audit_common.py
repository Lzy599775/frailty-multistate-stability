"""Read-only frozen evidence access for the authorized v7.2 reconstruction."""
from pathlib import Path
import hashlib, json, sys, platform, importlib.metadata
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from patsy import dmatrix

OUT=Path(__file__).resolve().parent
SUB=OUT.parent
ROOT=SUB.parents[1]
V7=SUB/'audit_v7_fi_corrected/authorized_repair_20261024'
REC=SUB/'controlled_recovery_v7_1'
V6=SUB/'audit_20261023_v6_resolution'
BASE=ROOT/'08_intermediate/analysis_ready_stageA_v1_20261010'
V8=ROOT/'05_outputs/final_v8_20261011'
FRAMES=['CHARLS','2008','2011','2014']
RUNS=[(.5,'sensitivity_lambda_0p5'),(1.,'formal_full_v2'),(2.,'sensitivity_lambda_2p0')]
TRANS=[(i,j) for i in range(3) for j in range(4) if i!=j]
SEEN={}

def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def seen(p):
    p=Path(p)
    if not p.is_file():raise FileNotFoundError('SOURCE_REQUIRED: '+str(p))
    r=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
    if str(p) in SEEN:assert SEEN[str(p)]==r,'Input changed while reading: '+str(p)
    SEEN[str(p)]=r
    return p
def js(p):return json.loads(seen(p).read_text(encoding='utf-8-sig'))
def csv(p,**kwargs):return pd.read_csv(seen(p),low_memory=False,**kwargs)
def save(name,rows):
    p=OUT/name
    if p.exists():raise FileExistsError(p)
    pd.DataFrame(rows).to_csv(p,index=False,encoding='utf-8-sig',float_format='%.17g')
def write(name,o):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))
def md(name,s):
    with (OUT/name).open('x',encoding='utf-8') as f:f.write(s.rstrip()+'\n')
def directory(frame,lam=1.):
    if frame=='CHARLS':return BASE/'stageB_charls_penalized_execution_v8'
    if frame=='2014' and lam!=1:return REC/'models/frame2014'
    return V7/'models'/('frame'+frame)
def prefix(lam):return dict(RUNS)[lam]
def sample(frame,all_intervals=False):
    d=directory(frame)
    if all_intervals:name='charls_model_input_all_intervals_v2.csv' if frame=='CHARLS' else f'clhls_frame{frame}_model_input_all_intervals_v2.csv'
    else:name='charls_model_ready_v2.csv' if frame=='CHARLS' else 'clhls_model_ready_v2.csv'
    return csv(d/name,dtype={'person_id':str})
def model(frame,lam):
    d=directory(frame,lam);stem=prefix(lam)
    fp=d/(stem+'_fit.json');cp=d/(stem+'_cluster_sandwich.npy');ip=d/(stem+'_information.json')
    f=js(fp);info=js(ip);cov=np.load(seen(cp));th=np.asarray(f['theta'],float)
    assert f['optimizer_success'] and f['gradient_inf_total']<=1e-4
    assert f['penalty_strength']==lam and info['penalty_strength']==lam
    assert info['qualified'] and info['minimum_eigenvalue']>0 and info['condition_number']<=1e10
    assert th.shape==(225,) and len(f['parameter_names'])==25 and cov.shape==(225,225)
    assert np.isfinite(th).all() and np.isfinite(cov).all() and np.max(abs(cov-cov.T))<1e-8
    return f,th,cov,info,dict(parameter_path=str(fp),parameter_sha256=sha(fp),covariance_path=str(cp),covariance_sha256=sha(cp),information_path=str(ip),information_sha256=sha(ip),frozen_version='FROZEN_CHARLS' if frame=='CHARLS' else ('V7_1_CONTROLLED' if frame=='2014' and lam!=1 else 'V7_FI_CORRECTED'),numerical_qualification='PASS')
def spline(frame,values):
    knots={'CHARLS':(3.,6.,7.,9.),'2008':(4.,7.,9.,12.),'2011':(4.,7.,8.,12.),'2014':(4.,7.,8.,12.)}[frame]
    x=np.r_[np.asarray(values,float),7.]
    b=np.asarray(dmatrix('cr(x, knots=inner, lower_bound=lower, upper_bound=upper)-1',dict(x=x,inner=knots[1:3],lower=knots[0],upper=knots[3])))
    return (b[:-1]-b[-1])[:,:3]
def contrast(frame,names,path,sleep,activity,refactivity):
    v=np.zeros(len(names));sp=spline(frame,[sleep])[0]
    for z in range(3):
        v[names.index(f'sleep_ns{z+1}')]=sp[z]
        v[names.index(f'sleep_ns{z+1}_activity')]=sp[z]*activity
    v[names.index('activity')]=activity-refactivity
    c=np.zeros(225);k=TRANS.index(path);c[k*25:(k+1)*25]=v
    return c
def estimate(c,th,cov):
    val=float(c@th);var=float(c@cov@c)
    if not np.isfinite(var) or var<0 or (var==0 and np.any(c)):
        return dict(logTIR=val,variance=var,status='NOT_ESTIMABLE',reason='NONPOSITIVE_OR_NONFINITE_VARIANCE')
    se=float(np.sqrt(var))
    with np.errstate(over='ignore',under='ignore'):
        vals=np.exp([val,val-1.96*se,val+1.96*se])
    ok=np.isfinite(vals).all() and (vals>0).all()
    return dict(logTIR=val,variance=var,TIR=vals[0],se=se,ci_lower=vals[1],ci_upper=vals[2],status='PASS' if ok else 'NOT_ESTIMABLE',reason='' if ok else 'EXPONENT_UNREPRESENTABLE',wide_interval_log_width=3.92*se)
def manifest(name):save(name,list(SEEN.values()))
def environment():return dict(utc=utc(),python=sys.version,executable=sys.executable,platform=platform.platform(),packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','patsy','numba','llvmlite','python-docx']})
