"""One bounded, preregistered v8 model adjudication; original evidence read-only."""
from pathlib import Path
import sys,json,hashlib,importlib.util,importlib.metadata,platform
from datetime import datetime,timezone
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent;SUB=OUT.parent;ROOT=SUB.parents[1]
G4=SUB/'audit_v7_2_g4_g5';REC=SUB/'controlled_recovery_v7_1'
spec=importlib.util.spec_from_file_location('frozen_g4_access',G4/'audit_common.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
FRAMES=old.FRAMES;TRANS=old.TRANS;RUNS=old.RUNS
NUISANCE=['baseline_age_c','female','urban','education_primary','education_middle','education_high','partnered','smoking','drinking','bmi_c','disease_count_c','memory_c','cesd9_c']
SEEDS=[2026102401,2026102402];CATEGORIES=['recovery','deterioration','death']

def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))
def save(p,rows):
    p=Path(p)
    if p.exists():raise FileExistsError(p)
    p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(p,index=False,encoding='utf-8-sig',float_format='%.17g')
def md(p,text):
    with Path(p).open('x',encoding='utf-8') as f:f.write(text.rstrip()+'\n')
def category(path):return 'death' if path[1]==3 else ('recovery' if path[1]<path[0] else 'deterioration')
def mapping(frame,kind):
    if kind=='original':
        names=old.model(frame,1.)[0]['parameter_names'];return np.eye(225),[f'{i}->{j}:{v}' for i,j in TRANS for v in names],names
    names=['intercept','age_band_55','age_band_65','age_band_75','age_band_85','activity','sleep_linear2h','sleep_quadratic2h','sleep_linear2h_activity']+NUISANCE
    par=[f'{i}->{j}:{v}' for i,j in TRANS for v in ['intercept','activity','sleep_linear2h','sleep_quadratic2h','sleep_linear2h_activity']]
    age=['age_band_55','age_band_65','age_band_75','age_band_85'] if frame=='CHARLS' else ['age_band_75','age_band_85']
    par += [f'{cat}:{v}' for cat in CATEGORIES for v in age+NUISANCE]
    mat=np.zeros((9*len(names),len(par)))
    for k,path in enumerate(TRANS):
        for v in ['intercept','activity','sleep_linear2h','sleep_quadratic2h','sleep_linear2h_activity']:mat[k*len(names)+names.index(v),par.index(f'{path[0]}->{path[1]}:{v}')]=1
        for v in age+NUISANCE:mat[k*len(names)+names.index(v),par.index(f'{category(path)}:{v}')]=1
    return np.ascontiguousarray(mat),par,names
def arrays(df,frame,kind):
    _,_,names=mapping(frame,kind);x=np.zeros((len(df),len(names)));x[:,0]=1
    if kind=='original':x[:,5:]=df[names[5:]].to_numpy(float)
    else:
        x[:,5]=df.activity.to_numpy(float);z=(df.sleep.to_numpy(float)-7.)/2.
        x[:,6]=z;x[:,7]=z*z;x[:,8]=z*x[:,5];x[:,9:]=df[NUISANCE].to_numpy(float)
    dt=np.zeros((len(df),5));bands=np.zeros((len(df),5),dtype=np.int64)
    for n,(lo,hi) in enumerate(zip(df.age_enter,df.age_exit)):
        br=[lo]+[c for c in [55.,65.,75.,85.] if lo<c<hi]+[hi]
        for s,(a,b) in enumerate(zip(br[:-1],br[1:])):dt[n,s]=b-a;bands[n,s]=np.searchsorted([55.,65.,75.,85.],(a+b)/2,side='right')
    args=(np.ascontiguousarray(x),dt,bands,df.i.to_numpy(np.int64),df.j.to_numpy(np.int64),df.weight.to_numpy(float))
    assert np.isfinite(x).all() and np.isfinite(args[-1]).all() and (args[-1]>0).all()
    return args
def penalty(par,lam):
    center=np.zeros(len(par));prec=np.full(len(par),lam)
    for i,name in enumerate(par):
        if name.endswith(':intercept'):center[i]=np.log(.1);prec[i]*=.25
    return center,prec
def empirical_start(df,par):
    theta=np.zeros(len(par))
    for i,j in TRANS:
        n=int((df.i.eq(i)&df.j.eq(j)).sum());den=float(df.loc[df.i.eq(i),'interval_years'].sum())
        if n<=0 or den<=0:raise ValueError('No empirical start support; do not invent counts')
        theta[par.index(f'{i}->{j}:intercept')]=np.log(n/den)
    return theta
def independent_start(par,seed):
    # Identical generative rule at every lambda; no data-dependent lucky seed.
    rng=np.random.default_rng(seed);out=rng.normal(0.,.15,len(par))
    for i,name in enumerate(par):
        if name.endswith(':intercept'):out[i]=np.log(.1)+rng.normal(0.,.5)
    return out
def cv_frames(df,frame):
    ids=sorted(df.person_id.unique());test={pid for pid in ids if int(hashlib.sha256(('20261011|'+frame+'|'+pid).encode()).hexdigest()[:16],16)%5==0}
    train=df[~df.person_id.isin(test)].copy();valid=df[df.person_id.isin(test)].copy()
    assert len(train)+len(valid)==len(df) and not set(train.person_id)&set(valid.person_id)
    unique=train.sort_values(['person_id','wave_start'],kind='stable').drop_duplicates('person_id')
    transforms={}
    for original,c in [('baseline_age','baseline_age_c'),('bmi','bmi_c'),('disease_count','disease_count_c'),('memory','memory_c'),('cesd9','cesd9_c')]:
        mu=float(unique[original].mean());sd=float(unique[original].std(ddof=0));assert sd>0
        train[c]=(train[original]-mu)/sd;valid[c]=(valid[original]-mu)/sd;transforms[c]=dict(mean=mu,sd=sd)
    meanweight=float(unique.subject_weight_raw.mean());assert meanweight>0
    train['weight']=train.subject_weight_raw/meanweight;valid['weight']=valid.subject_weight_raw/meanweight
    return train,valid,transforms
