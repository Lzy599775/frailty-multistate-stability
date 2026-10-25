"""Collect qualified v8 diagnostics; no model fitting and no manuscript edits."""
import sys,itertools,math
import numpy as np
import pandas as pd
from scipy.special import expit,logit
import study_v8 as s
from fit_runner import Interface
from repair_reporting_records import qualification_record

LABELS=['all_participants_set_robust','observed_robust_entrants_only','actual_observed_initial_state']

def contrast(frame,kind,path,sleep,active,refactive):
    mat,par,names=s.mapping(frame,kind)
    if kind=='original':return s.old.contrast(frame,names,path,sleep,active,refactive)
    z=(sleep-7.)/2.;c=np.zeros(len(par));prefix=f'{path[0]}->{path[1]}:'
    for name,value in [('activity',active-refactive),('sleep_linear2h',z),('sleep_quadratic2h',z*z),('sleep_linear2h_activity',z*active)]:c[par.index(prefix+name)]=value
    return c

def profiles(theta,cov,frame,kind):
    rows=[]
    for path in s.TRANS:
        for label,sleep,active,ref in [('active_vs_inactive_sleep7',7.,1,0),('sleep5_vs7_inactive',5.,0,0),('sleep5_vs7_active',5.,1,1)]:
            c=contrast(frame,kind,path,sleep,active,ref);est=s.old.estimate(c,theta,cov)
            rows.append(dict(transition=f'{path[0]}->{path[1]}',contrast=label,**est))
    return rows

def scenario_df(df,frame,kind,sleep,active):
    d=df.copy();d['age_exit']=d.age_enter+2.;d['sleep']=sleep;d['activity']=active;d['j']=3
    if kind=='original':
        sp=s.old.spline(frame,[sleep])[0]
        for k in range(3):d[f'sleep_ns{k+1}']=sp[k];d[f'sleep_ns{k+1}_activity']=sp[k]*active
    return d

def probability_gradient(theta,df,frame,kind,sleep,active,actual=False):
    d=scenario_df(df,frame,kind,sleep,active)
    if not actual:d['i']=0
    a=Interface(d,frame,kind,1.);_,_,scores,observed,lp,_=a.evaluate(theta)
    probabilities=np.exp(observed);free_scores=scores@a.mat
    gradients=-free_scores*probabilities[:,None]/d.weight.to_numpy()[:,None]
    return probabilities,gradients

def mortality(theta,cov,frame,kind,validate=False):
    d=s.old.sample(frame).sort_values(['person_id','wave_start'],kind='stable').drop_duplicates('person_id').copy();states=d.i.to_numpy();w=d.weight.to_numpy(float);rows=[];results={}
    for active in [0,1]:
        p,g=probability_gradient(theta,d,frame,kind,7.,active)
        pa,ga=probability_gradient(theta,d,frame,kind,7.,active,True)
        vals=[];grads=[]
        for label in LABELS:
            mask=(states==0) if label==LABELS[1] else np.ones(len(d),bool);ww=w[mask]/w[mask].sum()
            pp,gg=(pa,ga) if label==LABELS[2] else (p,g)
            value=float(ww@pp[mask]);grad=np.sum(ww[:,None]*gg[mask],axis=0);var=float(grad@cov@grad)
            if not 0<value<1 or not var>0 or not np.isfinite(var):raise ValueError('NOT_ESTIMABLE mortality probability/variance')
            se=np.sqrt(var);lo,hi=expit(logit(value)+np.array([-1,1])*1.96*se/(value*(1-value)))
            rows.append(dict(contrast='mortality_probability_7h',scenario=label,activity=active,estimate=value,se=float(se),ci_lower=float(lo),ci_upper=float(hi),n_people=int(mask.sum()),status='PASS'))
            vals.append(value);grads.append(grad)
        results[active]=(vals,grads)
    for k,label in enumerate(LABELS):
        value=results[1][0][k]-results[0][0][k];grad=results[1][1][k]-results[0][1][k];var=float(grad@cov@grad)
        if not var>0 or not np.isfinite(var):raise ValueError('NOT_ESTIMABLE paired mortality variance')
        se=np.sqrt(var);rows.append(dict(contrast='mortality_active_minus_inactive_7h',scenario=label,estimate=value,se=float(se),ci_lower=float(value-1.96*se),ci_upper=float(value+1.96*se),status='PASS'))
    return rows

def mortality_check(theta,frame,kind):
    # Targeted independent probability/gradient check of the free-coordinate
    # chain rule, using actual people/states and age-boundary crossings.
    from scipy.linalg import expm
    d=s.old.sample(frame).sort_values(['person_id','wave_start'],kind='stable').drop_duplicates('person_id')
    d=pd.concat([d.groupby('i').head(1),d[(d.age_enter<85)&(d.age_enter+2>85)].head(1)]).drop_duplicates('person_id')
    dd=scenario_df(d,frame,kind,7.,0);dd['i']=0;a=Interface(dd,frame,kind,1.);w=dd.weight.to_numpy();w=w/w.sum()
    p,g=probability_gradient(theta,d,frame,kind,7.,0);value=float(w@p);grad=np.sum(w[:,None]*g,axis=0)
    def ref(th):
        x,dt,bands,si,sj,_=a.args;beta=(a.mat@th).reshape(9,-1);values=[]
        for n in range(len(x)):
            pp=np.eye(4)
            for k in np.flatnonzero(dt[n]>0):
                xx=x[n].copy()
                if bands[n,k]>0:xx[bands[n,k]]=1
                rates=np.exp(beta@xx);q=np.zeros((4,4))
                for rate,(i,j) in zip(rates,s.TRANS):q[i,j]=rate;q[i,i]-=rate
                pp=pp@expm(q*dt[n,k])
            values.append(w[n]*pp[0,3])
        return math.fsum(values)
    fd=np.zeros(len(theta))
    for k in range(len(theta)):
        hi=theta.copy();lo=theta.copy();hi[k]+=1e-5;lo[k]-=1e-5;fd[k]=(ref(hi)-ref(lo))/(2e-5)
    verr=abs(value-ref(theta));gerr=float((abs(grad-fd)/(1+abs(fd))).max())
    check=dict(frame=frame,model=kind,rows=len(d),coordinates=len(theta),probability_error=verr,gradient_error=gerr,status='PASS' if verr<=1e-9 and gerr<=2e-6 else 'FAIL')
    if check['status']!='PASS':raise RuntimeError(f'STOP mortality derivative preflight {check}')
    return check

def collect():
    out=s.OUT/'diagnostic_evidence';out.mkdir(exist_ok=True);count=0
    # Frozen original rows are reused from already-verified G4 tables.
    tir=pd.read_csv(s.G4/'g4_972_all_lambda_TIR.csv');m=pd.read_csv(s.G4/'g4_mortality_initial_state_scenarios.csv')
    for frame in s.FRAMES:
        for lam,_ in s.RUNS:
            dest=out/f'frozen_original_{frame}_lambda{lam:g}.json'
            if dest.exists():continue
            f,theta,cov,info,lineage=s.old.model(frame,lam);tt=tir[tir.frame.eq(frame)&tir.lambda_value.eq(lam)&tir.contrast.isin(['active_vs_inactive_sleep7','sleep5_vs7_inactive','sleep5_vs7_active'])]
            assert len(tt)==27
            tr=tt[['transition','contrast','logTIR','variance','TIR','se','ci_lower','ci_upper','status']].to_dict('records')
            mm=m[m.frame.eq(frame)&m.lambda_value.eq(lam)].copy();mr=[]
            for r in mm.to_dict('records'):
                pair=r['record_type']=='paired_activity_difference';mr.append(dict(contrast='mortality_active_minus_inactive_7h' if pair else 'mortality_probability_7h',scenario=r['scenario'],estimate=r['difference_active_minus_inactive'] if pair else r['probability'],se=r['se'],ci_lower=r['ci_lower'],ci_upper=r['ci_upper'],status=r['status'],**({} if pair else {'activity':r['activity']})))
            assert len(mr)==9
            s.write(dest,dict(candidate_id=dest.stem,frame=frame,model='original',lambda_value=lam,start='frozen',partition='full',qualified=True,inference_qualified=True,penalized_objective=f['penalized_objective'],nll=f['nll'],penalty=f['penalty'],gradient_inf_total=f['gradient_inf_total'],information=info,lineage=lineage,TIR=tr,mortality=mr,source_reuse='G4 verified core tables; no repeat of 972 arithmetic checks'))
            count+=1
    for directory in sorted((s.OUT/'attempts').iterdir()):
        result=qualification_record(directory)
        if not result.exists():continue
        r=s.js(result)
        if r['partition']!='full' or not r['NUMERICALLY_QUALIFIED']:continue
        dest=out/f'{directory.name}.json'
        if dest.exists():continue
        theta=np.load(directory/'theta.npy');cov=np.load(directory/'subject_cluster_sandwich.npy');frame=r['frame'];kind=r['model'];lam=r['lambda_value']
        checkpath=out/f'mortality_chain_check_{frame}_{kind}.json'
        if not checkpath.exists():s.write(checkpath,mortality_check(theta,frame,kind))
        rows=profiles(theta,cov,frame,kind);mort=mortality(theta,cov,frame,kind)
        lineage=dict(parameter_path=str(directory/'theta.npy'),parameter_sha256=s.sha(directory/'theta.npy'),covariance_path=str(directory/'subject_cluster_sandwich.npy'),covariance_sha256=s.sha(directory/'subject_cluster_sandwich.npy'),result_path=str(result),result_sha256=s.sha(result))
        s.write(dest,dict(candidate_id=directory.name,frame=frame,model=kind,lambda_value=lam,start=r['start'],partition='full',qualified=r['NUMERICALLY_QUALIFIED'],inference_qualified=r['INFERENCE_QUALIFIED'],penalized_objective=r['penalized_objective'],nll=r['nll'],penalty=r['penalty'],gradient_inf_total=r['gradient_inf_total'],information=r['information'],lineage=lineage,TIR=rows,mortality=mort));count+=1
        print('EVIDENCE',directory.name,flush=True)
    print('NEW_EVIDENCE_CANDIDATES',count,flush=True)

def heldout():
    rows=[];people=[];plan=pd.read_csv(s.OUT/'fit_plan.csv')
    for frame in s.FRAMES:
        tests={};source={};d=s.old.sample(frame);train,test,_=s.cv_frames(d,frame)
        for kind in ['original','compact']:
            rr=plan[plan.frame.eq(frame)&plan.model.eq(kind)&plan.partition.eq('train')];assert len(rr)==1;idx=int(rr.index[0]);dirs=list((s.OUT/'attempts').glob(f'{idx:02d}_*'));assert len(dirs)==1
            result=qualification_record(dirs[0])
            if not result.exists() or not s.js(result)['NUMERICALLY_QUALIFIED']:
                rows.append(dict(frame=frame,status='NOT_ESTIMABLE',reason=f'{kind} registered training fit not numerically qualified'));break
            th=np.load(dirs[0]/'theta.npy');a=Interface(test,frame,kind,1.);tests[kind]=a.evaluate(th,False)[3];source[kind]=str(dirs[0])
        if len(tests)!=2:continue
        w=test.weight.to_numpy();delta=tests['compact']-tests['original'];mean=float(w@delta/w.sum());n=test.person_id.nunique();contrib=w*(delta-mean)/w.sum()
        grouped=pd.DataFrame(dict(person_id=test.person_id.to_numpy(),influence=contrib)).groupby('person_id',sort=True).influence.sum()
        se=float(np.sqrt(n/(n-1)*np.square(grouped).sum()));lo=mean-1.96*se;hi=mean+1.96*se
        rows.append(dict(frame=frame,status='PASS',train_persons=train.person_id.nunique(),validation_persons=n,validation_intervals=len(test),original_weighted_logscore=float(w@tests['original']/w.sum()),compact_weighted_logscore=float(w@tests['compact']/w.sum()),paired_compact_minus_original=mean,paired_subject_se=se,ci_lower=lo,ci_upper=hi,clearly_unacceptable_deterioration=bool(mean<-.02 and hi<0),inference_scope='Conditional on fixed trained parameters and prespecified split; no external validation or equivalence claim',**{k+'_training_source':v for k,v in source.items()}))
        for pid,ii in test.groupby('person_id',sort=True).groups.items():
            # sample() uses a non-contiguous index after train/test subsetting;
            # align by position explicitly rather than using original labels.
            mask=test.person_id.eq(pid).to_numpy();people.append(dict(frame=frame,person_id=pid,intervals=int(mask.sum()),weight_sum=float(w[mask].sum()),weighted_delta_sum=float(w[mask]@delta[mask])))
    s.save(s.OUT/'heldout_prediction_comparison.csv',rows);s.save(s.OUT/'heldout_subject_scores.csv',people)
    return rows

if __name__=='__main__':{'collect':collect,'heldout':heldout}[sys.argv[1]]()
