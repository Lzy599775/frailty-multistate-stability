"""Fixed-target optimizations and matched subject sandwich, v8 bounded batch."""
import sys,time,traceback,math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import expm
import study_v8 as s


class Interface:
    def __init__(self,df,frame,kind,lam,destination=None):
        import stable_log_ctmc as engine
        self.engine=engine;self.df=df;self.frame=frame;self.kind=kind;self.lam=lam
        self.mat,self.parameters,self.names=s.mapping(frame,kind)
        self.args=s.arrays(df,frame,kind);self.center,self.prec=s.penalty(self.parameters,lam)
        self.destination=destination;self.last=None;self.calls=0

    def evaluate(self,theta,need_gradient=True):
        self.calls+=1;theta=np.asarray(theta,float);expanded=self.mat@theta
        try:
            if not np.isfinite(theta).all():raise ValueError('Nonfinite free parameter')
            value,g,scores,lp,observed,counts=self.engine.data_evaluate(expanded,*self.args,need_gradient)
            prob=np.exp(lp)
            diag=[float(np.exp(observed.min())),float(np.exp(observed.max())),float(abs(prob.sum(2)-1).max()),float(prob.min()),float(prob.max())]
            if diag[2]>1e-9 or diag[3]<-1e-9 or diag[4]>1+1e-9:raise ValueError('Stable probability diagnostics failed')
            delta=theta-self.center;pen=.5*np.dot(self.prec*delta,delta)
            grad=self.mat.T@g+self.prec*delta
            if not np.isfinite(value+pen) or not np.isfinite(grad).all():raise ValueError('Nonfinite objective or analytic gradient')
            self.last=dict(value=float(value+pen),nll=float(value),penalty=float(pen),gradient_inf=float(abs(grad).max()),probability_diagnostics=diag)
            return float(value+pen),grad,scores,observed,lp,self.last
        except Exception:
            if self.destination is not None and not (self.destination/'candidate_exception.json').exists():
                original=traceback.format_exc();np.save(self.destination/'failing_free_theta.npy',theta);np.save(self.destination/'failing_expanded_theta.npy',expanded)
                located=[]
                for n in range(len(self.df)):
                    try:self.engine.data_evaluate(expanded,*tuple(a[n:n+1] for a in self.args),True)
                    except Exception as exc:
                        located.append(dict(row=n,person_id=str(self.df.iloc[n].person_id),wave_start=str(self.df.iloc[n].wave_start),error=repr(exc)))
                        np.savez(self.destination/'failing_interval_arrays.npz',**{f'arg{k}':a[n:n+1] for k,a in enumerate(self.args)})
                        break
                s.write(self.destination/'candidate_exception.json',dict(utc=s.utc(),call=self.calls,traceback=original,located=located))
            raise

    def hessian(self,theta):
        hess=np.empty((len(theta),len(theta)))
        for k in range(len(theta)):
            h=1e-4*max(1.,abs(theta[k]));hi=theta.copy();lo=theta.copy();hi[k]+=h;lo[k]-=h
            hess[:,k]=(self.evaluate(hi)[1]-self.evaluate(lo)[1])/(2*h)
        asym=float(abs(hess-hess.T).max());return (hess+hess.T)/2,asym


def scipy_value(interface,theta):
    x,dt,bands,si,sj,w=interface.args;beta=(interface.mat@theta).reshape(9,-1);terms=[]
    for n in range(len(x)):
        pp=np.eye(4)
        for k in np.flatnonzero(dt[n]>0):
            xx=x[n].copy()
            if bands[n,k]>0:xx[bands[n,k]]=1
            rates=np.exp(beta@xx);q=np.zeros((4,4))
            for rate,(i,j) in zip(rates,s.TRANS):q[i,j]=rate;q[i,i]-=rate
            pp=pp@expm(q*dt[n,k])
        terms.append(-w[n]*np.log(pp[si[n],sj[n]]))
    delta=theta-interface.center
    return math.fsum(terms)+.5*np.dot(interface.prec*delta,delta)


def preflight():
    registration=s.js(s.OUT/'batch_registration.json')
    for f in registration['files']:
        p=s.OUT/f['path'];assert p.stat().st_size==f['bytes'] and s.sha(p)==f['sha256'],str(p)
    rows=[]
    for frame in s.FRAMES:
        df=s.old.sample(frame)
        probe=pd.concat([df.groupby(['i','j'],sort=True).head(2),df[(df.age_enter<75)&(df.age_exit>75)].head(3),df[(df.age_enter<85)&(df.age_exit>85)].head(3)]).drop_duplicates(['person_id','wave_start']).reset_index(drop=True)
        for kind in ['original','compact']:
            a=Interface(probe,frame,kind,1.);th=s.independent_start(a.parameters,s.SEEDS[0]);v,g,*_=a.evaluate(th)
            ref=scipy_value(a,th);fd=np.zeros_like(th)
            for k in range(len(th)):
                h=1e-5;hi=th.copy();lo=th.copy();hi[k]+=h;lo[k]-=h
                fd[k]=(a.evaluate(hi)[0]-a.evaluate(lo)[0])/(2*h)
            err=float((abs(g-fd)/(1+abs(fd))).max())
            rows.append(dict(frame=frame,model=kind,test='scipy_and_all_free_gradient',coordinates=len(th),rows=len(probe),scipy_abs_error=abs(v-ref),fd_relative_error=err,status='PASS' if abs(v-ref)<=1e-9 and err<=2e-6 else 'FAIL'))
        train,test,_=s.cv_frames(df,frame)
        rows.append(dict(frame=frame,model='both',test='person_split',train_persons=train.person_id.nunique(),test_persons=test.person_id.nunique(),status='PASS' if len(train)>0 and len(test)>0 and not set(train.person_id)&set(test.person_id) else 'FAIL'))
    # Full-sample identity adapter equivalence at every frozen 2014 solution.
    for lam,_ in s.RUNS:
        f,th,_,_,_=s.old.model('2014',lam);a=Interface(s.old.sample('2014'),'2014','original',lam)
        v,g,*_=a.evaluate(th);direct=a.engine.evaluate(th,a.args,lam)
        dif=float(abs(g-direct[1]).max());err=abs(v-f['penalized_objective'])
        rows.append(dict(frame='2014',model='original',test=f'frozen_identity_lambda{lam:g}',objective_abs_error=err,gradient_identity_error=dif,status='PASS' if err<=1e-6 and dif<=1e-12 and abs(g).max()<=1e-4 else 'FAIL'))
    s.save(s.OUT/'preflight_checks.csv',rows);status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL'
    s.write(s.OUT/'preflight.json',dict(utc=s.utc(),status=status,checks=len(rows),registration_sha256=s.sha(s.OUT/'batch_registration.json')))
    print('PREFLIGHT',status,rows,flush=True)
    if status!='PASS':raise RuntimeError('STOP preflight failed')


def run(index):
    gate=s.js(s.OUT/'preflight.json');assert gate['status']=='PASS'
    plan=pd.read_csv(s.OUT/'fit_plan.csv');r=plan.iloc[index].to_dict();frame=str(r['frame']);kind=r['model'];lam=float(r['lambda_value'])
    dest=s.OUT/'attempts'/f'{index:02d}_{kind}_{frame}_lambda{lam:g}_{r["start"]}';dest.mkdir(parents=True,exist_ok=False)
    s.write(dest/'configuration.json',dict(**r,attempt_index=index,started_utc=s.utc(),registration_sha256=s.sha(s.OUT/'batch_registration.json'),preflight_sha256=s.sha(s.OUT/'preflight.json')))
    starttime=time.perf_counter();stage='input';log=(dest/'iterations.jsonl').open('x',encoding='utf-8')
    try:
        df=s.old.sample(frame)
        if r['partition']=='train':df=s.cv_frames(df,frame)[0]
        assert s.sha(r['start_path'])==r['start_sha256'];theta=np.load(r['start_path']);a=Interface(df,frame,kind,lam,dest)
        nsub=df.person_id.nunique();it=0
        def fun(th):
            val,g,*_=a.evaluate(th);return val/nsub,g/nsub
        def callback(th):
            nonlocal it
            it+=1;entry=dict(iteration=it,seconds=time.perf_counter()-starttime,**a.last)
            log.write(s.json.dumps(entry,allow_nan=False)+'\n');log.flush()
            if it==1 or it%100==0:print(index,kind,frame,lam,'ITER',it,'OBJ',entry['value'],'G',entry['gradient_inf'],flush=True)
        stage='LBFGS_fixed_lambda'
        result=minimize(fun,theta,method='L-BFGS-B',jac=True,callback=callback,options=dict(maxiter=2000,maxls=40,maxcor=50,ftol=1e-15,gtol=.5e-4/nsub))
        theta=result.x.copy();np.save(dest/'lbfgs_theta.npy',theta)
        s.write(dest/'optimizer.json',dict(success=bool(result.success),message=str(result.message),nit=int(result.nit),nfev=int(result.nfev),objective_scaled=float(result.fun)))
        stage='registered_numerical_polishing';polish=[]
        for k in range(4):
            val,g,*_=a.evaluate(theta);norm=float(abs(g).max())
            if norm<=1e-4:break
            hess,_=a.hessian(theta);mineig=float(np.linalg.eigvalsh(hess)[0])
            if mineig<=0:raise RuntimeError('STOP polishing Hessian not positive definite')
            direction=np.linalg.solve(hess,g);accepted=False
            for half in range(21):
                alpha=2.**(-half);candidate=theta-alpha*direction;val1,g1,*_=a.evaluate(candidate);norm1=float(abs(g1).max())
                if val1<=val+1e-9 and norm1<norm:theta=candidate;accepted=True;break
            entry=dict(polish_step=k+1,accepted=accepted,gradient_before=norm,gradient_after=norm1,objective_before=val,objective_after=val1,alpha=alpha,minimum_eigenvalue=mineig)
            polish.append(entry);log.write(s.json.dumps(entry,allow_nan=False)+'\n');log.flush()
            if not accepted:raise RuntimeError('STOP registered numerical polishing step failed')
        val,g,scores,observed,lp,diag=a.evaluate(theta);np.save(dest/'theta.npy',theta);np.save(dest/'gradient.npy',g);np.save(dest/'observed_logp.npy',observed)
        stage='information';hess,asym=a.hessian(theta);eig=np.linalg.eigvalsh(hess);cond=float(np.linalg.cond(hess));datah=hess-np.diag(a.prec)
        np.save(dest/'penalized_information.npy',hess);np.save(dest/'data_information.npy',datah)
        info_ok=bool(eig[0]>0 and cond<=1e10);inference_ok=False;info=dict(minimum_eigenvalue=float(eig[0]),maximum_eigenvalue=float(eig[-1]),condition_number=cond,unsymmetrized_asymmetry=asym,data_minimum_eigenvalue=float(np.linalg.eigvalsh(datah)[0]),qualified=info_ok)
        if info_ok:
            stage='matched_covariance';bread=np.linalg.inv(hess);free_scores=scores@a.mat
            grouped=pd.DataFrame(free_scores).groupby(df.person_id.to_numpy(),sort=True).sum().to_numpy();cov=bread@(grouped.T@grouped)@bread.T
            np.save(dest/'inverse_penalized_information.npy',bread);np.save(dest/'subject_cluster_sandwich.npy',cov)
            # Unsupported original age coefficients have zero DATA score and
            # zero sandwich variance. Qualification of a contrast is separate.
            inference_ok=bool(np.isfinite(cov).all() and np.max(abs(cov-cov.T))<1e-8 and np.diag(cov).min()>=0)
            info.update(cluster_count=len(grouped),effective_parameter_dimension=float(np.trace(bread@datah)),regularization_information_fraction=float(np.trace(bread@np.diag(a.prec))/len(theta)),sandwich_minimum_diagonal=float(np.diag(cov).min()))
        qualified=bool(result.success and abs(g).max()<=1e-4 and info_ok)
        rec=dict(**r,attempt_index=index,frame=frame,model=kind,status='PASS' if qualified else 'FAIL',NUMERICALLY_QUALIFIED=qualified,INFERENCE_QUALIFIED=qualified and inference_ok,SOLUTION_STABILITY='NOT_YET_ADJUDICATED',SCIENTIFIC_INTERPRETATION='NOT_YET_ADJUDICATED',n_intervals=len(df),n_persons=int(nsub),parameter_count=len(theta),optimizer_success=bool(result.success),nit=int(result.nit),nfev=int(result.nfev),penalized_objective=val,nll=diag['nll'],penalty=diag['penalty'],gradient_inf_total=float(abs(g).max()),probability_diagnostics=diag['probability_diagnostics'],information=info,polishing=polish,seconds=time.perf_counter()-starttime,completed_utc=s.utc())
        s.write(dest/'result.json',rec);print('COMPLETED',index,kind,frame,lam,rec['status'],'OBJ',val,'G',rec['gradient_inf_total'],'SEC',rec['seconds'],flush=True)
        return dict(index=index,status=rec['status'],path=str(dest))
    except Exception as exc:
        s.write(dest/'failure.json',dict(utc=s.utc(),stage=stage,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-starttime,independent_registered_attempt_stopped=True))
        print('ATTEMPT_FAILED',index,stage,repr(exc),flush=True);return dict(index=index,status='FAIL',path=str(dest),error=repr(exc))
    finally:log.close()


def batch():
    from concurrent.futures import ProcessPoolExecutor,as_completed
    plan=pd.read_csv(s.OUT/'fit_plan.csv');assert len(plan)==44
    assert not (s.OUT/'attempts').exists(),'No silent resume/duplicate execution'
    # A and B interleaved: bounded independent tasks, never replacement starts.
    indices=list(plan.index);a=[i for i in indices if plan.loc[i,'formal_role']=='same_lambda_diagnostic'];b=[i for i in indices if plan.loc[i,'formal_role']=='posthoc_compact_candidate'];c=[i for i in indices if plan.loc[i,'partition']=='train']
    order=[]
    while a or b:
        if a:order.append(a.pop(0))
        if b:order.append(b.pop(0))
    order+=c;receipts=[]
    s.write(s.OUT/'dispatch_registration.json',dict(utc=s.utc(),workers=3,order=order,expected_attempts=44))
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(run,i):i for i in order}
        for future in as_completed(futures):
            # Worker exceptions outside run() are not swallowed.
            receipts.append(future.result())
            s.write(s.OUT/f'execution_receipts/{futures[future]:02d}.json',receipts[-1])
    s.write(s.OUT/'batch_execution.json',dict(utc=s.utc(),expected=44,executed=len(receipts),receipts=sorted(receipts,key=lambda x:x['index'])))


if __name__=='__main__':
    {'preflight':preflight,'batch':batch}[sys.argv[1]]()
