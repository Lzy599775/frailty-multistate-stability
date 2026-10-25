"""Read-only diagnostics at exact rejected trial points; no optimizer calls."""
import numpy as np
import study_v8 as s
from fit_runner import Interface

def main():
    import stable_log_ctmc as engine
    rows=[]
    for directory in sorted((s.OUT/'attempts').iterdir()):
        p=directory/'candidate_exception.json'
        if not p.exists():continue
        out=directory/'trial_failure_read_only_diagnosis.json'
        if out.exists():continue
        cfg=s.js(directory/'configuration.json');frame=str(cfg['frame']);kind=cfg['model'];df=s.old.sample(frame)
        if cfg['partition']=='train':df=s.cv_frames(df,frame)[0]
        a=Interface(df,frame,kind,float(cfg['lambda_value']));theta=np.load(directory/'failing_expanded_theta.npy')
        try:
            val,g,scores,lp,observed,counts=engine.data_evaluate(theta,*a.args,True);prob=np.exp(lp);err=abs(prob.sum(2)-1);n,r=np.unravel_index(err.argmax(),err.shape)
            detail=dict(status='EXACT_TRIAL_REEVALUATED_NO_REFIT',weighted_data_nll=float(val),data_gradient_inf=float(abs(g).max()),maximum_rowsum_error=float(err.max()),minimum_probability=float(prob.min()),maximum_probability=float(prob.max()),person_id=str(df.iloc[n].person_id),wave_start=str(df.iloc[n].wave_start),interval_row=int(n),matrix_row=int(r),probability_matrix=prob[n].tolist(),log_probability_matrix=[[None if not np.isfinite(x) else float(x) for x in rr] for rr in lp[n]],observed_log_probability=float(observed[n]),series_terms_and_squarings=counts[n].tolist(),segments=[])
            x,dt,bands,*_=a.args;beta=theta.reshape(9,-1)
            for k in np.flatnonzero(dt[n]>0):
                xx=x[n].copy()
                if bands[n,k]>0:xx[bands[n,k]]=1
                rates=np.exp(beta@xx);q=np.zeros((4,4))
                for rate,(i,j) in zip(rates,s.TRANS):q[i,j]=rate;q[i,i]-=rate
                detail['segments'].append(dict(duration=float(dt[n,k]),age_band=int(bands[n,k]),rates=rates.tolist(),generator=q.tolist()))
            detail['interpretation']='Finite parameter evaluation returned a matrix outside the unchanged 1e-9 probability qualification tolerance; original attempt stopped. This is not evidence that a target-lambda optimum does not exist.'
            detail['root_cause_scope']='Probability computation qualification at an unaccepted optimizer trial; no data recoding or model correction attempted.'
        except Exception as exc:
            detail=dict(status='EXACT_TRIAL_ENGINE_EXCEPTION_REPRODUCED',error_type=type(exc).__name__,error=str(exc),root_cause_scope='Original exception preserved; no alternative probability method or retry')
        detail.update(source_exception_path=str(p),source_exception_sha256=s.sha(p),failing_theta_sha256=s.sha(directory/'failing_expanded_theta.npy'),utc=s.utc(),model_refit_performed=False)
        s.write(out,detail);print('TRIAL_DIAGNOSIS',directory.name,detail['status'],flush=True)

if __name__=='__main__':main()
