"""Explicit read-only reconstruction after a result-dict serialization bug.

Never optimizes, changes theta, edits a failed record, or retries a model.
The immutable fit_runner.py remains the registered executed version.
"""
import numpy as np
import pandas as pd
import study_v8 as s
from fit_runner import Interface

def qualification_record(directory):
    # Reporting correction is explicit and separately preserved, not a model
    # fallback or replacement parameter version.
    corrected=directory/'reporting_repair_result.json'
    original=directory/'result.json'
    if corrected.exists():
        assert not original.exists()
        return corrected
    return original

def main():
    count=0
    for directory in sorted((s.OUT/'attempts').iterdir()):
        failure=directory/'failure.json';corrected=directory/'reporting_repair_result.json'
        if not failure.exists() or corrected.exists():continue
        # A dispatch receipt is written only after the worker has closed all
        # files, so a live writer is never read as a completed exception.
        index=int(directory.name.split('_',1)[0])
        if not (s.OUT/f'execution_receipts/{index:02d}.json').exists():continue
        f=s.js(failure)
        if not (f['error_type']=='TypeError' and f['error']=="dict() got multiple values for keyword argument 'frame'" and 'rec=dict(**r,attempt_index=index,frame=frame' in f['traceback']):continue
        assert f['stage'] in ['information','matched_covariance'] and not (directory/'candidate_exception.json').exists()
        r=s.js(directory/'configuration.json');frame=str(r['frame']);kind=r['model'];lam=float(r['lambda_value']);index=r['attempt_index']
        df=s.old.sample(frame)
        if r['partition']=='train':df=s.cv_frames(df,frame)[0]
        th=np.load(directory/'theta.npy');stored_g=np.load(directory/'gradient.npy');a=Interface(df,frame,kind,lam)
        value,g,scores,observed,lp,diag=a.evaluate(th);assert abs(g-stored_g).max()<=1e-12
        assert abs(observed-np.load(directory/'observed_logp.npy')).max()<=1e-12
        h=np.load(directory/'penalized_information.npy');datah=np.load(directory/'data_information.npy')
        assert np.isfinite(h).all() and abs(h-h.T).max()<=1e-12
        assert abs(h-datah-np.diag(a.prec)).max()<=1e-10
        eigen=np.linalg.eigvalsh(h);cond=float(np.linalg.cond(h));info_ok=bool(eigen[0]>0 and cond<=1e10);optimizer=s.js(directory/'optimizer.json')
        inference_ok=False;covdiff=None
        entries=[s.json.loads(line) for line in (directory/'iterations.jsonl').read_text(encoding='utf-8').splitlines()]
        polish=[x for x in entries if 'polish_step' in x];qualified=bool(optimizer['success'] and abs(g).max()<=1e-4 and info_ok)
        info=dict(minimum_eigenvalue=float(eigen[0]),maximum_eigenvalue=float(eigen[-1]),condition_number=cond,data_minimum_eigenvalue=float(np.linalg.eigvalsh(datah)[0]),qualified=info_ok,unsymmetrized_asymmetry='NOT_RETAINED_BEFORE_SERIALIZATION_ERROR')
        if info_ok:
            bread=np.load(directory/'inverse_penalized_information.npy');cov=np.load(directory/'subject_cluster_sandwich.npy');assert abs(bread@h-np.eye(len(th))).max()<=1e-7
            free=scores@a.mat;grouped=pd.DataFrame(free).groupby(df.person_id.to_numpy(),sort=True).sum().to_numpy();recomputed_cov=bread@(grouped.T@grouped)@bread.T
            covdiff=float(abs(recomputed_cov-cov).max());assert covdiff<=1e-8
            inference_ok=bool(np.isfinite(cov).all() and abs(cov-cov.T).max()<1e-8 and np.diag(cov).min()>=0)
            info.update(cluster_count=len(grouped),effective_parameter_dimension=float(np.trace(bread@datah)),regularization_information_fraction=float(np.trace(bread@np.diag(a.prec))/len(th)),sandwich_minimum_diagonal=float(np.diag(cov).min()))
        record={**r,'frame':frame,'status':'PASS' if qualified else 'FAIL','NUMERICALLY_QUALIFIED':qualified,'INFERENCE_QUALIFIED':qualified and inference_ok,'SOLUTION_STABILITY':'NOT_YET_ADJUDICATED','SCIENTIFIC_INTERPRETATION':'NOT_YET_ADJUDICATED','n_intervals':len(df),'n_persons':int(df.person_id.nunique()),'parameter_count':len(th),'optimizer_success':optimizer['success'],'nit':optimizer['nit'],'nfev':optimizer['nfev'],'penalized_objective':value,'nll':diag['nll'],'penalty':diag['penalty'],'gradient_inf_total':float(abs(g).max()),'probability_diagnostics':diag['probability_diagnostics'],'information':info,'polishing':polish,'seconds':f['seconds'],'completed_utc':f['utc'],'qualification_reconstructed_utc':s.utc(),'record_origin':'READ_ONLY_RECONSTRUCTION_AFTER_REPORT_SERIALIZATION_ERROR','original_reporting_failure_path':str(failure),'original_reporting_failure_sha256':s.sha(failure),'covariance_recomputation_max_error':covdiff,'model_refit_performed':False,'unavailable_diagnostic':'Unsymmetrized Hessian asymmetry scalar was not saved; symmetric information and all qualification inputs retained'}
        s.write(corrected,record);count+=1;print('REPORT_RECORD_RECONSTRUCTED',index,record['status'],flush=True)
    print('REPORT_RECORDS_NEW',count,flush=True)

if __name__=='__main__':main()
