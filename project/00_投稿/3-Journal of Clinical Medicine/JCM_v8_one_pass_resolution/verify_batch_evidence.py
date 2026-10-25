"""Final evidence integrity and targeted independent scalar checks, no fitting."""
import numpy as np
import pandas as pd
import study_v8 as s
import evidence_v8 as e
from repair_reporting_records import qualification_record

def main():
    checks=[]
    for f in s.js(s.OUT/'batch_registration.json')['files']:
        p=s.OUT/f['path'];checks.append(dict(check='registered_source_unchanged',path=str(p),status='PASS' if p.stat().st_size==f['bytes'] and s.sha(p)==f['sha256'] else 'FAIL'))
    for f in pd.read_csv(s.OUT/'inputs/frozen_input_integrity.csv').to_dict('records'):
        p=s.Path(f['path']);checks.append(dict(check='original_input_unchanged',path=str(p),status='PASS' if p.stat().st_size==f['bytes'] and s.sha(p)==f['sha256'] else 'FAIL'))
    plan=pd.read_csv(s.OUT/'fit_plan.csv');actual=[]
    for i,r in plan.iterrows():
        dd=list((s.OUT/'attempts').glob(f'{i:02d}_*'));ok=len(dd)==1
        checks.append(dict(check='exact_one_registered_attempt',attempt=i,status='PASS' if ok else 'FAIL'))
        if not ok:continue
        p=dd[0];cfg=s.js(p/'configuration.json');assert cfg['start_sha256']==r['start_sha256'];q=qualification_record(p)
        if q.exists():
            z=s.js(q);theta=np.load(p/'theta.npy');g=np.load(p/'gradient.npy');h=np.load(p/'penalized_information.npy')
            ok=bool((np.abs(g).max()<=1e-4 and z['optimizer_success'] and np.linalg.eigvalsh(h)[0]>0 and np.linalg.cond(h)<=1e10)==z['NUMERICALLY_QUALIFIED'])
            checks.append(dict(check='numeric_status_matches_saved_evidence',attempt=i,status='PASS' if ok else 'FAIL'))
            if z['NUMERICALLY_QUALIFIED']:
                cov=np.load(p/'subject_cluster_sandwich.npy')
                # Independently calculate a core activity contrast without
                # calling the estimate helper used in the reconstruction.
                if z['model']=='compact':
                    pars=s.mapping(z['frame'],'compact')[1];k=pars.index('1->0:activity');v=np.zeros(len(theta));v[k]=1
                else:
                    v=np.zeros(225);v[s.TRANS.index((1,0))*25+s.mapping(z['frame'],'original')[2].index('activity')]=1
                logratio=float(np.dot(v,theta));variance=float(np.dot(v,np.dot(cov,v)));assert variance>0
                estimate=float(np.exp(logratio));lo=float(np.exp(logratio-1.96*np.sqrt(variance)));hi=float(np.exp(logratio+1.96*np.sqrt(variance)))
                ep=s.OUT/'diagnostic_evidence'/f'{p.name}.json'
                if z['partition']=='full':
                    ev=s.js(ep);tr=next(x for x in ev['TIR'] if x['transition']=='1->0' and x['contrast']=='active_vs_inactive_sleep7');err=max(abs(estimate-tr['TIR']),abs(lo-tr['ci_lower']),abs(hi-tr['ci_upper']))
                    checks.append(dict(check='independent_core_TIR_formula',attempt=i,error=err,status='PASS' if err<1e-10*max(1.,estimate,hi) else 'FAIL'))
                if z.get('record_origin')=='READ_ONLY_RECONSTRUCTION_AFTER_REPORT_SERIALIZATION_ERROR':
                    checks.append(dict(check='report_exception_preserved_no_refit',attempt=i,path=z['original_reporting_failure_path'],status='PASS' if (p/'failure.json').exists() and z['model_refit_performed']==False else 'FAIL'))
        else:
            f=s.js(p/'failure.json');checks.append(dict(check='true_failure_retained',attempt=i,status='PASS' if f['independent_registered_attempt_stopped'] and not (p/'result.json').exists() else 'FAIL'))
        actual.append(i)
    assert sorted(actual)==list(range(44));s.save(s.OUT/'final_evidence_verification.csv',checks)
    status='PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL'
    s.write(s.OUT/'final_evidence_verification.json',dict(utc=s.utc(),status=status,checks=len(checks),executed_attempts=44,expected_attempts=44,extra_optimization_attempts=0,old_evidence_mutations=sum(c['status']=='FAIL' for c in checks if c['check']=='original_input_unchanged'),note='Evidence verification PASS is not numerical-model or scientific-claim PASS'))
    print('FINAL_EVIDENCE_VERIFICATION',status,len(checks),flush=True)
    if status!='PASS':raise RuntimeError('STOP evidence verification failed')

if __name__=='__main__':main()
