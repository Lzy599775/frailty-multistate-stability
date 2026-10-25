"""v7 real optimizer execution using frozen v8i/v8g numerical routines.

The authorized difference is corrected data and full-block-only scope.
Configuration, seed, starts, qualification and each original solver are retained.
"""
from pathlib import Path
import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from build_corrected_stageA import ROOT, OUT, READY, model_module, sha, write

sys.stdout.reconfigure(encoding='utf-8')

def validate_g2():
    p=READY/'corrected_stageA_manifest.json'
    j=json.loads(p.read_text(encoding='utf-8'))
    assert j['status']=='PASS' and j['gate']=='G2'
    for f in j['inputs']+j['outputs']:
        p=Path(f['path']);assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],p
    return j

def qualified(m,fit,info):
    d=fit['probability_diagnostics']
    return bool(fit['optimizer_success'] and fit['gradient_inf_total']<=m.config()['convergence']['total_gradient_inf_max'] and info['qualified'] and d[2]<=1e-9 and d[3]>=-1e-9 and d[4]<=1+1e-9)

def lock_model(m):
    math=json.loads((m.OUT/'mathematical_validation_v8.json').read_text(encoding='utf-8'));assert math['status']=='PASS'
    paths=[READY/'corrected_stageA_manifest.json',READY/'transition_intervals_audited.csv',m.LONG,m.RAW,m.BASE/'clhls_sources/clhls_dictionary_utf8.csv',m.SCRIPT,Path(__file__),OUT/'build_corrected_stageA.py',OUT/'semantic_codebook_decision.md',m.OUT/'clhls_model_ready_v2.csv',m.OUT/f'clhls_frame{m.FRAME}_model_input_all_intervals_v2.csv',m.OUT/'model_input_assembly_v2.json',m.OUT/'environment_v2.json',m.OUT/'mathematical_validation_v8.json']
    payload=dict(execution_id=f'CLHLS_{m.FRAME}_FI_CORRECTED_V7_20261024',status='LOCKED_CONVERGENCE_PENDING',created_utc=m.utc(),hash_basis='canonical UTF8 JSON payload sorted keys compact separators; per-file raw bytes SHA256; manifest excluded',configuration=m.config(),scope='targeted full block only; corrected FI inputs; no core/behavior refit; original numerical configuration unchanged',files=[dict(path=str(p),size_bytes=p.stat().st_size,sha256=sha(p)) for p in paths])
    digest=hashlib.sha256(m.canonical(payload)).hexdigest()
    m.write(m.OUT/'execution_manifest_v2.json',dict(payload=payload,execution_checksum=digest))
    return digest

def run_frame(frame):
    validate_g2()
    m=model_module(frame)
    m.OUT=OUT/'models'/f'frame{frame}';m.ANALYSIS=READY;m.LONG=READY/f'clhls_frame{frame}_analysis_ready_long.csv'
    m.INPUTS=[READY/'corrected_stageA_manifest.json',READY/'transition_intervals_audited.csv',m.LONG,m.RAW,m.BASE/'clhls_sources/clhls_dictionary_utf8.csv']
    m.write(m.OUT/'environment_v2.json',m.environment())
    m.write(m.OUT/'original_model_configuration_preserved.json',m.config())
    print('MATHEMATICAL_PREFLIGHT_START',frame,m.utc(),flush=True)
    m.validate()
    digest=lock_model(m)
    m.verify()
    print('FIXED_TWO_START_QUALIFICATION_START',frame,m.utc(),flush=True)
    m.convergence()
    c=json.loads((m.OUT/'convergence_v2.json').read_text(encoding='utf-8'))
    assert c['status']=='PASS'
    m.verify()
    df=pd.read_csv(m.OUT/'clhls_model_ready_v2.csv',dtype={'person_id':str})
    m.PENALTY_STRENGTH=1.
    prefix='formal_full_v2'
    fit,th,args,_=m.fit(df,'full',np.load(m.OUT/'convergence_selected_theta_v2.npy'),prefix)
    info=m.information(th,args,df,prefix)
    ok=qualified(m,fit,info)
    m.write(m.OUT/'formal_full_status_v2.json',dict(status='PASS' if ok else 'FAIL',execution_checksum=digest,fit=fit,information=info))
    if not ok:raise RuntimeError(f'STOP: frame{frame} primary qualification failed')
    se=np.sqrt(np.diag(np.load(m.OUT/(prefix+'_cluster_sandwich.npy'))))
    rows=[]
    for k,(i,j) in enumerate(m.TRANS):
        for z,name in enumerate(args[-1]):
            idx=k*len(args[-1])+z
            rows.append(dict(transition=f'{i}->{j}',parameter=name,estimate=th[idx],penalized_cluster_se=se[idx],ci_lower=th[idx]-1.96*se[idx],ci_upper=th[idx]+1.96*se[idx],execution_checksum=digest))
    pd.DataFrame(rows).to_csv(m.OUT/'formal_full_parameters_v2.csv',index=False,encoding='utf-8-sig')
    m.write(m.OUT/'formal_clhls_completed_v2.json',dict(status='PASS',scope='AUTHORIZED_V7_FULL_ONLY',core='NOT_RUN_NOT_REQUIRED',behavior='NOT_RUN_NOT_REQUIRED',execution_checksum=digest,completed_utc=m.utc()))
    # Frozen sensitivity() starts at initial(df), uses frame-specific schedules,
    # and requires the newly qualified primary; no old theta/covariance reused.
    m.sensitivity()
    print('FRAME_REFITS_COMPLETED',frame,m.utc(),flush=True)

def diagnostics():
    rows=[]
    for frame in [2008,2011,2014]:
        d=OUT/'models'/f'frame{frame}'
        for strength,prefix in [(1.,'formal_full_v2'),(.5,'sensitivity_lambda_0p5'),(2.,'sensitivity_lambda_2p0')]:
            fp=d/(prefix+'_fit.json');ip=d/(prefix+'_information.json')
            row=dict(frame=frame,penalty_strength=strength,status='NOT_RUN',resolution='PREREQUISITE_NOT_COMPLETED',fit_path=str(fp),information_path=str(ip))
            if fp.exists():
                f=json.loads(fp.read_text(encoding='utf-8'));row.update({k:f[k] for k in ['n_intervals','n_persons','parameter_count','penalized_objective','nll','penalty','gradient_inf_total','optimizer_success','nit','nfev','seconds']})
                row.update(fit_sha256=sha(fp),theta_finite_fraction=float(np.isfinite(f['theta']).mean()),status='FAIL',resolution='INFORMATION_OR_QUALIFICATION_PENDING')
                if ip.exists():
                    i=json.loads(ip.read_text(encoding='utf-8'));m=model_module(frame)
                    row.update({k:i[k] for k in ['minimum_eigenvalue','condition_number','data_information_min_eigenvalue','data_information_condition_number','qualified']})
                    ok=qualified(m,f,i);row.update(status='PASS' if ok else 'FAIL',resolution='NEW_OPTIMIZER_AND_NEW_COVARIANCE' if ok else 'NUMERICAL_QUALIFICATION_FAILED',information_sha256=sha(ip))
            rows.append(row)
    return pd.DataFrame(rows)

def main():
    write(OUT/'refit_start.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),command=sys.argv,scope='3 CLHLS frames x primary/sensitivity with original numerical gates',model='9x25=225; original v8i frame2008 and v8g frame2011/2014',primary_lambda=1.,sensitivity=[.5,2.],seed=20261011))
    try:
        for frame in [2008,2011,2014]:run_frame(frame)
    except Exception as error:
        # Preserve and re-raise: never continue to another fit after failed gate.
        tb=traceback.format_exc()
        (OUT/'refit_failure_traceback.txt').write_text(tb,encoding='utf-8')
        diagnostics().to_csv(OUT/'clhls_refit_diagnostics_by_frame_lambda.csv',index=False,encoding='utf-8-sig')
        write(OUT/'refit_execution_stopped.json',dict(status='FAIL',resolution='STOPPED_AT_ORIGINAL_GATE',error_type=type(error).__name__,error=str(error),traceback_path=str(OUT/'refit_failure_traceback.txt'),submission='BLOCK_SUBMISSION',completed_utc=datetime.now(timezone.utc).isoformat()))
        raise
    diagnostics().to_csv(OUT/'clhls_refit_diagnostics_by_frame_lambda.csv',index=False,encoding='utf-8-sig')
    write(OUT/'refit_completed.json',dict(status='PASS',completed_utc=datetime.now(timezone.utc).isoformat(),g3='PASS',downstream='NOT_RUN'))

if __name__=='__main__':main()
