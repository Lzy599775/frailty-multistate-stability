"""Authorized controlled run. No resume, alternative starts, or downstream G4/G5."""
from pathlib import Path
from datetime import datetime,timezone
import ast, hashlib, importlib.util, json, shutil, sys, time, traceback
import numpy as np
import pandas as pd
import stable_log_ctmc as engine

sys.stdout.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parent;SUB=OUT.parent;ROOT=SUB.parents[1]
R2=SUB/'numeric_method_validation_v7_1_r2';DEBUG=SUB/'numeric_debug_v7_1'
V7=SUB/'audit_v7_fi_corrected/authorized_repair_20261024'
READY=V7/'corrected_stageA';MODEL=OUT/'controlled_model_v7_1.py'
OLD_FILES={};CURRENT={'phase':'START','target':None};FORMAL_ATTEMPTS=[]

def utc():return datetime.now(timezone.utc).isoformat()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False,default=lambda o:o.item() if isinstance(o,np.generic) else str(o))
def canonical(o):return json.dumps(o,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')
def rel(a,b):return float(np.max(np.abs(a-b)/(1+np.abs(b))))
def register(p,expected=None,size=None):
    p=Path(p);actual=sha(p)
    if expected is not None:assert actual==expected,'SHA mismatch: '+str(p)
    if size is not None:assert p.stat().st_size==size,'Size mismatch: '+str(p)
    if str(p) in OLD_FILES:assert OLD_FILES[str(p)]['sha256']==actual
    OLD_FILES[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=actual)
def module(path,frame,name):
    argv=sys.argv;sys.argv=[str(path),str(frame)]
    try:
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    finally:sys.argv=argv
    return m

def verify_sources():
    seals=[(V7,'v7_output_checksums.json'),(DEBUG,'output_checksums_v7_1.json'),
           (SUB/'numeric_method_validation_v7_1','round1_checksums.json'),
           (R2,'validation_output_checksums.json'),(R2,'validation_delivery_addendum_checksums.json')]
    inventories=[]
    for directory,name in seals:
        p=directory/name;register(p);records=load(p)['files']
        for row in records:register(directory/row['path'],row['sha256'],row.get('bytes',row.get('size_bytes')))
        inventories.append(dict(seal=str(p),recorded_count=len(records),status='PASS'))
    for frame in [2008,2011,2014]:
        p=V7/f'models/frame{frame}/execution_manifest_v2.json';j=load(p)
        assert hashlib.sha256(canonical(j['payload'])).hexdigest()==j['execution_checksum']
        register(p)
        for row in j['payload']['files']:register(row['path'],row['sha256'],row['size_bytes'])
    g2=load(READY/'corrected_stageA_manifest.json');assert g2['status']=='PASS' and g2['gate']=='G2'
    for row in g2['inputs']+g2['outputs']:register(row['path'],row['sha256'],row['bytes'])
    for name in ['JCM_manuscript.docx','JCM_Supplementary_Material.docx','JCM_v7_1_Codex_numeric_failure_diagnosis_protocol.md']:
        register(SUB/name)
    assert load(R2/'validation_completion_verified.json')['status']=='PASS'
    assert sha(OUT/'stable_log_ctmc.py')==load(R2/'validation_registration.json')['implementation_sha256']
    assert sha(OUT/'frozen_original_v8g.py')==sha(ROOT/'07_data_processing/stageB_10_clhls_penalized_execution_v8g.py')
    write(OUT/'baseline_dependency_snapshot.json',dict(created_utc=utc(),hash_basis='absolute path; per-file raw bytes SHA256; bytes',files=list(OLD_FILES.values()),seals=inventories))
    print('BASELINE_HASH_PASS',len(OLD_FILES),flush=True)

def preflight():
    CURRENT['phase']='PREFLIGHT';verify_sources()
    m=module(MODEL,2014,'controlled_2014');old=module(OUT/'frozen_original_v8g.py',2014,'original_2014')
    assert m.environment()==load(V7/'models/frame2014/environment_v2.json'),'Environment differs from frozen formal environment'
    write(OUT/'environment.json',m.environment())
    a=ast.parse(MODEL.read_text(encoding='utf-8'));b=ast.parse((OUT/'frozen_original_v8g.py').read_text(encoding='utf-8'))
    funcs_a={n.name:ast.dump(n,include_attributes=False) for n in a.body if isinstance(n,ast.FunctionDef)}
    funcs_b={n.name:ast.dump(n,include_attributes=False) for n in b.body if isinstance(n,ast.FunctionDef)}
    unchanged=[]
    for name in funcs_b:
        if name in ['config','data_evaluate']:continue
        assert funcs_a[name]==funcs_b[name],name+' changed beyond authorized numerical adapter'
        unchanged.append(name)
    legacy=next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name=='legacy_data_evaluate');legacy.name='data_evaluate'
    assert ast.dump(legacy,include_attributes=False)==funcs_b['data_evaluate']
    co,cn=old.config(),m.config()
    for k in ['execution_version']:co.pop(k);cn.pop(k)
    for k in ['matrix_exp','gradient','objective_reduction']:co['numeric'].pop(k,None);cn['numeric'].pop(k,None)
    assert co==cn,'Statistical configuration differs'
    write(OUT/'integration_scope_validation.json',dict(status='PASS',unchanged_functions=unchanged,legacy_engine_retained=True,only_config_differences=['execution_version','numeric.matrix_exp','numeric.gradient','numeric.objective_reduction']))
    source=V7/'models/frame2014/clhls_model_ready_v2.csv'
    shutil.copyfile(source,OUT/'preflight/clhls_model_ready_v2.csv')
    assert sha(source)==sha(OUT/'preflight/clhls_model_ready_v2.csv')
    df=pd.read_csv(source,dtype={'person_id':str})
    assert len(df)==2779 and df.person_id.nunique()==2779
    assert not df.duplicated(['person_id','wave_start']).any()
    assert np.isfinite(m.arrays(df)[0]).all() and (df.interval_years>0).all()
    assert df.groupby(['i','j']).size().get((2,0),0)==6
    write(OUT/'sample_input_validation.json',dict(status='PASS',n_intervals=len(df),n_persons=int(df.person_id.nunique()),duplicates=0,unmatched='NOT_APPLICABLE_NO_JOIN',missing_design_values=0,dropped=0,frail_robust_observed_endpoints=6,source_sha256=sha(source),sample_reconstruction='NOT_RUN_FROZEN_CORRECTED_INPUT_REUSED'))
    m.OUT=OUT/'preflight';m.PENALTY_STRENGTH=1.
    print('INTEGRATED_MATHEMATICAL_VALIDATION_START',utc(),flush=True);m.validate()
    rows=[]
    inputs=np.load(DEBUG/'throwing_input_arrays.npz');args=tuple(inputs[k] for k in ['x','dt','bands','si','sj','weights'])
    for name in ['throwing_candidate','last_accepted']:
        theta=np.load(DEBUG/(name+'_theta.npy'));m.PENALTY_STRENGTH=8.
        v,g,s,d=m.evaluate(theta,*args);reference=engine.evaluate(theta,args,8.)
        cached=np.load(R2/(name+'_prototype_arrays.npz'))
        status=load(R2/(name+'_point_status.json'))
        row=dict(point=name,objective_error=abs(v-status['objective']),gradient_error=rel(g,cached['gradient']),score_error=rel(s,cached['data_scores']),rowsum=float(d[2]))
        assert row['objective_error']<=1e-9 and row['gradient_error']<=2e-6 and row['score_error']<=2e-6
        assert v==reference[0] and np.array_equal(g,reference[1]) and np.array_equal(s,reference[2])
        assert np.array_equal(reference[3],cached['log_probability_matrix'])
        row['status']='PASS';rows.append(row);print('INTEGRATED_CAPTURE_POINT_PASS',name,flush=True)
    plan=load(R2/'validation_registration.json')
    for case in [c for c in plan['cases'] if 'fit_path' in c]:
        frame=case['frame'];strength=case['lambda_value'];name=case['name']
        new=module(MODEL,frame,'integrated_'+name)
        op=ROOT/'07_data_processing'/('stageB_10_clhls_penalized_execution_v8i.py' if frame==2008 else 'stageB_10_clhls_penalized_execution_v8g.py')
        original=module(op,frame,'old_'+name);new.PENALTY_STRENGTH=original.PENALTY_STRENGTH=strength
        data=pd.read_csv(V7/f'models/frame{frame}/clhls_model_ready_v2.csv',dtype={'person_id':str})
        na,oa=new.arrays(data),original.arrays(data)
        assert na[-1]==oa[-1]
        for x,y in zip(na[:-1],oa[:-1]):assert np.array_equal(x,y)
        frozen=load(case['fit_path']);theta=np.array(frozen['theta']);v,g,s,d=new.evaluate(theta,*na[:-1]);ov,og,os,od=original.evaluate(theta,*oa[:-1])
        cached=np.load(R2/name/'diagnostic_regression_arrays.npz')
        row=dict(point=name,objective_error=abs(v-ov),gradient_error=rel(g,og),score_error=rel(s,os),cached_gradient_error=rel(g,cached['gradient_new']),cached_scores_error=rel(s,cached['scores_new']),rowsum=float(d[2]))
        assert abs(ov-frozen['penalized_objective'])<=1e-9
        assert row['objective_error']<=1e-9 and row['gradient_error']<=2e-6 and row['score_error']<=2e-6
        assert row['cached_gradient_error']==0 and row['cached_scores_error']==0
        assert d[2]<=1e-9 and d[3]>=-1e-9 and d[4]<=1+1e-9
        assert np.isfinite(v) and np.isfinite(g).all()
        row['status']='PASS';rows.append(row)
        pd.DataFrame(rows).to_csv(OUT/'preflight/interface_regression.csv',index=False,encoding='utf-8-sig')
        print('INTEGRATED_FROZEN_PASS',name,flush=True)
    m=module(MODEL,2014,'formal_controlled_2014');m.OUT=OUT/'models/frame2014';m.PENALTY_STRENGTH=1.
    start=m.initial(df,25);v,g,_,_=m.evaluate(start,*m.arrays(df)[:-1]);v2,g2,_,_=m.evaluate(start,*m.arrays(df)[:-1])
    assert v==v2 and np.array_equal(g,g2) and np.isfinite(g).all()
    np.save(OUT/'preflight/original_initial_theta.npy',start)
    old.PENALTY_STRENGTH=1.
    assert np.array_equal(start,old.initial(df,25))
    write(OUT/'preflight_completed.json',dict(status='PASS',completed_utc=utc(),capture_points=2,frozen_models=7,original_mathematical_validation='PASS',configuration_and_function_identity='PASS',data='PASS',environment='PASS',seed=m.SEED,new_formal_fits=0,prior_two_start_qualification='RETAINED_FROZEN_EVIDENCE_NOT_RERUN',prototype_full_FD_Hessian='RETAINED_R2_NOT_REPEATED'))
    return m,df

def lock(m):
    paths=[p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.log')]
    payload=dict(created_utc=utc(),execution_version='v7_1_controlled_recovery_stable_log_r2',scope='2014 lambda0.5 then lambda2 ONLY',configuration=m.config(),preflight='PASS',hash_basis='absolute paths; raw bytes SHA256; canonical UTF8 JSON sorted keys compact separators; excludes manifest itself and future run outputs',files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)])
    digest=hashlib.sha256(canonical(payload)).hexdigest();write(OUT/'controlled_execution_manifest.json',dict(payload=payload,execution_checksum=digest));return digest

def verify_locked():
    record=load(OUT/'controlled_execution_manifest.json');assert hashlib.sha256(canonical(record['payload'])).hexdigest()==record['execution_checksum']
    for r in record['payload']['files']:
        assert Path(r['path']).stat().st_size==r['bytes'] and sha(r['path'])==r['sha256'],r['path']
    for r in OLD_FILES.values():assert sha(r['path'])==r['sha256'],r['path']

def fit_one(m,df,strength,digest):
    verify_locked();CURRENT.update(phase='FORMAL_FIT',target=strength)
    prefix='sensitivity_lambda_'+str(strength).replace('.','p');FORMAL_ATTEMPTS.append(strength)
    write(m.OUT/(prefix+'_started.json'),dict(started_utc=utc(),target_lambda=strength,continuation=[v for v in [8.,4.,2.] if v>strength]+[strength],seed=m.SEED,start='original initial(df), perturb=False',execution_checksum=digest))
    m.PENALTY_STRENGTH=strength
    original_evaluate=m.evaluate;original_minimize=m.minimize;count=0;accepted_count=0;last_accepted=None
    theta_stream=(m.OUT/(prefix+'_evaluated_theta_stream.npy')).open('xb')
    accepted_stream=(m.OUT/(prefix+'_accepted_theta_stream.npy')).open('xb')
    event_log=(m.OUT/(prefix+'_evaluation_trace.jsonl')).open('x',encoding='utf-8')
    def traced_evaluate(theta,*args):
        nonlocal count
        count+=1;np.save(theta_stream,theta,allow_pickle=False);theta_stream.flush()
        meta=dict(evaluation=count,phase=CURRENT['phase'],target_lambda=strength,evaluation_lambda=m.PENALTY_STRENGTH,utc=utc())
        try:
            result=original_evaluate(theta,*args)
            meta.update(objective=float(result[0]),gradient_inf=float(np.max(np.abs(result[1]))),diagnostics=result[3].tolist(),status='FINITE')
        except Exception as error:
            failure=m.OUT/(prefix+'_failure');failure.mkdir()
            np.save(failure/'throwing_theta.npy',theta)
            if last_accepted is not None:np.save(failure/'last_accepted_theta.npy',last_accepted)
            np.savez_compressed(failure/'throwing_input_arrays.npz',**dict(zip(['x','dt','bands','si','sj','weights'],args)))
            tb=traceback.format_exc();(failure/'traceback.txt').write_text(tb,encoding='utf-8')
            meta.update(status='FAIL',error_type=type(error).__name__,error=str(error));write(failure/'evaluation.json',meta)
            event_log.write(json.dumps(meta,allow_nan=False)+'\n');event_log.flush()
            raise
        event_log.write(json.dumps(meta,allow_nan=False)+'\n');event_log.flush();return result
    def traced_minimize(fun,start,**kwargs):
        original_callback=kwargs['callback'];stage=m.PENALTY_STRENGTH
        def callback(theta):
            nonlocal accepted_count,last_accepted
            accepted_count+=1;last_accepted=theta.copy();np.save(accepted_stream,theta,allow_pickle=False);accepted_stream.flush()
            original_callback(theta)
        kwargs['callback']=callback
        print('SOLVER_STAGE_START',strength,stage,utc(),flush=True)
        result=original_minimize(fun,start,**kwargs)
        print('SOLVER_STAGE_END',strength,stage,bool(result.success),result.nit,utc(),flush=True)
        return result
    m.evaluate=traced_evaluate;m.minimize=traced_minimize
    try:
        start=m.initial(df,25);assert np.array_equal(start,np.load(OUT/'preflight/original_initial_theta.npy'))
        fit,theta,args,_=m.fit(df,'full',start,prefix)
        CURRENT['phase']='INFORMATION'
        info=m.information(theta,args,df,prefix)
        v,g,scores,diag=m.evaluate(theta,*args[:-1])
        np.save(m.OUT/(prefix+'_theta.npy'),theta);np.save(m.OUT/(prefix+'_analytic_gradient.npy'),g)
        np.save(m.OUT/(prefix+'_individual_data_scores.npy'),scores)
        hess=np.load(m.OUT/(prefix+'_observed_information.npy'))
        if info['qualified']:np.save(m.OUT/(prefix+'_inverse_penalized_information.npy'),np.linalg.inv(hess))
        checks=dict(optimizer_success=bool(fit['optimizer_success']),total_gradient=fit['gradient_inf_total']<=1e-4,information=bool(info['qualified']),probability=bool(diag[2]<=1e-9 and diag[3]>=-1e-9 and diag[4]<=1+1e-9),finite_theta=bool(np.isfinite(theta).all()),finite_gradient=bool(np.isfinite(g).all()))
        status='PASS' if all(checks.values()) else 'FAIL'
        write(m.OUT/(prefix+'_status.json'),dict(status=status,checks=checks,fit=fit,information=info,evaluations_recorded=count,accepted_theta_recorded=accepted_count,execution_checksum=digest))
        if status!='PASS':raise RuntimeError('STOP: formal numerical qualification failed: '+prefix)
        print('TARGET_MODEL_PASS',strength,utc(),flush=True)
    finally:
        m.evaluate=original_evaluate;m.minimize=original_minimize
        theta_stream.close();accepted_stream.close();event_log.close()

def status_table():
    d=pd.read_csv(V7/'clhls_refit_diagnostics_by_frame_lambda_adjudicated.csv')
    d['baseline_status']=d['status'];d['current_evidence']='FROZEN_V7_RETAINED'
    for strength in [.5,2.]:
        mask=(d.frame==2014)&(d.penalty_strength==strength)
        p=OUT/'models/frame2014'/('sensitivity_lambda_'+str(strength).replace('.','p')+'_status.json')
        if p.exists():
            row=load(p);f=row['fit'];inf=row['information'];d.loc[mask,'status']=row['status']
            for k in ['penalized_objective','nll','penalty','gradient_inf_total','optimizer_success','nit','nfev','seconds']:d.loc[mask,k]=f[k]
            for k in ['minimum_eigenvalue','condition_number','qualified']:d.loc[mask,k]=inf[k]
            prefix=p.name.removesuffix('_status.json');d.loc[mask,'fit_path']=str(p.parent/(prefix+'_fit.json'));d.loc[mask,'information_path']=str(p.parent/(prefix+'_information.json'))
            d.loc[mask,'fit_sha256']=sha(p.parent/(prefix+'_fit.json'));d.loc[mask,'information_sha256']=sha(p.parent/(prefix+'_information.json'))
            d.loc[mask,'current_evidence']='CONTROLLED_NEW_VERSION'
        elif strength in FORMAL_ATTEMPTS:d.loc[mask,'status']='FAIL';d.loc[mask,'current_evidence']='CONTROLLED_ATTEMPT_STOPPED'
    d.to_csv(OUT/'execution_status_crosswalk.csv',index=False,encoding='utf-8-sig')

def main():
    write(OUT/'execution_started.json',dict(started_utc=utc(),scope='AUTHORIZED_CONTROLLED_FORMAL_RECOVERY',g4_g5='NOT_RUN',submission='BLOCK_SUBMISSION'))
    try:
        m,df=preflight();digest=lock(m)
        print('PREFLIGHT_PASS_AUTO_START_AUTHORIZED_FORMAL_RUN',utc(),flush=True)
        for strength in [.5,2.]:fit_one(m,df,strength,digest)
        verify_locked();status_table()
        write(OUT/'formal_recovery_completed.json',dict(status='PASS',completed_utc=utc(),target_models=[.5,2.],previous_seven_refitted=False,old_files_unchanged=True,g4_g5='NOT_RUN',word='UNCHANGED',submission='BLOCK_SUBMISSION',next='AUTHORIZED_KEY_COMPARISON_AND_DELIVERY_ONLY'))
    except Exception as error:
        tb=traceback.format_exc();(OUT/'execution_failure_traceback.txt').write_text(tb,encoding='utf-8')
        status_table();write(OUT/'execution_stopped.json',dict(status='FAIL',stopped_utc=utc(),phase=CURRENT,error_type=type(error).__name__,error=str(error),formal_targets_attempted=FORMAL_ATTEMPTS,g4_g5='NOT_RUN',submission='BLOCK_SUBMISSION'))
        raise

if __name__=='__main__':main()
