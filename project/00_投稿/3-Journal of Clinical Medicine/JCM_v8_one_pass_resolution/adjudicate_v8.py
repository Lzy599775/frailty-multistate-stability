"""Bounded A/B comparison; separate computation from scientific adjudication."""
import sys,itertools
import numpy as np
import pandas as pd
import study_v8 as s
import evidence_v8 as e
from fit_runner import Interface
from repair_reporting_records import qualification_record

def load_theta(c):
    p=s.Path(c['lineage']['parameter_path'])
    return np.asarray(s.js(p)['theta'],float) if p.suffix=='.json' else np.load(p)

def observed(c):
    p=s.Path(c['lineage']['parameter_path'])
    if c['start']!='frozen':return np.load(p.parent/'observed_logp.npy')
    return Interface(s.old.sample(c['frame']),c['frame'],'original',c['lambda_value']).evaluate(load_theta(c),False)[3]

def candidates():
    return [s.js(p) for p in sorted((s.OUT/'diagnostic_evidence').glob('*.json')) if not p.name.startswith('mortality_chain_check')]

def main():
    execution=s.js(s.OUT/'batch_execution.json');assert execution['expected']==execution['executed']==44
    assert sorted(x['index'] for x in execution['receipts'])==list(range(44))
    e.collect();cv=e.heldout();plan=pd.read_csv(s.OUT/'fit_plan.csv');inventory=[]
    for i,r in plan.iterrows():
        dd=list((s.OUT/'attempts').glob(f'{i:02d}_*'));assert len(dd)==1;p=dd[0];result=qualification_record(p);failure=p/'failure.json';assert result.exists() or failure.exists()
        if result.exists():
            z=s.js(result);inventory.append(dict(attempt_index=i,**r.to_dict(),status=z['status'],NUMERICALLY_QUALIFIED=z['NUMERICALLY_QUALIFIED'],INFERENCE_QUALIFIED=z['INFERENCE_QUALIFIED'],parameter_count=z['parameter_count'],n_intervals=z['n_intervals'],n_persons=z['n_persons'],penalized_objective=z['penalized_objective'],nll=z['nll'],penalty=z['penalty'],gradient_inf_total=z['gradient_inf_total'],minimum_information_eigenvalue=z['information']['minimum_eigenvalue'],condition_number=z['information']['condition_number'],nit=z['nit'],nfev=z['nfev'],optimizer_message=s.js(p/'optimizer.json')['message'],seconds=z['seconds'],result_path=str(result),reporting_reconstruction=z.get('record_origin','NONE'),original_reporting_failure_path=z.get('original_reporting_failure_path','')))
        else:
            z=s.js(failure);inventory.append(dict(attempt_index=i,**r.to_dict(),status='FAIL',NUMERICALLY_QUALIFIED=False,INFERENCE_QUALIFIED=False,failing_stage=z['stage'],error_type=z['error_type'],error=z['error'],seconds=z['seconds'],failure_path=str(failure)))
    s.save(s.OUT/'fit_execution_inventory.csv',inventory)
    cs=candidates();selected=[];pair_rows=[];core=[];source=[]
    for c in cs:
        common={k:c[k] for k in ['candidate_id','frame','model','lambda_value','start','penalized_objective','nll','penalty','gradient_inf_total','qualified','inference_qualified']}
        info=c['information'];common.update(minimum_information_eigenvalue=info['minimum_eigenvalue'],condition_number=info['condition_number'],regularization_information_fraction=info.get('regularization_information_fraction'),**c['lineage'])
        for tr in c['TIR']:core.append(dict(**common,record_type='transition_intensity_ratio',**tr))
        for mr in c['mortality']:core.append(dict(**common,record_type='mortality',**mr))
        source.append(dict(candidate_id=c['candidate_id'],frame=c['frame'],model=c['model'],lambda_value=c['lambda_value'],start=c['start'],**c['lineage']))
    for kind in ['original','compact']:
        for frame in s.FRAMES:
            for lam,_ in s.RUNS:
                group=[c for c in cs if c['model']==kind and c['frame']==frame and c['lambda_value']==lam]
                if not group:
                    selected.append(dict(model=kind,frame=frame,lambda_value=lam,status='NOT_ESTIMABLE',reason='No qualified registered fit'));continue
                best=min(group,key=lambda c:c['penalized_objective'])
                selected.append(dict(model=kind,frame=frame,lambda_value=lam,status='PASS',candidate_id=best['candidate_id'],penalized_objective=best['penalized_objective'],qualified_candidates=len(group),original_frozen_replaced_in_history=False))
                for c1,c2 in itertools.combinations(group,2):
                    delta=abs(c1['penalized_objective']-c2['penalized_objective']);pdiff=float(abs(load_theta(c1)-load_theta(c2)).max());tdiff=max(abs(x['logTIR']-y['logTIR']) for x,y in zip(c1['TIR'],c2['TIR']));p1=observed(c1);p2=observed(c2)
                    reversals=[x['transition']+':'+x['contrast'] for x,y in zip(c1['TIR'],c2['TIR']) if x['logTIR']*y['logTIR']<0]
                    label='MULTIMODAL_LOCAL_SOLUTIONS' if delta>1e-6 else ('PRACTICAL_NONIDENTIFICATION_CANDIDATE' if tdiff>1e-3 else ('PARAMETER_DISAGREEMENT' if pdiff>1e-3 else 'AGREEMENT'))
                    pair_rows.append(dict(model=kind,frame=frame,lambda_value=lam,candidate_1=c1['candidate_id'],candidate_2=c2['candidate_id'],objective_abs_difference=delta,parameter_max_difference=pdiff,key_logTIR_max_difference=tdiff,observed_probability_max_difference=float(abs(np.exp(p1)-np.exp(p2)).max()),observed_logprobability_RMS_difference=float(np.sqrt(np.square(p1-p2).mean())),key_direction_reversal_count=len(reversals),key_direction_reversals='|'.join(reversals),classification=label))
    s.save(s.OUT/'model_comparison_core.csv',core);s.save(s.OUT/'same_lambda_solution_comparison.csv',pair_rows);s.save(s.OUT/'selected_diagnostic_candidates.csv',selected);s.save(s.OUT/'parameter_covariance_source_index.csv',source)
    sel={r['candidate_id'] for r in selected if r['status']=='PASS'};selected_core=[r for r in core if r['candidate_id'] in sel];s.save(s.OUT/'selected_candidate_core_estimates.csv',selected_core)
    support=[]
    for candidate in [c for c in cs if c['candidate_id'] in sel]:
        pp=s.Path(candidate['lineage']['parameter_path']);kind=candidate['model'];frame=candidate['frame'];lam=candidate['lambda_value']
        hp=(pp.parent/(pp.name.replace('_fit.json','_observed_information.npy'))) if candidate['start']=='frozen' else pp.parent/'penalized_information.npy'
        hess=np.load(hp);bread=np.linalg.inv(hess);_,parameters,_=s.mapping(frame,kind);_,prec=s.penalty(parameters,lam);datah=hess-np.diag(prec)
        for row in candidate['TIR']:
            path=tuple(map(int,row['transition'].split('->')));active=0 if row['contrast']=='sleep5_vs7_inactive' else 1;sleep=7. if row['contrast']=='active_vs_inactive_sleep7' else 5.;ref=0 if row['contrast']=='active_vs_inactive_sleep7' else active
            c=e.contrast(frame,kind,path,sleep,active,ref);direction=bread@c;total=float(direction@hess@direction);pen=float(np.dot(prec*direction,direction));data=float(direction@datah@direction)
            assert total>0 and abs(total-pen-data)<=1e-7*max(1.,total)
            support.append(dict(candidate_id=candidate['candidate_id'],frame=frame,model=kind,lambda_value=lam,transition=row['transition'],contrast=row['contrast'],inverse_penalized_contrast_curvature=total,data_direction_curvature=data,penalty_direction_curvature=pen,penalty_direction_fraction=pen/total,interpretation='Curvature along Hpen_inverse times contrast, not sandwich variance fraction; data curvature may be negative',information_path=str(hp),information_sha256=s.sha(hp)))
    s.save(s.OUT/'contrast_information_support.csv',support)
    counts=[]
    for frame in s.FRAMES:
        d=s.old.sample(frame)
        for (i,j),g in d.groupby(['i','j'],sort=True):counts.append(dict(frame=frame,start_state=int(i),end_state=int(j),observed_endpoint_intervals=len(g),unique_people=g.person_id.nunique(),interpretation='Observed interval endpoints, not unobserved internal CTMC jumps'))
    s.save(s.OUT/'observed_endpoint_support.csv',counts)
    sensitivity=[]
    for kind in ['original','compact']:
        for frame in s.FRAMES:
            for path in ['1->0','2->0','2->1']:
                for contrast in ['active_vs_inactive_sleep7','sleep5_vs7_active','sleep5_vs7_inactive']:
                    rr=sorted([r for r in selected_core if r['model']==kind and r['frame']==frame and r.get('transition')==path and r['contrast']==contrast],key=lambda r:r['lambda_value'])
                    if len(rr)!=3:continue
                    logs=[r['logTIR'] for r in rr];values=[r['TIR'] for r in rr];sensitivity.append(dict(model=kind,frame=frame,transition=path,contrast=contrast,estimate_lambda0p5=values[0],estimate_lambda1=values[1],estimate_lambda2=values[2],lambda_log_range=max(logs)-min(logs),max_min_ratio=max(values)/min(values),direction_consistent=bool(min(logs)>=0 or max(logs)<=0),status='DIRECTION_SENSITIVE' if min(logs)<0<max(logs) else 'DIRECTION_CONSISTENT_MAGNITUDE_REPORTED'))
            rr=sorted([r for r in selected_core if r['model']==kind and r['frame']==frame and r['contrast']=='mortality_active_minus_inactive_7h' and r['scenario']==e.LABELS[0]],key=lambda r:r['lambda_value'])
            if len(rr)==3:
                values=[r['estimate'] for r in rr];sensitivity.append(dict(model=kind,frame=frame,contrast='mortality_active_minus_inactive_7h',estimate_lambda0p5=values[0],estimate_lambda1=values[1],estimate_lambda2=values[2],absolute_range=max(values)-min(values),direction_consistent=bool(min(values)>=0 or max(values)<=0),status='DIRECTION_SENSITIVE' if min(values)<0<max(values) else 'DIRECTION_CONSISTENT_MAGNITUDE_REPORTED'))
    s.save(s.OUT/'cross_lambda_scientific_sensitivity.csv',sensitivity)
    # Mechanical evidence flags only; final prose adjudication also examines
    # assumptions, covariance, uncertainty and the predefined central targets.
    central=[('CHARLS','2->0','active_vs_inactive_sleep7'),('CHARLS','1->0','sleep5_vs7_active'),('CHARLS','2->1','sleep5_vs7_active'),('2014','1->0','active_vs_inactive_sleep7')]
    summary={}
    for kind in ['original','compact']:
        centres=[r for r in sensitivity if r['model']==kind and (r['frame'],r.get('transition'),r['contrast']) in central]
        pair=[r for r in pair_rows if r['model']==kind];inv=[r for r in inventory if r['model']==kind and r['partition']=='full']
        summary[kind]=dict(full_attempts=len(inv),full_numeric_pass=sum(r['status']=='PASS' for r in inv),registered_full_failures=sum(r['status']!='PASS' for r in inv),same_lambda_pair_count=len(pair),same_lambda_disagreement_count=sum(r['classification']!='AGREEMENT' for r in pair),central_targets=centres,any_central_direction_reversal=any(not r['direction_consistent'] for r in centres),selected_all_frame_lambda=len([r for r in selected if r['model']==kind and r['status']=='PASS'])==12)
    s.write(s.OUT/'analysis_summary.json',dict(utc=s.utc(),attempts=44,numeric_pass=sum(r['status']=='PASS' for r in inventory),numeric_fail=sum(r['status']!='PASS' for r in inventory),models=summary,heldout=cv,scientific_decision='PENDING_EVIDENCE_REVIEW'))
    print('A_B_EVIDENCE_COMPLETE',summary,flush=True)

if __name__=='__main__':main()
