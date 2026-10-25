"""Conditional final numerical sources, only after a positive scientific gate."""
import numpy as np
import pandas as pd
from scipy.special import expit,logit
import study_v8 as s
import evidence_v8 as e
from adjudicate_v8 import candidates,load_theta

def main():
    decision=s.js(s.OUT/'scientific_adjudication.json')
    if not decision['authorize_manuscript_reconstruction']:raise RuntimeError('STOP scientific gate does not authorize manuscript reconstruction')
    kind=decision['selected_model_family'];selected=pd.read_csv(s.OUT/'selected_diagnostic_candidates.csv');bank={c['candidate_id']:c for c in candidates()};dest=s.OUT/'final_numeric_source_tables';dest.mkdir(exist_ok=False)
    rows=[];curves=[];mort=[];scenarios=[];lineage=[];baseline=[];endpoint=[]
    for frame in s.FRAMES:
        df=s.old.sample(frame);d=df.sort_values(['person_id','wave_start'],kind='stable').drop_duplicates('person_id');w=d.weight.to_numpy(float);states=d.i.to_numpy()
        base=dict(frame=frame,n_people=len(d),n_intervals=len(df),robust=int(d.i.eq(0).sum()),prefrail=int(d.i.eq(1).sum()),frail=int(d.i.eq(2).sum()),inactive=int(d.activity.eq(0).sum()),active=int(d.activity.eq(1).sum()),different_endpoints=int(df.i.ne(df.j).sum()),same_living_state=int(df.i.eq(df.j).sum()),death_endpoints=int(df.j.eq(3).sum()),mean_intervals=len(df)/len(d))
        for (i,j),gg in df.groupby(['i','j'],sort=True):endpoint.append(dict(frame=frame,i=int(i),j=int(j),count=len(gg)))
        for (lo,hi),gg in df.groupby(['wave_start','wave_end'],sort=True):base[f'intervals_{lo}_{hi}']=len(gg)
        baseline.append(base)
        for lam,_ in s.RUNS:
            rr=selected[selected.model.eq(kind)&selected.frame.eq(frame)&selected.lambda_value.eq(lam)];assert len(rr)==1 and rr.iloc[0].status=='PASS';candidate=bank[rr.iloc[0].candidate_id];th=load_theta(candidate);cov=np.load(candidate['lineage']['covariance_path']);common=dict(frame=frame,model=kind,lambda_value=lam,candidate_id=candidate['candidate_id'],**candidate['lineage'])
            lineage.append(dict(**common,parameter_count=len(th),NUMERICALLY_QUALIFIED=candidate['qualified'],INFERENCE_QUALIFIED=candidate['inference_qualified'],gradient_inf_total=candidate['gradient_inf_total']))
            # Reuse unchanged verified exports; the user's v8 request forbids
            # another 972-item arithmetic audit of the old original family.
            if kind=='original' and candidate['start']=='frozen':
                tt=pd.read_csv(s.G4/'g4_972_all_lambda_TIR.csv');tt=tt[tt.frame.eq(frame)&tt.lambda_value.eq(lam)]
                for r in tt.to_dict('records'):
                    fields={k:r[k] for k in ['transition','contrast','sleep_hours','activity','reference_activity','reference_sleep','logTIR','variance','TIR','se','ci_lower','ci_upper','status']}
                    rows.append(dict(row_id=r['row_id'],**common,**fields,inference_scope='Reused unchanged G4 verified estimate; approximate conditional penalized sandwich'))
                if lam==1.:
                    cc=pd.read_csv(s.G4/'g4_figure2_corrected_mesh.csv');cc=cc[cc.frame.eq(frame)&cc.sleep_hours.between(5.,9.)]
                    for r in cc.to_dict('records'):
                        fields={k:r[k] for k in ['transition','sleep_hours','activity','logTIR','variance','TIR','se','ci_lower','ci_upper','status']}
                        curves.append(dict(row_id=f'{candidate["candidate_id"]}_{r["transition"]}_s{r["sleep_hours"]:g}_a{r["activity"]}',**common,**fields))
                mm=pd.read_csv(s.G4/'g4_mortality_all_lambda_profiles.csv');mm=mm[mm.frame.eq(frame)&mm.lambda_value.eq(lam)]
                for r in mm.to_dict('records'):
                    fields={k:r[k] for k in ['record_type','scenario','horizon_years','sleep_hours','activity','reference_activity','se','ci_lower','ci_upper','status']}
                    fields['estimate']=r['probability'] if r['record_type']=='probability' else r['difference_active_minus_inactive']
                    mort.append(dict(row_id=r['row_id'],**common,**fields,inference_scope='Reused unchanged G4 verified probability/paired result'))
                for r in candidate['mortality']:scenarios.append(dict(**common,**r))
                print('REUSED_UNCHANGED_G4_SELECTED',kind,frame,lam,flush=True)
                continue
            for path in s.TRANS:
                for label,sleep,active,ref in [('active_vs_inactive_sleep7',7.,1,0)]+[(f'sleep{sl:g}_vs7_{"active" if ac else "inactive"}',float(sl),ac,ac) for sl in [5,6,8,9] for ac in [0,1]]:
                    cc=e.contrast(frame,kind,path,sleep,active,ref);estimate=s.old.estimate(cc,th,cov)
                    if estimate['status']!='PASS':raise RuntimeError(f'STOP unqualified final contrast {common} {path} {label}')
                    rows.append(dict(row_id=f'{candidate["candidate_id"]}_{path[0]}to{path[1]}_{label}',**common,transition=f'{path[0]}->{path[1]}',contrast=label,sleep_hours=sleep,activity=active,reference_activity=ref,reference_sleep=7.,**estimate,inference_scope='Approximate conditional penalized subject sandwich; no multiplicity correction'))
                if lam==1.:
                    for active in [0,1]:
                        for sleep in np.linspace(5.,9.,81):
                            cc=e.contrast(frame,kind,path,float(sleep),active,active)
                            if sleep==7.:estimate=dict(logTIR=0.,TIR=1.,ci_lower=1.,ci_upper=1.,se=0.,variance=0.,status='REFERENCE')
                            else:estimate=s.old.estimate(cc,th,cov)
                            if estimate['status'] not in ['PASS','REFERENCE']:raise RuntimeError('STOP final curve uncertainty not estimable')
                            curves.append(dict(row_id=f'{candidate["candidate_id"]}_{path[0]}to{path[1]}_s{sleep:g}_a{active}',**common,transition=f'{path[0]}->{path[1]}',sleep_hours=float(sleep),activity=active,**estimate))
            results={}
            for sleep in [5.,6.,7.,8.,9.]:
                for active in [0,1]:
                    pp,gg=e.probability_gradient(th,d,frame,kind,sleep,active);ww=w/w.sum();value=float(ww@pp);grad=np.sum(ww[:,None]*gg,axis=0);var=float(grad@cov@grad)
                    if not 0<value<1 or not var>0:raise RuntimeError('STOP final probability or covariance unqualified')
                    se=np.sqrt(var);lo,hi=expit(logit(value)+np.array([-1,1])*1.96*se/(value*(1-value)));results[(sleep,active)]=(value,grad)
                    mort.append(dict(row_id=f'{candidate["candidate_id"]}_mortality_s{sleep:g}_a{active}',**common,record_type='probability',scenario=e.LABELS[0],horizon_years=2,sleep_hours=sleep,activity=active,estimate=value,se=float(se),ci_lower=float(lo),ci_upper=float(hi),status='PASS',inference_scope='Conditional logit delta; empirical population fixed'))
            value=results[(7.,1)][0]-results[(7.,0)][0];grad=results[(7.,1)][1]-results[(7.,0)][1];var=float(grad@cov@grad);assert var>0;se=float(np.sqrt(var))
            mort.append(dict(row_id=f'{candidate["candidate_id"]}_mortality_paired7',**common,record_type='paired_activity_difference',scenario=e.LABELS[0],horizon_years=2,sleep_hours=7,activity=1,reference_activity=0,estimate=value,se=se,ci_lower=value-1.96*se,ci_upper=value+1.96*se,status='PASS',inference_scope='Paired shared-covariance delta on difference scale'))
            for r in candidate['mortality']:scenarios.append(dict(**common,**r))
            print('FINAL_SELECTED_SOURCES',kind,frame,lam,flush=True)
    assert len(rows)==972 and sum(r['lambda_value']==1 for r in rows)==324 and len(mort)==132
    if kind=='compact':assert len(curves)==5832
    s.save(dest/'all_lambda_TIR.csv',rows);s.save(dest/'main_324_TIR.csv',[r for r in rows if r['lambda_value']==1]);s.save(dest/'main_sleep_curves.csv',curves);s.save(dest/'all_lambda_mortality.csv',mort);s.save(dest/'main_40_probabilities_and_4_paired.csv',[r for r in mort if r['lambda_value']==1]);s.save(dest/'initial_state_scenarios_7h.csv',scenarios);s.save(dest/'parameter_covariance_lineage.csv',lineage);s.save(dest/'sample_characteristics.csv',baseline);s.save(dest/'observed_endpoint_pairs.csv',endpoint)
    s.write(dest/'completion.json',dict(utc=s.utc(),status='PASS',model_family=kind,primary_TIR=sum(r['lambda_value']==1 for r in rows),all_lambda_TIR=len(rows),curves=len(curves),main_probabilities=sum(r['lambda_value']==1 and r['record_type']=='probability' for r in mort),main_paired_differences=sum(r['lambda_value']==1 and r['record_type']=='paired_activity_difference' for r in mort),IPCW='WITHDRAWN_NOT_RECREATED',versions=[r['candidate_id'] for r in lineage],scientific_adjudication_sha256=s.sha(s.OUT/'scientific_adjudication.json')))

if __name__=='__main__':main()
