"""Conditional CHARLS matched community variance, never refits a model."""
import numpy as np
import pandas as pd
import study_v8 as s
import evidence_v8 as e
from adjudicate_v8 import candidates,load_theta
from fit_runner import Interface

def main():
    decision=s.js(s.OUT/'scientific_adjudication.json')
    if not decision['authorize_manuscript_reconstruction']:raise RuntimeError('STOP scientific gate')
    dest=s.OUT/'final_numeric_source_tables'
    if (dest/'selected_CHARLS_community_covariance.npy').exists():raise FileExistsError('Selected variance output exists; no overwrite')
    selected=pd.read_csv(s.OUT/'selected_diagnostic_candidates.csv');kind=decision['selected_model_family'];cid=selected[selected.model.eq(kind)&selected.frame.eq('CHARLS')&selected.lambda_value.eq(1.)].iloc[0].candidate_id
    c=next(z for z in candidates() if z['candidate_id']==cid);th=load_theta(c);person_cov=np.load(c['lineage']['covariance_path']);df=s.old.sample('CHARLS');a=Interface(df,'CHARLS',kind,1.);_,g,scores,*_=a.evaluate(th);scores=scores@a.mat
    p=s.Path(c['lineage']['parameter_path']);hp=p.parent/('formal_full_v2_observed_information.npy' if c['start']=='frozen' else 'penalized_information.npy');hess=np.load(hp);bread=np.linalg.inv(hess);individual=pd.DataFrame(scores).groupby(df.person_id.to_numpy(),sort=True).sum().to_numpy();check=bread@(individual.T@individual)@bread.T
    assert np.allclose(check,person_cov,rtol=1e-7,atol=1e-9),'STOP selected person covariance does not reproduce'
    prior=pd.read_csv(s.old.V6/'charls_psu_join_audit.csv').iloc[0];src=s.Path(prior.source_path);assert s.sha(src)==prior.source_sha256
    raw=pd.read_csv(src,dtype={'ID':str,'communityID':str},usecols=['ID','communityID']).rename(columns={'ID':'person_id'});conflicts=raw.groupby('person_id',dropna=False).communityID.nunique(dropna=False)
    assert not conflicts.gt(1).any(),'STOP conflicting community mapping'
    linked=df[['person_id']].merge(raw.drop_duplicates('person_id'),on='person_id',how='left',validate='many_to_one',indicator=True);assert len(linked)==len(df) and linked._merge.eq('both').all() and linked.communityID.notna().all()
    grouped=pd.DataFrame(scores).groupby(linked.communityID.to_numpy(),sort=True).sum().to_numpy();G=len(grouped);n=len(df);p=len(th);assert G==414 and n==5720
    correction=G/(G-1)*(n-1)/(n-p);cov=bread@(grouped.T@grouped)@bread.T*correction;assert np.isfinite(cov).all() and abs(cov-cov.T).max()<1e-8
    cp=dest/'selected_CHARLS_community_covariance.npy';np.save(cp,cov);rows=[]
    for path in s.TRANS:
        contrast=e.contrast('CHARLS',kind,path,7.,1,0)
        for basis,cc in [('subject',person_cov),('community414',cov)]:
            estimate=s.old.estimate(contrast,th,cc);assert estimate['status']=='PASS';rows.append(dict(row_id=f'{cid}_{path[0]}to{path[1]}_{basis}',frame='CHARLS',model=kind,candidate_id=cid,lambda_value=1.,transition=f'{path[0]}->{path[1]}',variance_basis=basis,**estimate,correction=correction if basis=='community414' else 1.,n_clusters=G if basis=='community414' else n,parameter_count=p,parameter_path=c['lineage']['parameter_path'],parameter_sha256=c['lineage']['parameter_sha256'],covariance_path=str(cp) if basis=='community414' else c['lineage']['covariance_path'],covariance_sha256=s.sha(cp) if basis=='community414' else c['lineage']['covariance_sha256'],scope='Community sensitivity, not full stratified survey variance'))
    s.save(dest/'selected_CHARLS_community_contrasts.csv',rows)
    s.save(dest/'selected_CHARLS_community_join_audit.csv',[dict(input_fit_rows=n,input_mapping_rows=len(raw),unique_mapping_persons=raw.person_id.nunique(),duplicate_mapping_persons=int(raw.person_id.duplicated().sum()),conflicting_keys=int(conflicts.gt(1).sum()),output_rows=len(linked),unmatched=int(linked._merge.ne('both').sum()),missing_community=int(linked.communityID.isna().sum()),clusters=G,source_path=str(src),source_sha256=s.sha(src),parameter_path=c['lineage']['parameter_path'],information_path=str(hp),information_sha256=s.sha(hp),person_covariance_max_reconstruction_error=float(abs(check-person_cov).max()))])
    # Retain the qualified historical original-model sensitivity, explicitly
    # separate from the selected compact model; never reuse its variance there.
    old=pd.read_csv(s.G4/'g4_CHARLS_retained_community_contrasts.csv');old['model']='original_225_historical';old['use']='Historical model sensitivity only; not selected compact covariance';s.save(dest/'retained_original_CHARLS_community_contrasts.csv',old.to_dict('records'))
    s.write(dest/'community_variance_completion.json',dict(utc=s.utc(),status='PASS',candidate_id=cid,parameters=p,clusters=G,finite_cluster_correction=correction,model_refit_performed=False,full_complex_survey_variance='NOT_ESTIMATED'))
    print('SELECTED_COMMUNITY_VARIANCE_PASS',cid,G,flush=True)

if __name__=='__main__':main()
