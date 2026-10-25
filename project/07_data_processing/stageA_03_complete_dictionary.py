"""Stage A source/export reconciliation and complete candidate dictionary. No models."""
from pathlib import Path
import hashlib, json, re
import pandas as pd

ROOT=Path(r'F:\A_CHARLS+CLHLS\00_新选题')
BASE=ROOT/'08_intermediate/analysis_ready_stageA_v1_20261010'
OUT=BASE/'reconciled_dictionary'

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def domains(v,label,cohort):
    t=(v+' '+label).lower(); z=[]
    patterns={'sleep':r'sleep|nap|restless', 'activity':r'exercise|physical activity|vigorous|moderate physical|light physical',
      'vital':r'death|died|surviv|lost to follow|interview status', 'date':r'interview.*(year|month|day)|year.*interview|month.*interview|day.*interview|validated.*death',
      'design':r'weight|psu|strat|community|county', 'covariate':r'gender|sex$|validated age|age in years|birth year|education|schooling|marital|smok|drink|residence'}
    for k,p in patterns.items():
        if re.search(p,t): z.append(k)
    if (cohort=='CHARLS' and (re.match(r'^(db\d|dc\d|da007_|da001$|da002$|da033$|da034$|da039$|da041$|zdisease|r[1-4](?:hibp|diab|cancr|lung|heart|stroke|arthr|kidney|liver|digst|memry|psyche|dyslip|asthma|shlt|sight|hear|cesd|cog|adl|iadl))',v))) or (cohort=='CLHLS' and re.match(r'^(e(?:[1-9]|1[0-4])(?:_|$)|g15[a-v]1(?:_|$)|b(?:12|2[1-7])(?:_|$)|g(?:1|81|82|83|9|11|130|131)(?:_|$)|h1(?:_|$)|c\d)',v)):
        z.append('FI_candidate')
    if re.search(r'sleep|restless|exercise or not|physical activity',t): z.append('exclude_from_main_FI_exposure_overlap')
    return ';'.join(z)

def main():
    OUT.mkdir(exist_ok=True)
    ci=pd.read_csv(BASE/'charls_sources/charls_export_inventory.csv',keep_default_na=False,dtype=str)
    li=pd.read_csv(BASE/'clhls_sources/clhls_export_inventory.csv',keep_default_na=False,dtype=str)
    sources=[]; profiles=[]; clhls_shas={}
    for r in ci.to_dict('records'):
        # Read strings to preserve leading-zero IDs and tagged missing values.
        d=pd.read_csv(r['export_file'],dtype=str,keep_default_na=False)
        ident=r['id_column'] or ('communityID' if 'communityID' in d else '')
        ids=d[ident] if ident else pd.Series(dtype=str)
        valid=ids.ne('')
        eh=sha(r['export_file'])
        assert len(d)==int(r['rows']) and d.shape[1]==int(r['cols'])
        assert eh==r['export_sha256']
        sources.append(dict(cohort='CHARLS',wave=r['wave'],source_file=r['source_file'],source_sha256=r['sha256'],source_hash_basis='export-time source bytes',export_file=r['export_file'],export_sha256=eh,rows=len(d),cols=d.shape[1],id_column=ident,id_missing=int((~valid).sum()),id_unique=int(ids[valid].nunique()),id_duplicate_valid=int(ids[valid].duplicated().sum()),shape_basis='fresh CSV parse reconciled with source export shape',dropped=0))
        print('RECONCILED CHARLS',r['wave'],Path(r['source_file']).name,flush=True)
    for r in li.to_dict('records'):
        sp=ROOT/'06_audit/clhls_verified_extract'/r['file']
        ep=BASE/'clhls_sources'/f"{Path(r['file']).stem}_raw.csv"
        d=pd.read_csv(ep,dtype=str,keep_default_na=False)
        assert len(d)==int(r['rows']) and d.shape[1]==int(r['cols'])
        ids=d['id']; valid=ids.ne(''); h=sha(sp); clhls_shas[r['file']]=h
        sources.append(dict(cohort='CLHLS',wave=r['file'][6:10],source_file=str(sp),source_sha256=h,source_hash_basis='exact SAV extracted copy bytes; original archive provenance in earlier extraction record',export_file=str(ep),export_sha256=sha(ep),rds_file=str(ep.with_name(ep.name.replace('_raw.csv','_labelled.rds'))),rows=len(d),cols=d.shape[1],id_column='id',id_missing=int((~valid).sum()),id_unique=int(ids[valid].nunique()),id_duplicate_valid=int(ids[valid].duplicated().sum()),shape_basis='fresh CSV parse reconciled with R/haven source shape',dropped=0))
        for v in d:
            s=d[v]; n=pd.to_numeric(s,errors='coerce'); u=s[s.ne('')].nunique()
            profiles.append(dict(cohort='CLHLS',source_file=str(sp),variable=v,n_rows=len(d),n_missing_system=int(s.eq('').sum()),n_unique=int(u),n_non_numeric_nonempty=int((s.ne('')&n.isna()).sum()),n_negative_numeric=int(n.lt(0).sum()),numeric_min=n.min(),numeric_max=n.max(),counts_json=json.dumps(s.value_counts(dropna=False).to_dict(),ensure_ascii=False) if u<150 else 'HIGH_CARDINALITY'))
        print('RECONCILED CLHLS',r['file'],flush=True)
    sd=pd.DataFrame(sources); sd.to_csv(OUT/'source_inventory_reconciled_v1.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(profiles).to_csv(OUT/'clhls_profiles_complete_v1.csv',index=False,encoding='utf-8-sig')
    c=pd.read_csv(BASE/'charls_sources/charls_dictionary_complete.csv',dtype=str,keep_default_na=False).rename(columns={'value_labels_json':'value_labels','user_missing_json':'user_missing'})
    l=pd.read_csv(BASE/'clhls_sources/clhls_dictionary_utf8.csv',dtype=str,keep_default_na=False).rename(columns={'variable':'source_variable'})
    l['source_basename']=l.source_file; l['source_sha256']=l.source_file.map(clhls_shas)
    l['export_file']=l.source_file.map(lambda f:str(BASE/'clhls_sources'/f'{Path(f).stem}_raw.csv'))
    l['wave']=l.source_file.str[6:10]; l['source_file']=l.source_file.map(lambda f:str(ROOT/'06_audit/clhls_verified_extract'/f)); l['cohort']='CLHLS'
    d=pd.concat([c,l],ignore_index=True).fillna('')
    d['candidate_domains']=[domains(v,lb,ch) for v,lb,ch in zip(d.source_variable,d.label,d.cohort)]
    d['role_status']='SOURCE_DICTIONARY_ONLY_NOT_APPROVED_FI'
    d.to_csv(OUT/'variable_crosswalk_complete_v1.csv',index=False,encoding='utf-8-sig')
    d[d.candidate_domains.str.contains('FI_candidate',regex=False)].to_csv(OUT/'fi_item_candidates_v1.csv',index=False,encoding='utf-8-sig')
    report={'stage':'A_DICTIONARY_AND_SOURCE_EXPORT_RECONCILED','sources':len(sd),'dictionary_rows':len(d),'FI_candidate_rows':int(d.candidate_domains.str.contains('FI_candidate',regex=False).sum()),'FI_scoring':'NOT RUN','formal_models':'NOT RUN','raw_modified':False,'limitations':['source inventory replaces earlier incomplete/hardcoded step01 evidence; no existing authority changed','no full-cell RDS-vs-CSV equality claim; shape and IDs reconciled','candidate selection is transparent discovery, not a final FI definition'],'next_task':'build source-level person-wave/status/death-bound tables'}
    (OUT/'step_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__': main()
