"""Source-level longitudinal construction. No sample exclusions, FI, imputation or models."""
from pathlib import Path
import calendar, datetime as dt, hashlib, json, re
import numpy as np
import pandas as pd

ROOT=Path(r'F:\A_CHARLS+CLHLS\00_新选题')
BASE=ROOT/'08_intermediate/analysis_ready_stageA_v1_20261010'
OUT=BASE/'longitudinal_sources'
INV=pd.read_csv(BASE/'charls_sources/charls_export_inventory.csv',dtype=str,keep_default_na=False)
DIC=pd.read_csv(BASE/'reconciled_dictionary/variable_crosswalk_complete_v1.csv',dtype=str,keep_default_na=False)
LOG=[]; MAP=[]; PRODUCTS=[]

def save(d,name):
    p=OUT/name
    if p.exists(): raise FileExistsError(p)
    d.to_csv(p,index=False,encoding='utf-8-sig')
    with p.open('rb') as f: h=hashlib.file_digest(f,'sha256').hexdigest()
    PRODUCTS.append(dict(path=str(p),sha256=h,rows=len(d),cols=d.shape[1]))

def readchar(w,stem):
    r=INV[INV.wave.eq(str(w))&INV.source_file.map(lambda p:Path(p).stem.lower()).eq(stem.lower())]
    if len(r)!=1: raise ValueError((w,stem,len(r)))
    p=r.iloc[0].export_file; d=pd.read_csv(p,dtype=str,keep_default_na=False)
    x=d.ID
    ok=x.str.fullmatch(r'\d{11}' if w==2011 else r'\d{12}')
    LOG.append(dict(cohort='CHARLS',wave=w,source=p,input_rows=len(d),id_invalid=int((~ok).sum()),invalid_source_rows_retained_separately=True))
    if (~ok).any(): save(d.loc[~ok],f'charls_{w}_{stem}_invalid_id_source_rows.csv')
    d=d.loc[ok].copy()
    d['person_id']=x.loc[ok].str[:-2]+'0'+x.loc[ok].str[-2:] if w==2011 else x.loc[ok]
    if d.person_id.duplicated().any(): raise ValueError('Duplicate valid ID: '+p)
    return d,str(r.iloc[0].source_file)

def nums(s): return pd.to_numeric(s,errors='coerce')

def bounds(y,m='',day=''):
    def integer(x):
        try:
            n=float(x)
            return int(n) if n.is_integer() else None
        except (ValueError,TypeError): return None
    yy,mm,dd=integer(y),integer(m),integer(day)
    if yy is None: return ('','','missing_or_tagged')
    if yy<1900 or yy>2026: return ('','','invalid_year_or_special')
    if mm is None or not 1<=mm<=12:
        return (f'{yy}-01-01',f'{yy}-12-31','year')
    last=calendar.monthrange(yy,mm)[1]
    if dd is None or not 1<=dd<=last:
        return (f'{yy}-{mm:02d}-01',f'{yy}-{mm:02d}-{last:02d}','month' if dd in [None,88,99,-7,-8,-9] else 'month_with_invalid_day')
    date=f'{yy}-{mm:02d}-{dd:02d}'
    return (date,date,'day')

def adddates(z,prefix,ys,ms,ds):
    a=[bounds(y,m,d) for y,m,d in zip(ys,ms,ds)]
    z[prefix+'_lower']=[x[0] for x in a];z[prefix+'_upper']=[x[1] for x in a];z[prefix+'_precision']=[x[2] for x in a]

def join(z,d,prefix,source):
    q=d.rename(columns={c:prefix+c for c in d if c!='person_id'})
    z=z.merge(q,on='person_id',how='left',validate='one_to_one',indicator='_match')
    LOG.append(dict(cohort='CHARLS',wave=int(z.wave.iloc[0]),source=source,operation='left_join',input_rows=len(z),source_rows=len(d),output_rows=len(z),matched=int(z._match.eq('both').sum()),unmatched=int(z._match.eq('left_only').sum()),source_not_in_universe=int((~d.person_id.isin(z.person_id)).sum()),dropped=0))
    z[prefix+'source_present']=z._match.eq('both')
    return z.drop(columns='_match')

def charls():
    h,hp=readchar('harmonized','H_CHARLS_D_Data'); eol,ep=readchar('harmonized','H_CHARLS_EOL_a')
    data={}; universe=set(h.person_id)|set(eol.person_id)
    for w in [2011,2013,2015,2018,2020]:
        stems=['health_status_and_functioning','demographic_background','weight' if w==2011 else 'Weights']
        if w>=2015:stems+=['Sample_Infor']
        if w==2013:stems+=['Exit_Interview']
        if w==2020:stems+=['Exit_Module']
        data[w]={}
        for stem in stems:
            d,sp=readchar(w,stem);data[w][stem.lower()]=(d,sp);universe.update(d.person_id)
    bset=set(data[2011]['health_status_and_functioning'][0].person_id)
    rows=[];death=[]
    # Complete source union provides explicit missing/noninterview rows; no study sample is selected.
    for k,w in enumerate([2011,2013,2015,2018,2020],1):
        z=pd.DataFrame({'person_id':sorted(universe)});z['wave']=w;z['in_2011_health_source']=z.person_id.isin(bset)
        health,sp=data[w]['health_status_and_functioning']; dic=DIC[DIC.source_file.eq(sp)]
        fv=dic[dic.candidate_domains.str.contains('FI_candidate',regex=False)].source_variable.tolist()
        raw=health[['person_id']+fv].copy();raw.insert(1,'wave',w);save(raw,f'charls_{w}_FI_candidate_raw.csv')
        pa=[f'da032_{i}_' if w==2020 else f'da051_{i}_' for i in [1,2,3]]
        sn,nap=('da030','da031') if w==2020 else ('da049','da050')
        cols=['person_id','ID','communityID','householdID',sn,nap]+[c for c in health if re.fullmatch(r'da0(?:3[2-6]|5[1-5])_.*',c)]
        cols=list(dict.fromkeys(c for c in cols if c in health))
        z=join(z,health[cols],'health__',sp)
        for stem,(d,p) in data[w].items():
            if stem=='health_status_and_functioning':continue
            prefix='demo__' if stem=='demographic_background' else ('weight__' if 'weight' in stem else ('sample__' if stem=='sample_infor' else 'exit__'))
            use=d if prefix!='exit__' else d[[c for c in ['person_id','ID','exb001_1','exb001_2','exb001_3'] if c in d]]
            z=join(z,use,prefix,p)
        hc=['person_id','ragender','rabyear','raeduc_c']+[c for c in h if re.fullmatch(fr'r{k}(?:iwstat|iwy|iwm|agey|mstat|smoken|drinkl|mbmi|wtrespb|wtrespl)',c)] if k<=4 else ['person_id','ragender','rabyear','raeduc_c']
        z=join(z,h[[c for c in hc if c in h]],'harm__',hp)
        z['health_observed']=z.health__source_present
        z['sleep_raw']=z['health__'+sn];z['nap_minutes_raw']=z['health__'+nap]
        s=nums(z.sleep_raw);z['sleep_hours_valid_physical']=s.where(s.between(0,24))
        z['sleep_value_class']=np.select([z.sleep_raw.isna()|z.sleep_raw.eq(''),s.isna(),s.lt(0),s.gt(24),s.eq(0)],['system_or_no_health','tagged_or_nonnumeric','negative_special_requires_codebook','outside_physical_range','zero_flag'],default='in_physical_range')
        a=z[['health__'+c for c in pa]].apply(nums)
        z['activity_binary']=np.where(a.eq(1).any(axis=1),1,np.where(a.eq(2).all(axis=1),0,np.nan))
        z['activity_response_class']=np.where(z.activity_binary.notna(),'classified','unknown_not_no')
        z['activity_measure']='any_10min_CHARLS_NOT_CLHLS_regular_exercise'
        z['activity_design_flag']='NOT_IDENTIFIED_no_ordinary_imputation'
        dateprefix='weight__' if w in [2011,2013] else 'sample__'
        adddates(z,'sample_record_date',z[dateprefix+'iyear'],z[dateprefix+'imonth'],['']*len(z))
        if k<=4:adddates(z,'harm_interview_date',z[f'harm__r{k}iwy'],z[f'harm__r{k}iwm'],['']*len(z))
        flag='weight__INDV_L_Died' if w==2013 else 'sample__died' if w>=2015 else None
        z['death_flag_raw']=z[flag] if flag else ''
        z['harm_status_raw']=z[f'harm__r{k}iwstat'] if k<=4 else ''
        z['death_evidence_this_wave']=nums(z.death_flag_raw).eq(1)|nums(z.harm_status_raw).isin([5,6])
        z['source_status']=np.select([z.health_observed&z.death_evidence_this_wave,z.health_observed,z.death_evidence_this_wave,nums(z.harm_status_raw).eq(4)],['CONFLICT_health_and_death','health_interview','death_report','nonresponse_confirmed_alive'],default='not_observed_or_unknown')
        if w in [2013,2020]:
            q=z[z.exit__source_present].copy(); adddates(q,'death_date',q.exit__exb001_1,q.exit__exb001_2,q.exit__exb001_3 if w==2020 else ['']*len(q))
            death.append(q[['person_id','wave','death_date_lower','death_date_upper','death_date_precision']].assign(death_source='exit_module'))
        save(z,f'charls_{w}_source_linked.csv')
        core=['person_id','wave','in_2011_health_source','health_observed','sleep_raw','nap_minutes_raw','sleep_hours_valid_physical','sleep_value_class','activity_binary','activity_response_class','activity_measure','activity_design_flag','sample_record_date_lower','sample_record_date_upper','sample_record_date_precision','death_flag_raw','harm_status_raw','death_evidence_this_wave','source_status','harm__ragender','harm__rabyear','harm__raeduc_c']
        core += [c for c in z if c.startswith('harm_interview_date_') or re.fullmatch(r'harm__r\d(?:agey|mstat|smoken|drinkl|mbmi|wtrespb|wtrespl)',c)]
        rows.append(z[core]); print('BUILT CHARLS',w,len(z),flush=True)
        MAP.extend(dict(cohort='CHARLS',wave=w,source_file=sp,source_variable=v,role='raw_FI_candidate_NOT_SCORED') for v in fv)
    for d,source,yy,mm in [(h,'harmonized_regular','radyear','radmonth'),(eol,'harmonized_EOL','raxyear','raxmonth')]:
        if yy not in d: raise KeyError(yy)
        q=d[d[yy].ne('')].copy();q['wave']='source_specific';adddates(q,'death_date',q[yy],q[mm],['']*len(q))
        death.append(q[['person_id','wave','death_date_lower','death_date_upper','death_date_precision']].assign(death_source=source))
    long=pd.concat(rows,ignore_index=True);assert not long.duplicated(['person_id','wave']).any()
    save(long,'charls_person_wave_source_long.csv');save(pd.concat(death,ignore_index=True),'charls_death_date_source_candidates.csv')
    return long

def clhls():
    inv=pd.read_csv(BASE/'clhls_sources/clhls_export_inventory.csv',dtype=str); panels={}
    for r in inv.to_dict('records'):
        b=int(r['file'][6:10]);sp=str(ROOT/'06_audit/clhls_verified_extract'/r['file'])
        d=pd.read_csv(BASE/'clhls_sources'/f"{Path(r['file']).stem}_raw.csv",dtype=str,keep_default_na=False)
        assert not d.id.duplicated().any();parts=[]
        waves=[w for w in [2008,2011,2014,2018] if w>=b]
        for w in waves:
            suf='' if w==b else '_'+str(w)[2:];st=None if w==b else {2011:'dth08_11',2014:'dth11_14',2018:'dth14_18'}[w]
            sleep='b310b_18' if w==2018 else 'g02'+suf
            fields={'year_raw':'yearin'+suf,'month_raw':'monthin'+suf,'day_raw':'dayin'+suf,'age_raw':'vage_11' if b==2008 and w==2011 else 'trueage'+suf,'activity_raw':'d91'+suf,'sleep_raw':sleep,'baseline_sex_raw':'a1','residence_raw':'resid_11' if b==2008 and w==2011 else 'residenc'+suf,'baseline_education_raw':'f1','marital_raw':'f41'+suf}
            z=pd.DataFrame({'person_id':d.id,'baseline_frame':b,'wave':w})
            for out,v in fields.items():
                if v not in d:raise KeyError((b,w,v))
                z[out]=d[v];MAP.append(dict(cohort='CLHLS',baseline_frame=b,wave=w,source_file=sp,source_variable=v,role=out))
            z['status_raw']=d[st] if st else 'BASELINE'
            z['baseline_weight_raw']=d['w_'+str(b)]
            adddates(z,'interview_date',z.year_raw,z.month_raw,z.day_raw)
            z['interview_date_recorded']=z.interview_date_lower.ne('')
            status=nums(z.status_raw)
            z['death_evidence_this_wave']=status.isin([1,2]);z['alive_report_this_wave']=status.eq(0) if st else True
            z['source_status']=np.select([z.death_evidence_this_wave&z.interview_date_recorded&status.ne(2),z.death_evidence_this_wave,status.eq(-9),status.eq(-8),z.alive_report_this_wave],['CONFLICT_interview_and_death','death_report','lost_to_followup','previous_dead_or_lost','alive_report'],default='unknown_status')
            z['alive_then_died_between_2011_2012']=status.eq(2) if w==2011 else False
            if st:
                ds='d'+str(w)[2:]+'v'
                adddates(z,'death_date',d[ds+'year'],d[ds+'month'],d[ds+'day'])
            else:
                for f in ['lower','upper','precision']:z['death_date_'+f]='not_applicable_baseline' if f=='precision' else ''
            s=nums(z.sleep_raw);z['sleep_hours_valid_physical']=s.where(s.between(0,24))
            z['sleep_value_class']=np.select([z.sleep_raw.eq(''),s.isna(),s.isin([-7,-8,-9]),s.eq(88),s.eq(99),~s.between(0,24),s.eq(0)],['system_missing','nonnumeric','structural_negative','dont_know','coded_missing','outside_physical_range','zero_flag'],default='in_physical_range')
            a=nums(z.activity_raw);z['activity_binary']=a.map({1:1,2:0});z['activity_measure']='CLHLS_current_regular_exercise_NOT_CHARLS_any_10min'
            dic=DIC[DIC.source_file.eq(sp)]
            fv=[v for v in dic[dic.candidate_domains.str.contains('FI_candidate',regex=False)].source_variable if (v.endswith(suf) if suf else not re.search(r'_(?:11|14|18)$',v))]
            if w==b:fv=[v for v in fv if not re.search(r'^d(?:11|14|18)',v)]
            raw=d[['id']+fv].rename(columns={'id':'person_id'});raw.insert(1,'wave',w);raw.insert(1,'baseline_frame',b)
            save(raw,f'clhls_frame{b}_wave{w}_FI_candidate_raw.csv')
            MAP.extend(dict(cohort='CLHLS',baseline_frame=b,wave=w,source_file=sp,source_variable=v,role='raw_FI_candidate_NOT_SCORED') for v in fv)
            parts.append(z)
        long=pd.concat(parts,ignore_index=True);assert not long.duplicated(['person_id','wave']).any()
        save(long,f'clhls_frame{b}_person_wave_source_long.csv');panels[b]=long
        print('BUILT CLHLS',b,len(long),flush=True)
    overlap=[]
    for a,b in [(2008,2011),(2008,2014),(2011,2014)]:
        aa=set(panels[a].person_id);bb=set(panels[b].person_id)
        overlap.append(dict(frame_a=a,frame_b=b,unique_a=len(aa),unique_b=len(bb),overlap_ids=len(aa&bb),interpretation='OVERLAPPING_FRAMES_NOT_INDEPENDENT_REPLICATIONS'))
    save(pd.DataFrame(overlap),'clhls_baseline_frame_overlap.csv')
    return panels

def sequences(d,name):
    counts=[];issues=[]
    for w,g in d.groupby('wave'):
        counts.append(dict(dataset=name,wave=w,rows=len(g),unique_id=g.person_id.nunique(),duplicate=int(g.person_id.duplicated().sum()),death_reports=int(g.death_evidence_this_wave.sum()),activity_classified=int(g.activity_binary.notna().sum()),sleep_physical=int(g.sleep_hours_valid_physical.notna().sum()),sleep_zero=int(g.sleep_value_class.eq('zero_flag').sum()),status_counts=json.dumps(g.source_status.value_counts().to_dict())))
    for pid,g in d.groupby('person_id',sort=False):
        g=g.sort_values('wave');dead=False;lost=False
        for row in g.itertuples():
            observed=row.health_observed if name=='CHARLS' else row.alive_report_this_wave or row.interview_date_recorded
            if dead and observed:issues.append(dict(dataset=name,person_id=pid,wave=row.wave,issue='observed_after_prior_death_report'))
            if lost and observed:issues.append(dict(dataset=name,person_id=pid,wave=row.wave,issue='return_after_nonobservation_or_loss'))
            if row.source_status.startswith('CONFLICT'):issues.append(dict(dataset=name,person_id=pid,wave=row.wave,issue=row.source_status))
            if row.death_evidence_this_wave:dead=True
            if row.source_status in ['not_observed_or_unknown','lost_to_followup','nonresponse_confirmed_alive']:lost=True
            if observed:lost=False
    return counts,issues

def main():
    OUT.mkdir(exist_ok=True)
    a=charls();b=clhls();counts=[];issues=[]
    for name,d in [('CHARLS',a)]+[(f'CLHLS_{k}',v) for k,v in b.items()]:
        c,i=sequences(d,'CHARLS' if name=='CHARLS' else name);counts+=c;issues+=i
    save(pd.DataFrame(counts),'wave_source_summary.csv');save(pd.DataFrame(issues,columns=['dataset','person_id','wave','issue']),'source_state_sequence_issues.csv')
    save(pd.DataFrame(LOG),'join_lineage.csv');save(pd.DataFrame(MAP),'explicit_wave_field_mapping.csv')
    summary={'status':'SOURCE_LONGITUDINAL_BUILT_NOT_ANALYSIS_READY','raw_modified':False,'sample_exclusions':False,'FI_scoring':'NOT RUN','formal_models':'NOT RUN','issues_are_unadjudicated':True,'outputs':PRODUCTS,'next_task':'FI item recoding adjudication and source-status/date conflict resolution within Stage A'}
    (OUT/'source_longitudinal_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary['status'])

if __name__=='__main__':main()
