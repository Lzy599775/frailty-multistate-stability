"""Deterministic Stage-A final FI/state recoding and analysis-ready manifest.

All raw and source-linked files are read-only inputs. This script records the
exact item maps and creates versioned analysis-ready tables; it does not fit a
formal model or impute missing values.
"""
from pathlib import Path
import hashlib, json, re
import numpy as np
import pandas as pd

ROOT=Path(r'F:\A_CHARLS+CLHLS\00_新选题')
BASE=ROOT/'08_intermediate/analysis_ready_stageA_v1_20261010'
L=BASE/'longitudinal_sources'; OUT=BASE/'analysis_ready_final_v1'; OUT.mkdir(exist_ok=True)
MAN={'version':'analysis_ready_final_v1_20261011','rules':{},'outputs':[]}

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(d,name):
 p=OUT/name; d.to_csv(p,index=False,encoding='utf-8-sig'); MAN['outputs'].append({'path':str(p),'rows':len(d),'cols':d.shape[1],'sha256':sha(p)}); return p
def num(s): return pd.to_numeric(s,errors='coerce')
def missable(s,lo,hi):
 x=num(s); return x.where(x.between(lo,hi))
def map_ord(s,lo,hi):
 x=missable(s,lo,hi); return (x-lo)/(hi-lo)
def map_yesno(s):
 x=num(s); return x.map({1:1,2:0})

def charls_items(w, raw):
 """Common 26-item CHARLS FI. 2020 disease order follows the official 1..14 disease block."""
 out={}; itemmap=[]
 health='da002' if 'da002' in raw else 'da001'
 if w==2020: health='da001'
 out['self_health']=map_ord(raw[health],1,5);itemmap.append(('self_health',health,'ordinal 1 best to 5 worst'))
 adl_old=['db010','db011','db012','db013','db014','db015'];adl_new=['db001','db003','db005','db007','db009','db011']
 iadl_old=['db016','db017','db018','db019','db020'];iadl_new=['db012','db014','db016','db020','db022']
 for i,c in enumerate(adl_new if w==2020 else adl_old,1):out[f'adl_{i}']=map_ord(raw[c],1,4);itemmap.append((f'adl_{i}',c,'ordinal 1 independent to 4 unable'))
 for i,c in enumerate(iadl_new if w==2020 else iadl_old,1):out[f'iadl_{i}']=map_ord(raw[c],1,4);itemmap.append((f'iadl_{i}',c,'ordinal 1 independent to 4 unable'))
 disease=[f'da003_{i}_' for i in range(1,15)] if w==2020 else [f'da007_{i}_' for i in range(1,15)]
 for i,c in enumerate(disease,1):out[f'disease_{i}']=map_yesno(raw[c]);itemmap.append((f'disease_{i}',c,'1 yes/deficit; 2 no; 2020 preserves official disease order'))
 return out,itemmap

def final_charls():
 rows=[]; maps=[]
 for w in [2011,2013,2015,2018,2020]:
  raw=pd.read_csv(L/f'charls_{w}_FI_candidate_raw.csv',dtype=str,keep_default_na=False)
  src=pd.read_csv(L/f'charls_{w}_source_linked.csv',dtype=str,keep_default_na=False)
  z=src.merge(raw,on=['person_id','wave'],how='left',validate='one_to_one',indicator=True); z=z.drop(columns='_merge')
  if w==2020:
   disease=[f'da003_{i}_' for i in range(1,15)]
   extra=pd.read_csv(BASE/'charls_sources/2020_Health_Status_and_Functioning_raw.csv',usecols=['ID']+disease,dtype=str,keep_default_na=False)
   extra['person_id']=extra.ID;extra=extra.drop(columns='ID');z=z.drop(columns=[c for c in disease if c in z])
   z=z.merge(extra,on='person_id',how='left',validate='one_to_one')
  rec,im=charls_items(w,z); maps += [dict(cohort='CHARLS',wave=w,item=i,source_variable=v,rule=r) for i,v,r in im]
  for k,v in rec.items():z[k]=v
  items=list(rec); nobs=z[items].notna().sum(axis=1);z['fi_n_items']=len(items);z['fi_n_observed']=nobs;z['fi_completeness']=nobs/len(items);z['fi_score']=z[items].mean(axis=1).where(z.fi_completeness>=.8);z['fi_state']=pd.cut(z.fi_score,[-np.inf,.10,.21,np.inf],labels=['robust','prefrail','frail'],right=True,include_lowest=True).astype('string')
  agecol={1:'harm__r1agey',2:'harm__r2agey',3:'harm__r3agey',4:'harm__r4agey'}.get([2011,2013,2015,2018,2020].index(w)+1,'demo__ba003_1')
  if w==2020: agecol='demo__ba003_1'
  z['age_raw']=z[agecol];z['age_years']=num(z.age_raw)
  z['baseline_eligible']=z.in_2011_health_source.eq('True') & z['age_years'].where(z.wave.eq(2011),np.nan).ge(45)
  if w!=2011:
   base=pd.read_csv(L/'charls_2011_source_linked.csv',dtype=str,keep_default_na=False)[['person_id','harm__r1agey']].rename(columns={'harm__r1agey':'baseline_age_raw'})
   z=z.merge(base,on='person_id',how='left',validate='one_to_one');z['baseline_age']=num(z.baseline_age_raw)
  else:z['baseline_age']=z.age_years
  z['baseline_eligible']=z.baseline_age.ge(45)&z.in_2011_health_source.eq('True')
  z['state']=np.where(z.death_evidence_this_wave.eq('True'),'death',z.fi_state.fillna('unknown'))
  z['death_absorbing_candidate']=z.state.eq('death')
  z['exposure_eligible']=z.health_observed.eq('True') & z.sleep_hours_valid_physical.ne('') & z.activity_binary.notna()
  z['weight_raw']=np.nan
  candidates={2011:'weight__ind_weight_ad2',2013:'weight__INDV_L_weight',2015:'weight__INDV_weight_ad2',2018:'weight__INDV_weight',2020:'weight__INDV_weight'}
  if candidates[w] in z:z['weight_raw']=num(z[candidates[w]])
  else:z['weight_raw']=np.nan
  z['weight_positive']=z.weight_raw.gt(0)
  keep=['person_id','wave','baseline_eligible','baseline_age','age_years','health_observed','sleep_hours_valid_physical','activity_binary','exposure_eligible','weight_raw','weight_positive','fi_n_items','fi_n_observed','fi_completeness','fi_score','fi_state','state','death_evidence_this_wave','death_absorbing_candidate']+items
  rows.append(z[keep])
 save(pd.concat(rows,ignore_index=True),'charls_analysis_ready_long.csv')
 return pd.concat(rows,ignore_index=True),maps

def clhls_items(frame,w,raw):
 suf='' if w==frame else '_'+str(w)[2:]; out={};maps=[]
 def g(v): return v if v in raw else None
 def add(name,v,fun,rule):
  if g(v):out[name]=fun(raw[v]);maps.append((name,v,rule))
 add('self_health','b12'+suf,lambda x:map_ord(x,1,5),'1 very good to 5 very bad')
 for i,v in enumerate(['e1','e2','e3','e4','e5','e6'],1):add(f'adl_{i}',v+suf,lambda x:map_ord(x,1,3),'1 independent to 3 dependent')
 for i,v in enumerate(['e7','e8','e9','e10','e14'],1):add(f'iadl_{i}',v+suf,lambda x:map_ord(x,1,3),'1 able to 3 unable; conceptually non-exposure IADL')
 add('vision','g1'+suf,lambda x:map_ord(x,1,4),'1 best to 4 blind')
 add('hearing','h1'+suf,lambda x:map_ord(x,1,4),'1 best to 4 unable')
 for letter in 'abcdefghijklmnopqrstuv':
  add('disease_'+letter,'g15'+letter+'1'+suf,lambda x:map_yesno(x.replace({'3':'','9':''})), '1 yes; 2 no; 3/9 missing')
 for i,v in enumerate(['b21','b22','b23','b24','b25','b26','b27'],1):
  def mental(x,v=v):
   q=missable(x,1,5); return (5-q)/4 if v in ['b23','b24','b26'] else (q-1)/4
  add(f'psych_{i}',v+suf,mental,'1..5; negative items 23/24/26 and positive items 21/22/25/27 directionally recoded')
 add('serious_illness','g131'+suf,lambda x:num(x).map(lambda q: np.nan if q==99 else (0 if q==0 else 1)),'0 none; positive or 88 bedridden deficit; 99 missing')
 return out,maps

def final_clhls():
 allrows=[];maps=[]
 for p in sorted(L.glob('clhls_frame*_person_wave_source_long.csv')):
  frame=int(re.search(r'frame(\d+)',p.name).group(1)); src=pd.read_csv(p,dtype=str,keep_default_na=False)
  frame_rows=[]
  for w in sorted(src.wave.astype(int).unique()):
   raw=pd.read_csv(L/f'clhls_frame{frame}_wave{w}_FI_candidate_raw.csv',dtype=str,keep_default_na=False)
   z=src[src.wave.eq(str(w))].merge(raw,on=['person_id','baseline_frame','wave'],how='left',validate='one_to_one',indicator=True);assert z._merge.eq('both').all();z=z.drop(columns='_merge')
   rec,im=clhls_items(frame,w,z);maps += [dict(cohort='CLHLS',baseline_frame=frame,wave=w,item=i,source_variable=v,rule=r) for i,v,r in im]
   for k,v in rec.items():z[k]=v
   items=list(rec);nobs=z[items].notna().sum(axis=1);z['fi_n_items']=len(items);z['fi_n_observed']=nobs;z['fi_completeness']=nobs/len(items);z['fi_score']=z[items].mean(axis=1).where(z.fi_completeness>=.8);z['fi_state']=pd.cut(z.fi_score,[-np.inf,.10,.21,np.inf],labels=['robust','prefrail','frail'],right=True,include_lowest=True).astype('string')
   z['age_years']=num(z.age_raw);z['baseline_eligible']=z.baseline_frame.eq(str(frame))&z.age_years.ge(65);z['state']=np.where(z.death_evidence_this_wave.eq('True'),'death',z.fi_state.fillna('unknown'));z['death_absorbing_candidate']=z.state.eq('death');z['exposure_eligible']=z.sleep_hours_valid_physical.ne('') & z.activity_binary.notna();z['weight_raw']=num(z.baseline_weight_raw);z['weight_positive']=z.weight_raw.gt(0)
   keep=['person_id','baseline_frame','wave','baseline_eligible','age_years','interview_date_lower','interview_date_upper','source_status','sleep_hours_valid_physical','activity_binary','exposure_eligible','weight_raw','weight_positive','fi_n_items','fi_n_observed','fi_completeness','fi_score','fi_state','state','death_evidence_this_wave','death_absorbing_candidate']+items
   out=z[keep];frame_rows.append(out)
  frame_long=pd.concat(frame_rows,ignore_index=True)
  save(frame_long,f'clhls_frame{frame}_analysis_ready_long.csv');allrows.append(frame_long)
 return allrows,maps

def transitions(df,cohort,frame=''):
 d=df.copy();d['wave_num']=num(d.wave);out=[]
 for pid,g in d.groupby('person_id',sort=False):
  g=g.sort_values('wave_num');
  for a,b in zip(g.iloc[:-1].to_dict('records'),g.iloc[1:].to_dict('records')):
   if not bool(a.get('baseline_eligible',False)):continue
   sa,sb=a['state'],b['state'];
   if sa=='unknown' or sb=='unknown':continue
   out.append({'cohort':cohort,'baseline_frame':frame,'person_id':pid,'wave_start':a['wave'],'wave_end':b['wave'],'state_start':sa,'state_end':sb,'interval_years':float(b['wave_num']-a['wave_num']),'death_absorbing_ok':(sa=='death' and sb=='death') or sa!='death'})
 return pd.DataFrame(out)

def audit_and_manifest(charls,cmaps,clhls, lm):
 trans=[];trans.append(transitions(charls,'CHARLS','2011'))
 for x in clhls: trans.append(transitions(x,'CLHLS',str(x.baseline_frame.iloc[0])))
 tr=pd.concat(trans,ignore_index=True);save(tr,'transition_intervals_audited.csv')
 mat=tr.groupby(['cohort','baseline_frame','state_start','state_end']).size().reset_index(name='n_intervals');save(mat,'transition_event_counts.csv')
 pos=[]
 for name,d in [('CHARLS',charls)]+[(f'CLHLS_frame{x.baseline_frame.iloc[0]}',x) for x in clhls]:
  for w,g in d.groupby('wave'):
   wt=num(g.weight_raw);pos.append({'cohort':name,'wave':w,'n':len(g),'positive_weight_n':int(wt.gt(0).sum()),'weight_missing_n':int(wt.isna().sum()),'weight_min_positive':wt[wt.gt(0)].min(),'weight_max':wt.max(),'positivity_status':'PASS' if wt.gt(0).any() else 'FAIL'})
 save(pd.DataFrame(pos),'weight_positivity_audit.csv')
 MAN['rules']={'charls_baseline_age_min':45,'clhls_baseline_age_min':65,'fi_completeness_min':0.8,'fi_thresholds':{'robust':'<=0.10','prefrail':'>0.10 and <=0.21','frail':'>0.21'},'death_state':'death absorbing; source death evidence takes precedence over FI state','missingness':'no imputation; unknown remains missing','charls_fi_items':'self health + 6 ADL + 5 IADL + 14 disease blocks; 2020 disease order preserved from official block','clhls_fi_items':'self health + 6 ADL + 5 IADL + vision + hearing + disease + psychological + serious illness','exposure':'sleep physical range and queue-specific current activity; no 24h composition'}
 MAN['status']='ANALYSIS_READY_STAGE_A_COMPLETED';MAN['formal_models']='NOT RUN';MAN['software_fit']='NOT RUN';MAN['transition_intervals']=len(tr);MAN['transition_event_counts']=mat.to_dict('records');MAN['maps']=lm+cmaps
 (OUT/'analysis_ready_manifest_v1.json').write_text(json.dumps(MAN,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def main():
 c,cm=final_charls();l,lm=final_clhls();audit_and_manifest(c,cm,l,lm);print('ANALYSIS_READY_STAGE_A_COMPLETED')
if __name__=='__main__':main()
