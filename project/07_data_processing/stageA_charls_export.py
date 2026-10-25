from pathlib import Path
import pandas as pd, pyreadstat, json, hashlib, re
ROOT=Path(r"F:\A_CHARLS+CLHLS\00_新选题")
RAW=Path(r"E:\公开数据\Charls")
OUT=ROOT/"08_intermediate/analysis_ready_stageA_v1_20261010/charls_sources"
OUT.mkdir(parents=True,exist_ok=True)
files=[]
for wave in [2011,2013,2015,2018,2020]:
 for p in sorted((RAW/f"{wave} Spss").glob("Raw data_dta version*/*.dta")):
  if p.stem.lower() in ['health_status_and_functioning','demographic_background','sample_infor','weight','weights','exit_interview','exit_module','psu','cognition']:
   files.append((str(wave),p))
for module,name in [('1. Regular wave','H_CHARLS_D_Data'),('3. End of life','H_CHARLS_EOL_a')]:
 files.append(('harmonized',RAW/'Harmonized Charls Spss'/module/(name+'.dta')))
inv=[]; dictionaries=[]; profiles=[]
for wave,p in files:
 print('READ',wave,p.name,flush=True)
 d,m=pyreadstat.read_dta(str(p),apply_value_formats=False,user_missing=True)
 h=hashlib.sha256(p.read_bytes()).hexdigest()
 name=f'{wave}_{p.stem}_raw.csv'
 op=OUT/name
 if op.exists():raise FileExistsError(op)
 d.to_csv(op,index=False,encoding='utf-8-sig')
 for c in m.column_names:
  vl=m.variable_value_labels.get(c,{})
  dictionaries.append({'cohort':'CHARLS','wave':wave,'source_file':str(p),'source_sha256':h,'export_file':str(op),'source_variable':c,'label':m.column_names_to_labels.get(c,''),'value_labels_json':json.dumps(vl,ensure_ascii=False),'user_missing_json':json.dumps(m.missing_user_values.get(c,[]),ensure_ascii=False)})
  x=d[c];n=pd.to_numeric(x,errors='coerce')
  profiles.append({'wave':wave,'source_file':str(p),'variable':c,'n_rows':len(d),'n_missing_system':int(x.isna().sum()),'n_unique':int(x.nunique()),'n_negative_numeric':int(n.lt(0).sum()),'numeric_min':n.min(),'numeric_max':n.max(),'counts_json':json.dumps({str(k):int(v)for k,v in x.value_counts(dropna=False).items()},ensure_ascii=False) if x.nunique()<150 else ''})
 inv.append({'wave':wave,'source_file':str(p),'sha256':h,'rows':len(d),'cols':len(d.columns),'export_file':str(op),'export_sha256':hashlib.sha256(op.read_bytes()).hexdigest(),'id_column':'ID' if 'ID' in d else '', 'id_missing':int(d.ID.isna().sum()) if 'ID' in d else None,'id_duplicate_rows':int(d.ID.duplicated(keep=False).sum()) if 'ID' in d else None})
 print('DONE',len(d),len(d.columns),flush=True)
pd.DataFrame(dictionaries).to_csv(OUT/'charls_dictionary_complete.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(profiles).to_csv(OUT/'charls_profiles.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(inv).to_csv(OUT/'charls_export_inventory.csv',index=False,encoding='utf-8-sig')
print('EXPORT_COMPLETED',len(files),'source files; NO FI OR MODELS',flush=True)
