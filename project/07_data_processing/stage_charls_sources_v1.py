"""Source staging only. No FI scoring, sample exclusions, imputation or model fitting."""
from pathlib import Path
import hashlib, json, re
import numpy as np
import pandas as pd
import pyreadstat

ROOT = Path(r"F:\A_CHARLS+CLHLS\00_新选题")
RAW = Path(r"E:\公开数据\Charls")
OUT = ROOT / "08_intermediate" / "source_stage_v1_20261010"
OUT.mkdir(parents=True, exist_ok=True)
WAVES = [(2011,20),(2013,19),(2015,18),(2018,14),(2020,10)]

def sha(p):
    with p.open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()

def save(df, name):
    p=OUT/name
    if p.exists(): raise FileExistsError(p)
    df.to_csv(p,index=False,encoding="utf-8-sig")
    return {"file":str(p),"rows":len(df),"cols":df.shape[1],"sha256":sha(p)}

def ids(df, wave):
    if "ID" not in df: raise KeyError("ID")
    x=df["ID"].astype("string")
    if x.isna().any() or x.str.strip().eq("").any(): raise ValueError("Missing health ID")
    if not x.str.fullmatch(r"\d{11}" if wave==2011 else r"\d{12}").all():
        raise ValueError("Unexpected health ID format")
    norm=x.str[:-2]+"0"+x.str[-2:] if wave==2011 else x
    if norm.duplicated().any(): raise ValueError("Duplicate health ID")
    return norm

records=[]; lineage=[]; outputs=[]; dictionaries=[]; baseline_ids=None
for wave,n in WAVES:
    name="health_status_and_functioning" if wave==2011 else "Health_Status_and_Functioning"
    p=RAW/f"{wave} Spss/Raw data_dta version(N={n})/{name}.dta"
    d,m=pyreadstat.read_dta(str(p),apply_value_formats=False)
    d.insert(0,"ID_norm",ids(d,wave))
    if wave==2011: baseline_ids=set(d.ID_norm)
    sleep="da030" if wave==2020 else "da049"
    nap="da031" if wave==2020 else "da050"
    pa=[f"da032_{k}_" if wave==2020 else f"da051_{k}_" for k in [1,2,3]]
    for c in [sleep,nap]+pa:
        if c not in d: raise KeyError(f"{wave} {c}")
    # Retain raw values and fields; one row per health interview, no sample filtering.
    outputs.append(save(d,f"charls_health_{wave}_source.csv"))
    a=d[pa]
    yes=a.eq(1).any(axis=1)
    no=a.eq(2).all(axis=1)
    activity=pd.Series(np.nan,index=d.index)
    activity.loc[yes]=1; activity.loc[no]=0
    s=d[sleep]
    z=pd.DataFrame({"ID_norm":d.ID_norm,"wave":wave,"sleep_source":sleep,
       "sleep_hours_raw":s,"nap_minutes_raw":d[nap],
       "activity_source":";".join(pa),"activity_any10min":activity,
       "activity_all_items_missing":a.isna().all(axis=1),
       "activity_unclassified":activity.isna(),"is_2011_baseline":d.ID_norm.isin(baseline_ids),
       "sleep_outside_physical_range":s.notna() & ~s.between(0,24),
       "sleep_zero_flag":s.eq(0)})
    for c in pa: z[c]=d[c]
    records.append(z)
    lineage.append({"wave":wave,"source":str(p),"source_sha256":sha(p),"input_rows":len(d),
        "output_rows":len(z),"unique_id":int(z.ID_norm.nunique()),"duplicate_id":int(z.ID_norm.duplicated().sum()),
        "baseline_matched":int(z.is_2011_baseline.sum()),"baseline_unmatched":len(baseline_ids-set(z.ID_norm)),
        "nonbaseline_rows":int((~z.is_2011_baseline).sum()),
        "activity_yes":int(yes.sum()),"activity_no_all3":int(no.sum()),
        "activity_unclassified":int(activity.isna().sum()),"sleep_raw_missing":int(s.isna().sum()),
        "sleep_outside_physical_range":int(z.sleep_outside_physical_range.sum()),"dropped_rows":0})
    dictionaries.extend({"wave":wave,"source_variable":c,"label":m.column_names_to_labels.get(c),
       "value_labels_json":json.dumps(m.variable_value_labels.get(c,{}),ensure_ascii=False)}
       for c in m.column_names)
    print(f"STAGED CHARLS {wave}: rows={len(d)} classified_activity={int(activity.notna().sum())}")

long=pd.concat(records,ignore_index=True)
if long.duplicated(["ID_norm","wave"]).any(): raise ValueError("Duplicate person-wave")
outputs.append(save(long,"charls_exposure_source_long.csv"))
outputs.append(save(pd.DataFrame(dictionaries),"charls_source_dictionary.csv"))

# Harmonized data used only as official identity, age and status source. No legacy results reused.
hp=RAW/"Harmonized Charls Spss/1. Regular wave/H_CHARLS_D_Data.dta"
_,hm=pyreadstat.read_dta(str(hp),metadataonly=True)
cols=[c for c in hm.column_names if c in ["ID","communityID","ragender","raeduc_c"] or
      re.fullmatch(r"r[1-4](iwstat|iwy|iwm|agey|mstat|smoken|drinkl|mbmi|wtrespb|wtrespl)",c)]
h,_=pyreadstat.read_dta(str(hp),usecols=cols)
if h.ID.isna().any() or h.ID.duplicated().any(): raise ValueError("Invalid harmonized IDs")
outputs.append(save(h,"charls_harmonized_status_covariates_source.csv"))
b=pd.DataFrame({"ID_norm":sorted(baseline_ids)})
match=b.merge(h.rename(columns={"ID":"ID_norm"}),on="ID_norm",how="left",validate="one_to_one",indicator=True)
lineage.append({"source":str(hp),"source_sha256":sha(hp),"input_rows":len(h),"output_rows":len(match),
    "baseline_matched":int(match["_merge"].eq("both").sum()),
    "baseline_unmatched":int(match["_merge"].eq("left_only").sum()),"dropped_rows":0})
outputs.append(save(match,"charls_baseline_status_link_source.csv"))

# Export source death flags and dates, without deciding conflicting records.
for wave,n in WAVES[1:]:
    fn="Weights" if wave==2013 else "Sample_Infor"
    dp=RAW/f"{wave} Spss/Raw data_dta version(N={n})/{fn}.dta"
    _,dm=pyreadstat.read_dta(str(dp),metadataonly=True)
    dc=[c for c in dm.column_names if c in ["ID","died","INDV_L_Died","iyear","imonth","iday"]]
    expected="INDV_L_Died" if wave==2013 else "died"
    if expected not in dc: raise KeyError(f"{wave} {expected}")
    dx,_=pyreadstat.read_dta(str(dp),usecols=dc)
    outputs.append(save(dx,f"charls_vital_{wave}_source.csv"))
    valid=dx.ID.notna() & dx.ID.astype("string").str.fullmatch(r"\d{12}",na=False)
    lineage.append({"source":str(dp),"source_sha256":sha(dp),"input_rows":len(dx),"output_rows":len(dx),
         "invalid_or_blank_ids":int((~valid).sum()),"duplicate_valid_ids":int(dx.loc[valid,"ID"].duplicated().sum()),
         "death_flags_raw":int(dx[expected].eq(1).sum()),"dropped_rows":0})

summary=long.groupby("wave").agg(source_rows=("wave","size"),activity_classified=("activity_any10min","count"),
                              sleep_nonmissing_raw=("sleep_hours_raw","count"))
outputs.append(save(summary.reset_index(),"charls_exposure_availability.csv"))
result={"stage":"SOURCE_STAGING_COMPLETED","formal_models":"NOT RUN","FI_scoring":"NOT RUN",
        "sample_exclusions_applied":False,"raw_modified":False,"lineage":lineage,"outputs":outputs,
        "activity_definition":"CHARLS any >=10 minute vigorous/moderate/walking; NOT CLHLS regular exercise",
        "missing_activity_note":"Design nonadministration and refusal not inferred from raw NA"}
p=OUT/"charls_stage_summary.json"
if p.exists(): raise FileExistsError(p)
p.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print("SOURCE_STAGING_COMPLETED; no FI or model.")

