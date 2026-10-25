"""Independent vectorized interval/eligibility checks; no optimization or writes to frozen inputs."""
from pathlib import Path
import json
import hashlib
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parent
READY=OUT/'corrected_stageA'
checks=[]

def check(frame,name,ok,detail=''):
    checks.append(dict(frame=frame,check=name,status='PASS' if ok else 'FAIL',detail=str(detail)))
    assert ok,(frame,name,detail)

def main():
    manifest=json.loads((READY/'corrected_stageA_manifest.json').read_text(encoding='utf-8'))
    for f in manifest['inputs']+manifest['outputs']:
        p=Path(f['path'])
        with p.open('rb') as h: actual=hashlib.file_digest(h,'sha256').hexdigest()
        assert actual==f['sha256'] and p.stat().st_size==f['bytes'],p
    tr=pd.read_csv(READY/'transition_intervals_audited.csv',dtype={'person_id':str})
    for frame in [2008,2011,2014]:
        d=pd.read_csv(READY/f'clhls_frame{frame}_analysis_ready_long.csv',dtype={'person_id':str}).sort_values(['person_id','wave'])
        n=d.groupby('person_id',sort=False).shift(-1)
        mask=n.wave.notna()&d.baseline_eligible&d.state.ne('unknown')&n.state.ne('unknown')
        expected=pd.DataFrame(dict(person_id=d.loc[mask,'person_id'],wave_start=d.loc[mask,'wave'].astype(int),wave_end=n.loc[mask,'wave'].astype(int),state_start=d.loc[mask,'state'],state_end=n.loc[mask,'state'])).reset_index(drop=True)
        actual=tr[tr.baseline_frame.eq(frame)]
        join=expected.merge(actual,on=['person_id','wave_start','wave_end','state_start','state_end'],how='outer',indicator=True,validate='one_to_one')
        check(frame,'vectorized_all_adjacent_interval_keys_and_states',join._merge.eq('both').all(),len(expected))
        check(frame,'raw_person_wave_keys_unique',not d.duplicated(['person_id','wave']).any(),len(d))
        check(frame,'FI_bounds',d.fi_score.dropna().between(0,1).all())
        check(frame,'FI_min36',d.fi_score.notna().eq(d.fi_n_observed.ge(36)).all())
        check(frame,'death_priority',d.loc[d.death_evidence_this_wave,'state'].eq('death').all())
        check(frame,'death_absorbing_evidence',actual.death_absorbing_ok.all())
        m=OUT/'models'/f'frame{frame}'
        allrows=pd.read_csv(m/f'clhls_frame{frame}_model_input_all_intervals_v2.csv',dtype={'person_id':str})
        fit=pd.read_csv(m/'clhls_model_ready_v2.csv',dtype={'person_id':str})
        cov=['female','urban','education_primary','education_middle','education_high','partnered','smoking','drinking','bmi','disease_count','memory','cesd9']
        eligible=allrows.i.isin([0,1,2])&allrows.j.isin([0,1,2,3])&allrows.interval_years.gt(0)&allrows.age_enter.notna()&allrows.sleep.between(0,24)&allrows.activity.isin([0,1])&allrows.baseline_age.notna()&allrows.subject_weight_raw.gt(0)&allrows[cov].notna().all(axis=1)
        check(frame,'independent_complete_case_mask',eligible.eq(allrows.model_eligible).all())
        check(frame,'independent_complete_case_count',eligible.sum()==len(fit),len(fit))
        key=['person_id','wave_start','wave_end']
        merged=allrows.loc[eligible,key].merge(fit[key],on=key,how='outer',indicator=True,validate='one_to_one')
        check(frame,'complete_case_exact_keys',merged._merge.eq('both').all())
        people=fit.drop_duplicates('person_id')
        check(frame,'unique_person_weight_normalization',abs(people.weight.mean()-1)<1e-12)
        for c in ['baseline_age','bmi','disease_count','memory','cesd9']:
            z=(fit[c]-people[c].mean())/people[c].std(ddof=0)
            check(frame,'unique_person_scaling_'+c,np.allclose(z,fit[c+'_c'],rtol=0,atol=1e-12))
    p=OUT/'G2_independent_delivery_validation.csv'
    if p.exists():raise FileExistsError(p)
    pd.DataFrame(checks).to_csv(p,index=False,encoding='utf-8-sig')
    print('G2_INDEPENDENT_DELIVERY_VALIDATION_PASS',len(checks))

if __name__=='__main__':main()
