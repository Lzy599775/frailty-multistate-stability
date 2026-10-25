"""Authorized v7 FI repair: frozen raw candidates -> new Stage A + model inputs.

Only serious_illness scoring changes. Original source functions are AST loaded
without executing their top-level writes. No optimizer is called here.
"""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(r'F:\A_CHARLS+CLHLS\00_新选题')
BASE=ROOT/'08_intermediate/analysis_ready_stageA_v1_20261010'
OLD=BASE/'analysis_ready_final_v1'
OUT=Path(__file__).resolve().parent
G1=OUT.parent
PREV=G1/'continuation_20261024'
L=BASE/'longitudinal_sources'
READY=OUT/'corrected_stageA'
SEEN=set()

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def csv(p,**kw):
    p=Path(p);SEEN.add(p)
    return pd.read_csv(p,low_memory=False,**kw)

def save(d,name):
    p=READY/name
    if p.exists():
        if '--resume-verified-outputs' in sys.argv and p.read_bytes()==d.to_csv(index=False).encode('utf-8-sig'):
            print('REUSED_AFTER_EXACT_BYTE_VERIFICATION',name,flush=True);return
        raise FileExistsError(p)
    d.to_csv(p,index=False,encoding='utf-8-sig')

def write(p,j):
    if p.exists(): raise FileExistsError(p)
    p.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def serious(s):
    q=pd.to_numeric(s,errors='coerce')
    out=pd.Series(np.nan,index=s.index,dtype=float)
    out.loc[q.eq(0)]=0.
    out.loc[q.eq(88)]=1. # Explicit bedridden deficit, not 88 episodes.
    counts=np.isfinite(q)&q.gt(0)&q.mod(1).eq(0)&~q.isin([88,99])
    out.loc[counts]=1.
    return out

def model_module(frame):
    script=ROOT/'07_data_processing'/('stageB_10_clhls_penalized_execution_v8i.py' if frame==2008 else 'stageB_10_clhls_penalized_execution_v8g.py')
    SEEN.add(script);old=sys.argv
    try:
        sys.argv=[str(script),str(frame)]
        spec=importlib.util.spec_from_file_location('v7_model_'+str(frame),script)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    finally:sys.argv=old
    return m

def same(a,b,tol=1e-12):
    if pd.api.types.is_bool_dtype(a) or pd.api.types.is_bool_dtype(b):
        return a.astype(str).eq(b.astype(str)).to_numpy()
    if pd.api.types.is_numeric_dtype(a) or pd.api.types.is_numeric_dtype(b):
        aa=pd.to_numeric(a,errors='raise').to_numpy(float)
        bb=pd.to_numeric(b,errors='raise').to_numpy(float)
        return np.isclose(aa,bb,atol=tol,rtol=0,equal_nan=True)
    return a.replace('',pd.NA).fillna('<NA>').astype(str).eq(b.replace('',pd.NA).fillna('<NA>').astype(str)).to_numpy()

def adjacent(df,frame):
    """All adjacent candidate intervals, including unknown endpoints, for audit."""
    d=df.sort_values(['person_id','wave']).copy()
    nxt=d.groupby('person_id',sort=False).shift(-1)
    keep=nxt.wave.notna()
    return pd.DataFrame(dict(baseline_frame=frame,person_id=d.loc[keep,'person_id'],wave_start=d.loc[keep,'wave'].astype(int),wave_end=nxt.loc[keep,'wave'].astype(int),state_start=d.loc[keep,'state'],state_end=nxt.loc[keep,'state'],baseline_eligible=d.loc[keep,'baseline_eligible'])).reset_index(drop=True)

def main():
    assert not (READY/'corrected_stageA_manifest.json').exists(),'G2 already frozen; no rebuild permitted'
    READY.mkdir(exist_ok='--resume-verified-outputs' in sys.argv)
    # Verify the immediately preceding evidence seal and prior diagnostic scope.
    seal=json.loads((PREV/'continuation_output_checksums.json').read_text(encoding='utf-8'))
    for e in seal['files']:
        assert sha(PREV/e['path'])==e['sha256'],e['path']
    SEEN.update([PREV/'continuation_output_checksums.json',G1/'actual_sav_fi_metadata.csv',G1/'g131_code_frequency_by_frame_wave.csv',G1/'fi_recode_targeted_unit_tests.csv',G1/'sav_candidate_cell_reconciliation.csv',PREV/'official_questionnaire_codebook_excerpts_new.csv',PREV/'other_fi_unlabelled_code_review_new.csv'])
    decision='''# G1作者书面裁定及冻结规则

用户原文答复：“同意按上述官方代码簿和规则继续”。答复对应问题明确列出2008官方DOC缺G章节、同包官方代码簿定义，以及0→0、有效正整数次数→1、88按全年卧床独立记1、缺失及负数码→missing。

此为本轮正式语义授权，保留2008问卷正文缺章事实，不冒充已读取其G13。证据见上一轮continuation_20261024来源哈希、逐问项提取及图像。2011/2014/2018问项已核对；各SAV九组实际变量来源已逐格核对。

冻结：仅更正serious_illness缺失传播；88按既有卧床缺陷构念记1。有效正整数无人为次数上限，99为明确缺失，负数/系统缺失/不可解析/非整数均missing。未知意外正码若出现则阻止计算，不默认当缺陷。其余43项保留既定合法域和计分；G15原始0含义未说明，但不在有效1/2域内，保持missing，不推定无病。其来源标签未解决不等于计分未裁定。

不改变44项、36项最低有效数、有效项分母、0.10/0.21阈值、生存证据优先、原暴露/协变量/年龄/权重资格或模型参数。IPCW采用E1，不沿用旧CLHLS区间及点估计。

G1=PASS / RESOLVED_WITH_OFFICIAL_EVIDENCE_AND_AUTHOR_DECISION。G1通过不等于新版数据或模型已通过。
'''
    dp=OUT/'semantic_codebook_decision.md'
    if dp.exists():
        assert '--resume-verified-outputs' in sys.argv and dp.read_text(encoding='utf-8')==decision
    else: dp.write_text(decision,encoding='utf-8')
    source=ROOT/'07_data_processing/stageA_06_final_recode.py';SEEN.add(source)
    tree=ast.parse(source.read_text(encoding='utf-8-sig'))
    names={'num','missable','map_ord','map_yesno','clhls_items','final_clhls','transitions'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    ns=dict(pd=pd,np=np,re=re,L=L,save=save)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    original=ns['clhls_items']
    def corrected(frame,w,raw):
        items,maps=original(frame,w,raw)
        suffix='' if frame==w else '_'+str(w)[2:]
        var='g131'+suffix;q=pd.to_numeric(raw[var],errors='coerce')
        observed=set(q.dropna().tolist())
        known=csv(G1/'g131_code_frequency_by_frame_wave.csv',dtype={'raw_code':str})
        prior=set(pd.to_numeric(known[(known.frame==frame)&(known.wave==w)].raw_code,errors='coerce').dropna())
        assert observed<=prior,(frame,w,observed-prior)
        assert all(x in [0,88,99,-7,-9] or (x>0 and float(x).is_integer()) for x in observed)
        items['serious_illness']=serious(raw[var])
        maps=[(i,v,'v7:0 none; positive integer episodes deficit;88 bedridden separate deficit;99/negative/system missing remain missing') if i=='serious_illness' else (i,v,r) for i,v,r in maps]
        return items,maps
    ns['clhls_items']=corrected
    # Targeted missingness tests run before a corrected dataset is built.
    tests=[]
    for val,expected in [('0',0.),('1',1.),('33',1.),('88',1.),('99',np.nan),('-7',np.nan),('-9',np.nan),('',np.nan),('NaN',np.nan),('BAD_STRING',np.nan),('1.5',np.nan),('100',1.)]:
        actual=serious(pd.Series([val])).iloc[0]
        ok=(pd.isna(actual) and pd.isna(expected)) or actual==expected
        tests.append(dict(item='serious_illness',input=val,expected=expected,actual=actual,status='PASS' if ok else 'FAIL',scope='100 is synthetic implementation test, not an observed response'))
        assert ok,val
    tp=OUT/'fi_recode_targeted_unit_tests.csv';tb=pd.DataFrame(tests).to_csv(index=False).encode('utf-8-sig')
    if tp.exists(): assert '--resume-verified-outputs' in sys.argv and tp.read_bytes()==tb
    else: tp.write_bytes(tb)
    longs,maps=ns['final_clhls']()
    diffs=[];ivdiff=[];validation=[];unaffected=[];trans=[];flows=[];allnew=[]
    for new in longs:
        frame=int(new.baseline_frame.iloc[0]);new=new.copy();new.baseline_frame=new.baseline_frame.astype(int);new.wave=new.wave.astype(int)
        old=csv(OLD/f'clhls_frame{frame}_analysis_ready_long.csv',dtype={'person_id':str})
        assert len(old)==len(new) and list(old.person_id)==list(new.person_id) and list(old.wave)==list(new.wave)
        items=[m['item'] for m in maps if m['baseline_frame']==frame and m['wave']==frame]
        assert len(items)==44
        for w,g in new.groupby('wave',sort=True):
            raw=csv(L/f'clhls_frame{frame}_wave{w}_FI_candidate_raw.csv',dtype=str,keep_default_na=False)
            assert len(raw)==len(g) and raw.person_id.is_unique
            orig,mp=original(frame,w,raw)
            independent={}
            suffix='' if w==frame else '_'+str(w)[2:]
            for item,var,rule in mp:
                b=var.removesuffix(suffix) if suffix else var
                q=pd.to_numeric(raw[var],errors='coerce')
                if item=='serious_illness':
                    def scalar(x):
                        if not np.isfinite(x) or x<0 or x==99 or x%1: return np.nan
                        if x==0:return 0.
                        if x==88:return 1.
                        return 1.
                    z=q.map(scalar)
                elif b.startswith('g15'):z=q.map({1:1.,2:0.})
                else:
                    high=5 if b.startswith('b') else (4 if b in ['g1','h1'] else 3)
                    q=q.where(q.isin(range(1,high+1)))
                    z=(high-q)/(high-1) if b in ['b23','b24','b26'] else (q-1)/(high-1)
                independent[item]=z
                a=g.set_index('person_id').loc[raw.person_id,item].to_numpy(float)
                equal=np.isclose(a,z.to_numpy(float),atol=1e-12,rtol=0,equal_nan=True)
                assert equal.all(),(frame,w,item)
            scored=pd.DataFrame(independent);n=scored.notna().sum(axis=1)
            fi=scored.sum(axis=1,min_count=1)/n.replace(0,np.nan);fi=fi.where(n>=36)
            target=g.set_index('person_id').loc[raw.person_id]
            assert np.isclose(target.fi_score.to_numpy(float),fi,atol=1e-12,rtol=0,equal_nan=True).all()
            independent_state=pd.Series(np.where(fi.isna(),'unknown',np.where(fi<=.1,'robust',np.where(fi<=.21,'prefrail','frail'))),index=raw.index)
            src=csv(L/f'clhls_frame{frame}_person_wave_source_long.csv',dtype=str,keep_default_na=False)
            sw=src[src.wave.eq(str(w))].set_index('person_id').loc[raw.person_id]
            independent_state.loc[sw.death_evidence_this_wave.eq('True').to_numpy()]='death'
            assert target.state.to_numpy().tolist()==independent_state.tolist()
            assert target.fi_n_observed.tolist()==n.tolist()
            validation.append(dict(frame=frame,wave=w,rows=len(raw),unique_ids=raw.person_id.nunique(),duplicates=raw.person_id.duplicated().sum(),unmatched=0,item_cells=len(raw)*44,item_mismatches=0,fi_mismatches=0,state_mismatches=0,death_priority_mismatches=0,tolerance=1e-12,status='PASS'))
        mutable={'serious_illness','fi_n_observed','fi_completeness','fi_score','fi_state','state','death_absorbing_candidate'}
        for c in new:
            if c not in mutable:
                eq=same(old[c],new[c]);assert eq.all(),(frame,c)
                unaffected.append(dict(frame=frame,field=c,rows=len(new),mismatches=0,status='PASS'))
        d=new[['person_id','baseline_frame','wave','death_evidence_this_wave']].copy()
        for c in mutable:
            d['old_'+c]=old[c].to_numpy();d['new_'+c]=new[c].to_numpy();d[c+'_changed']=~same(old[c],new[c])
        d['fi_validity_changed']=old.fi_score.notna().ne(new.fi_score.notna()).to_numpy()
        diffs.append(d);allnew.append(new)
        oi=adjacent(old,frame);ni=adjacent(new,frame)
        ii=oi.merge(ni,on=['baseline_frame','person_id','wave_start','wave_end'],how='outer',validate='one_to_one',suffixes=('_old','_new'),indicator=True)
        assert ii._merge.eq('both').all()
        for tag in ['old','new']:
            ii['stageA_eligible_'+tag]=ii['baseline_eligible_'+tag]&ii['state_start_'+tag].ne('unknown')&ii['state_end_'+tag].ne('unknown')
        ii['state_start_changed']=ii.state_start_old.ne(ii.state_start_new)
        ii['state_end_changed']=ii.state_end_old.ne(ii.state_end_new)
        ivdiff.append(ii.drop(columns='_merge'))
        tr=ns['transitions'](new,'CLHLS',str(frame));trans.append(tr)
        flows.append(dict(frame=frame,person_waves=len(new),unique_ids=new.person_id.nunique(),duplicate_person_wave=new.duplicated(['person_id','wave']).sum(),all_adjacent_intervals=len(ni),old_stageA_intervals=int(ii.stageA_eligible_old.sum()),new_stageA_intervals=len(tr),old_valid_fi=int(old.fi_score.notna().sum()),new_valid_fi=int(new.fi_score.notna().sum()),fi_changed=int(d.fi_score_changed.sum()),state_changed=int(d.state_changed.sum()),death_changed=int(old.state.eq('death').ne(new.state.eq('death')).sum())))
        print('FI_INDEPENDENT_VALIDATION_PASS',frame,len(new),flows[-1],flush=True)
    save(pd.concat(allnew,ignore_index=True),'corrected_fi_person_wave.csv')
    save(pd.concat(diffs,ignore_index=True),'old_new_fi_state_diff.csv')
    save(pd.concat(ivdiff,ignore_index=True),'old_new_stageA_interval_diff.csv')
    save(pd.DataFrame(validation),'fi_independent_validation.csv')
    save(pd.DataFrame(unaffected),'unchanged_source_field_validation.csv')
    save(pd.DataFrame(flows),'new_sample_flow_by_frame.csv')
    save(pd.DataFrame(maps),'corrected_fi_item_map.csv')
    tr=pd.concat(trans,ignore_index=True);save(tr,'transition_intervals_audited.csv')
    save(tr.groupby(['baseline_frame','state_start','state_end']).size().reset_index(name='n_intervals'),'new_candidate_observed_endpoint_pairs.csv')
    interval_changes=[];model_flows=[];counts=[]
    for frame in [2008,2011,2014]:
        m=model_module(frame);frozen=m.OUT
        SEEN.add(frozen/'clhls_model_ready_v2.csv');SEEN.add(frozen/f'clhls_frame{frame}_model_input_all_intervals_v2.csv')
        m.OUT=OUT/'models'/f'frame{frame}';m.OUT.mkdir(parents=True,exist_ok=False)
        m.LONG=READY/f'clhls_frame{frame}_analysis_ready_long.csv';m.ANALYSIS=READY
        m._build_model_inputs()
        new=csv(m.OUT/'clhls_model_ready_v2.csv',dtype={'person_id':str})
        old=csv(frozen/'clhls_model_ready_v2.csv',dtype={'person_id':str})
        key=['person_id','wave_start','wave_end']
        audit=pd.concat(ivdiff,ignore_index=True);audit=audit[audit.baseline_frame.eq(frame)].copy()
        for tag,dd in [('old',old),('new',new)]:
            audit=audit.merge(dd[key+['weight']].rename(columns={'weight':'weight_'+tag}),on=key,how='left',validate='one_to_one')
            audit['model_eligible_'+tag]=audit['weight_'+tag].notna()
        audit['eligibility_change']=np.select([~audit.model_eligible_old&audit.model_eligible_new,audit.model_eligible_old&~audit.model_eligible_new],['ENTERED','EXITED'],default='UNCHANGED')
        audit['eligibility_reason']=np.where(audit.eligibility_change.eq('UNCHANGED'),'SAME_QUALIFICATION',np.where(~audit.stageA_eligible_new,'CORRECTED_UNKNOWN_STATE','QUALIFICATION_FROM_REBUILT_STATE'))
        interval_changes.append(audit)
        for (i,j),g in new.groupby(['i','j']): counts.append(dict(frame=frame,i=int(i),j=int(j),state_start=g.state_start.iloc[0],state_end=g.state_end.iloc[0],n_intervals=len(g)))
        common=old.merge(new,on=key,suffixes=('_old','_new'),validate='one_to_one')
        # New person set affects normalized weights and continuous covariate scaling.
        # Check all unscaled covariates/exposures on common intervals exactly.
        allowed={'i','j','state_start','state_end','weight','model_eligible'}
        for c in old.columns:
            if c in key or c in allowed or c.endswith('_c') or c.startswith('sleep_ns'):continue
            assert same(common[c+'_old'],common[c+'_new']).all(),(frame,c)
        model_flows.append(dict(frame=frame,old_fitted_persons=old.person_id.nunique(),new_fitted_persons=new.person_id.nunique(),old_fitted_intervals=len(old),new_fitted_intervals=len(new),entered_intervals=int(audit.eligibility_change.eq('ENTERED').sum()),exited_intervals=int(audit.eligibility_change.eq('EXITED').sum()),common_intervals=len(common),state_start_changed_common=int(common.state_start_old.ne(common.state_start_new).sum()),state_end_changed_common=int(common.state_end_old.ne(common.state_end_new).sum()),new_duplicate_interval_keys=int(new.duplicated(key).sum()),status='PASS'))
        print('MODEL_INPUT_REBUILD_PASS',frame,model_flows[-1],flush=True)
    save(pd.concat(interval_changes,ignore_index=True),'old_new_interval_eligibility_diff.csv')
    save(pd.DataFrame(model_flows),'new_model_sample_flow_by_frame.csv')
    save(pd.DataFrame(counts),'new_observed_endpoint_pairs.csv')
    for p in L.glob('clhls_frame*csv'):SEEN.add(p)
    files=[p for p in READY.glob('*') if p.is_file()]+[p for p in (OUT/'models').rglob('*') if p.is_file()]+[Path(__file__),OUT/'semantic_codebook_decision.md',OUT/'fi_recode_targeted_unit_tests.csv']
    inputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(SEEN)]
    outputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]
    manifest=dict(version='CLHLS_FI_CORRECTED_V7_20261024',created_utc=datetime.now(timezone.utc).isoformat(),status='PASS',gate='G2',g1='PASS_AUTHOR_WRITTEN_DECISION',hash_basis='per-file raw bytes SHA256; absolute paths sorted; manifest excluded from own hashes',rules=dict(items=44,min_observed=36,denominator='observed items',robust='FI<=0.10',prefrail='0.10<FI<=0.21',frail='FI>0.21',death='independent death evidence takes precedence',only_changed_item='serious_illness',model='unchanged'),inputs=inputs,outputs=outputs,formal_optimizer='NOT_RUN',model_flows=model_flows)
    write(READY/'corrected_stageA_manifest.json',manifest)
    print('G2_PASS_MANIFEST_FROZEN',sha(READY/'corrected_stageA_manifest.json'),flush=True)

if __name__=='__main__': main()
