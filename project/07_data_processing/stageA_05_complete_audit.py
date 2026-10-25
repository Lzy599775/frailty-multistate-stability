"""Stage A final source audit. Produces audit evidence only; no FI score or models."""
from pathlib import Path
import hashlib, json, re
import pandas as pd
import numpy as np

ROOT=Path(r'F:\A_CHARLS+CLHLS\00_新选题')
BASE=ROOT/'08_intermediate/analysis_ready_stageA_v1_20261010'
L=BASE/'longitudinal_sources'; OUT=BASE/'final_audit'
DIC=pd.read_csv(BASE/'reconciled_dictionary/variable_crosswalk_complete_v1.csv',dtype=str,keep_default_na=False)

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write(d,n):
 p=OUT/n;d.to_csv(p,index=False,encoding='utf-8-sig');return {'path':str(p),'rows':len(d),'cols':d.shape[1],'sha256':sha(p)}

def item_audit():
 rows=[]
 for p in sorted(L.glob('*_FI_candidate_raw.csv')):
  d=pd.read_csv(p,dtype=str,keep_default_na=False)
  cohort='CHARLS' if p.name.startswith('charls_') else 'CLHLS'
  wave=re.search(r'(?:wave|charls_)(\d{4})',p.name).group(1)
  frame=re.search(r'frame(\d{4})',p.name);frame=frame.group(1) if frame else ''
  source=(DIC[DIC.export_file.eq(str(p))] if 'export_file' in DIC else pd.DataFrame())
  # The candidate raw table is an explicit extraction; every non-ID field is audited.
  for v in [x for x in d.columns if x not in ['person_id','id','wave','baseline_frame']]:
   s=d[v];num=pd.to_numeric(s,errors='coerce');nonempty=s.ne('')
   counts=s[nonempty].value_counts().head(12).to_dict()
   dr=DIC[DIC.source_variable.eq(v)&DIC.cohort.eq(cohort)]
   rows.append(dict(cohort=cohort,baseline_frame=frame,wave=wave,source_extract=p.name,source_variable=v,label=' | '.join(dr.label.unique()) if len(dr) else '',value_labels=' | '.join(dr.value_labels.unique()) if len(dr) else '',n_rows=len(s),n_blank=int((~nonempty).sum()),n_nonblank=int(nonempty.sum()),n_numeric=int(num.notna().sum()),n_non_numeric_nonblank=int((nonempty&num.isna()).sum()),n_negative_numeric=int(num.lt(0).sum()),numeric_min=num.min(),numeric_max=num.max(),prevalent_values_json=json.dumps(counts,ensure_ascii=False),audit_status='CANDIDATE_ITEM_AUDITED_NOT_RECODED'))
 return pd.DataFrame(rows)

def exposure_audit():
 rows=[]
 p=L/'charls_person_wave_source_long.csv';d=pd.read_csv(p,dtype=str,keep_default_na=False)
 for w,g in d.groupby('wave'):
  an=pd.to_numeric(g.activity_binary,errors='coerce')
  rows.append(dict(cohort='CHARLS',baseline_frame='2011',wave=w,rows=len(g),unique_id=g.person_id.nunique(),health_observed=int(g.health_observed.eq('True').sum()),sleep_valid_physical=int(g.sleep_hours_valid_physical.ne('').sum()),sleep_outside_or_special=int((~g.sleep_value_class.isin(['in_physical_range','system_or_no_health',''])).sum()),sleep_zero=int(g.sleep_value_class.eq('zero_flag').sum()),activity_yes=int(an.eq(1).sum()),activity_no=int(an.eq(0).sum()),activity_unknown=int(an.isna().sum()),activity_design_unknown=int(g.activity_design_flag.eq('NOT_IDENTIFIED_no_ordinary_imputation').sum())))
 for p in sorted(L.glob('clhls_frame*_person_wave_source_long.csv')):
  d=pd.read_csv(p,dtype=str,keep_default_na=False);frame=re.search(r'frame(\d+)',p.name).group(1)
  for w,g in d.groupby('wave'):
   an=pd.to_numeric(g.activity_binary,errors='coerce')
   rows.append(dict(cohort='CLHLS',baseline_frame=frame,wave=w,rows=len(g),unique_id=g.person_id.nunique(),health_observed='',sleep_valid_physical=int(g.sleep_hours_valid_physical.ne('').sum()),sleep_outside_or_special=int((~g.sleep_value_class.isin(['in_physical_range','system_missing',''])).sum()),sleep_zero=int(g.sleep_value_class.eq('zero_flag').sum()),activity_yes=int(an.eq(1).sum()),activity_no=int(an.eq(0).sum()),activity_unknown=int(an.isna().sum()),activity_design_unknown='not_applicable'))
 return pd.DataFrame(rows)

def death_audit():
 rows=[]
 p=L/'charls_death_date_source_candidates.csv';d=pd.read_csv(p,dtype=str,keep_default_na=False)
 for src,g in d.groupby('death_source'):
  rows.append(dict(cohort='CHARLS',source=src,rows=len(g),unique_id=g.person_id.nunique(),lower_present=int(g.death_date_lower.ne('').sum()),upper_present=int(g.death_date_upper.ne('').sum()),exact_day=int(g.death_date_precision.eq('day').sum()),month_interval=int(g.death_date_precision.eq('month').sum()),year_interval=int(g.death_date_precision.eq('year').sum()),invalid=int(g.death_date_precision.str.contains('invalid|missing',na=False).sum()),reversed_bounds=int((g.death_date_lower>g.death_date_upper).sum())))
 for p in sorted(L.glob('clhls_frame*_person_wave_source_long.csv')):
  d=pd.read_csv(p,dtype=str,keep_default_na=False);frame=re.search(r'frame(\d+)',p.name).group(1)
  for w,g in d.groupby('wave'):
   rows.append(dict(cohort='CLHLS',source=f'frame{frame}',wave=w,rows=len(g),unique_id=g.person_id.nunique(),lower_present=int(g.death_date_lower.ne('').sum()),upper_present=int(g.death_date_upper.ne('').sum()),exact_day=int(g.death_date_precision.eq('day').sum()),month_interval=int(g.death_date_precision.eq('month').sum()),year_interval=int(g.death_date_precision.eq('year').sum()),invalid=int(g.death_date_precision.str.contains('invalid|missing',na=False).sum()),reversed_bounds=int((g.death_date_lower>g.death_date_upper).sum())))
 return pd.DataFrame(rows)

def main():
 OUT.mkdir(exist_ok=True)
 ia=item_audit();ea=exposure_audit();da=death_audit()
 arts=[write(ia,'fi_item_audit_v1.csv'),write(ea,'exposure_audit_v1.csv'),write(da,'death_date_audit_v1.csv')]
 issues=pd.read_csv(L/'source_state_sequence_issues.csv',dtype=str,keep_default_na=False)
 joins=pd.read_csv(L/'join_lineage.csv',dtype=str,keep_default_na=False)
 manifest=json.loads((L/'source_longitudinal_manifest.json').read_text(encoding='utf-8'))
 summary={'stage':'A_SOURCE_AND_VARIABLE_AUDIT_COMPLETED_PENDING_SCIENTIFIC_ADJUDICATION','raw_modified':False,'formal_models':'NOT RUN','FI_scoring':'NOT RUN','audit_artifacts':arts,'candidate_items_audited':len(ia),'candidate_item_sources':int(ia.source_extract.nunique()),'sequence_issue_rows':len(issues),'sequence_issue_counts':issues.issue.value_counts().to_dict(),'join_rows':len(joins),'join_unmatched_total':int(pd.to_numeric(joins.unmatched,errors='coerce').fillna(0).sum()),'date_reversed_bounds':int(pd.to_numeric(da.reversed_bounds,errors='coerce').fillna(0).sum()),'important_scientific_gates':['final FI item set and direction','FI denominator completeness threshold','age inclusion threshold for each cohort','frail state cutpoints and allowed transitions','CHARLS design-specific activity administration flag','CLHLS baseline-frame estimand and treatment of overlapping frames'],'next_task':'user adjudicates one consolidated FI/age/transition data gate; then recode final FI/state fields without reopening source audit'}
 (OUT/'stageA_final_audit_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 report=['# 阶段 A：两队列源级 longitudinal 数据与变量/状态审计','', '状态：`A_SOURCE_AND_VARIABLE_AUDIT_COMPLETED_PENDING_SCIENTIFIC_ADJUDICATION`。所有当前可独立完成的源数据、变量、暴露、日期、死亡、权重来源、连接粒度和状态序列审计已在本阶段完成；正式 FI/模型均未运行。','', '## 已完成',f'- CHARLS 5 波和 CLHLS 3 个基线框架分别保留为 person-wave 长表；未把不同 CLHLS 框架直接 append。',f'- 字典 `{BASE}/reconciled_dictionary/variable_crosswalk_complete_v1.csv`：{len(DIC)} 个变量记录；候选 FI 字段仅作为候选，未评分。',f'- FI 候选逐字段审计 `{OUT}/fi_item_audit_v1.csv`：{len(ia)} 条。',f'- 暴露审计 `{OUT}/exposure_audit_v1.csv`；死亡日期边界审计 `{OUT}/death_date_audit_v1.csv`。',f'- 连接血缘 `{L}/join_lineage.csv`、状态序列问题 `{L}/source_state_sequence_issues.csv`。','', '## 必须在阶段 A 内一次性裁决的科学定义', '- 队列纳入年龄、最终 FI 项目、编码方向和最小完整度；候选项不能直接视为最终 FI。', '- robust/prefrail/frail 阈值及允许的恢复/恶化转移；状态不足要标记不可估计，不能回看结果改矩阵。', '- CHARLS 2013/2015 活动题的实际施测/设计性缺失标记；当前导出明确不把未知当作不运动。', '- CLHLS 2008、2011、2014 框架的主复制层级；重叠 ID 不能作为独立复制样本。', '- 死亡区间精度只能使用上下界；不生成中点或第 1 天。','', '## 下一步具体流程', '1. 对以上裁决形成用户批准的数据 gate 表；', '2. 依据 gate 对候选字段做确定性重编码，输出最终 FI 分数、状态、暴露和可用观察区间；', '3. 在同一阶段 A 内重新核对 FI 分母、状态转移起点/事件数、死亡吸收和权重 positivity；', '4. 生成 `analysis_ready_manifest`，并把不可估计项目清楚标记；只有这一步完成后，才进入阶段 B 的模型资格 dry-run。']
 (ROOT/'06_audit/stageA_final_audit_20261010.md').write_text('\n'.join(report),encoding='utf-8')
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
