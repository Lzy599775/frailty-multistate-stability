from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pyreadstat


TARGET_ROOT = Path(r"F:\A_CHARLS+CLHLS\00_新选题")
RAW_ROOT = Path(r"E:\公开数据\Charls")
OUT_ROOT = TARGET_ROOT / "08_intermediate" / "analysis_ready_stageA_v1_20261010" / "variable_crosswalk"
AUDIT_ROOT = TARGET_ROOT / "06_audit"
CLHLS_DICT_ROOT = AUDIT_ROOT / "clhls_variable_audit_20261010"
CHARLS_AUDIT_JSON = AUDIT_ROOT / "charls_variable_audit_20261010_134840.json"
CLHLS_CLEAN_TSV = TARGET_ROOT / "08_intermediate" / "analysis_ready_stageA_v1_20261010" / "variable_crosswalk" / "clhls_dictionary_clean.tsv"


PATTERNS = {
    "sleep": r"sleep|nap|睡眠|午睡",
    "activity": r"exercise|physical activity|vigorous|moderate|walking|walk|activity|运动|锻炼|活动",
    "frailty": r"frail|frailty|grip|handgrip|gait|walking speed|fatigue|exhaust|bmi|body mass|height|weight|adl|iadl|functional|function|memory|cognitive|depress|ces[- ]?d|疲劳|握力|步速|认知|抑郁|功能",
    "death": r"death|died|mortality|surviv|vital|cause of death|end of life|dead|死亡|生存|死因",
    "date_interview": r"interview|yearin|monthin|dayin|date|year|month|day|访谈|调查日期",
    "design_weight": r"survey weight|sample weight|longitudinal weight|individual weight|weighting|wtres|weight",
    "design_cluster": r"psu|cluster|strata|community|primary sampling|抽样|社区",
    "covariate": r"age|sex|gender|education|educ|marital|marriage|urban|rural|residence|smok|drink|income|chronic|hypertension|diabet|stroke|cancer|province|城乡|年龄|性别|教育|婚姻|吸烟|饮酒|慢病",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def candidate_domains(variable: str, label: str) -> list[str]:
    text = f"{variable} {label}".lower()
    return [name for name, pattern in PATTERNS.items() if re.search(pattern, text, flags=re.I)]


def charls_source_paths() -> list[Path]:
    # Scan only public source files relevant to the planned analysis; raw data
    # remain read-only and no values are exported in this step.
    paths: list[Path] = []
    for wave_dir in sorted(RAW_ROOT.glob("* Spss")):
        raw_dirs = sorted(wave_dir.glob("Raw data_dta version*"))
        for raw_dir in raw_dirs:
            for p in sorted(raw_dir.glob("*.dta")):
                stem = p.stem.lower()
                if any(token in stem for token in [
                    "health", "demographic", "sample", "weight", "psu", "exit", "life", "cognition", "biomarker", "community"
                ]):
                    paths.append(p)
    harmonized = RAW_ROOT / "Harmonized Charls Spss" / "1. Regular wave" / "H_CHARLS_D_Data.dta"
    if harmonized.exists():
        paths.append(harmonized)
    return sorted(set(paths))


def scan_charls() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not CHARLS_AUDIT_JSON.exists():
        raise FileNotFoundError(f"Required retained CHARLS audit artifact missing: {CHARLS_AUDIT_JSON}")
    # Reuse the completed, hash-recorded CHARLS variable audit instead of
    # rescanning labelled DTA files. This preserves the exact prior evidence
    # and avoids a second uncontrolled parser pass.
    audit = json.loads(CHARLS_AUDIT_JSON.read_text(encoding="utf-8"))
    for record in audit.get("records", []):
        path = Path(record["path"])
        labels = record.get("labels", {}) or {}
        value_label_keys = record.get("value_label_keys", {}) or {}
        for variable in record.get("columns", []):
            label = str(labels.get(variable, "") or "")
            domains = candidate_domains(variable, label)
            if not domains:
                continue
            rows.append({
                "cohort": "CHARLS",
                "source_file": str(path),
                "source_sha256": record.get("sha256", "SOURCE_AUDIT_JSON"),
                "source_variable": variable,
                "label": label,
                "value_labels": json.dumps(value_label_keys.get(variable, []), ensure_ascii=False),
                "user_missing": "NOT_READ_IN_THIS_STEP",
                "candidate_domains": ";".join(domains),
                "grain_hint": "person-wave" if "health" in path.stem.lower() or "H_CHARLS_D_Data" in path.name else "person/source-specific",
                "crosswalk_status": "CANDIDATE_REQUIRES_ITEM_LEVEL_REVIEW",
            })
    return rows


def scan_clhls() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not CLHLS_CLEAN_TSV.exists():
        raise FileNotFoundError(f"Required clean CLHLS dictionary missing: {CLHLS_CLEAN_TSV}")
    d = pd.read_csv(CLHLS_CLEAN_TSV, sep="\t", encoding="utf-8-sig", dtype="string")
    for _, r in d.iterrows():
            variable = str(r.get("variable", "") or "")
            label = str(r.get("label", "") or "")
            domains = candidate_domains(variable, label)
            if not domains:
                continue
            rows.append({
                "cohort": "CLHLS",
                "source_file": str(AUDIT_ROOT / "clhls_verified_extract" / str(r.get("source_file", ""))),
                "source_sha256": "SEE_SOURCE_INVENTORY_STEP_01",
                "source_variable": variable,
                "label": label,
                "value_labels": str(r.get("value_labels", "") or ""),
                "user_missing": str(r.get("user_missing", "") or ""),
                "candidate_domains": ";".join(domains),
                "grain_hint": "person with repeated wave columns",
                "crosswalk_status": "CANDIDATE_REQUIRES_ITEM_LEVEL_REVIEW",
                "n_missing_from_prior_audit": r.get("n_missing", ""),
            })
    return rows


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = scan_charls() + scan_clhls()
    d = pd.DataFrame(rows)
    d = d.sort_values(["cohort", "candidate_domains", "source_file", "source_variable"], kind="stable")
    out_csv = OUT_ROOT / "variable_crosswalk_candidates_v1.csv"
    d.to_csv(out_csv, index=False, encoding="utf-8-sig")
    summary = {
        "stage": "STAGE_A_STEP_02_VARIABLE_CROSSWALK_CANDIDATES_COMPLETED",
        "formal_models": "NOT RUN",
        "raw_modified": False,
        "candidate_rows": int(len(d)),
        "charls_rows": int((d.cohort == "CHARLS").sum()),
        "clhls_rows": int((d.cohort == "CLHLS").sum()),
        "domain_counts": {k: int(d.candidate_domains.fillna("").str.contains(k, regex=False).sum()) for k in PATTERNS},
        "source_files_charls": len({r["source_file"] for r in rows if r["cohort"] == "CHARLS"}),
        "source_dictionary_files_clhls": 3,
        "source_sha256": {str(r["source_file"]): r["source_sha256"] for r in rows if r["cohort"] == "CHARLS"},
        "next_step": "STAGE_A_STEP_03_ITEM_LEVEL_FI_AND_EXPOSURE_AUDIT",
    }
    (OUT_ROOT / "variable_crosswalk_candidates_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = "\n".join([
        "# 阶段 A 第二步：变量与粒度 crosswalk 候选清单",
        "",
        "状态：`STAGE_A_STEP_02_VARIABLE_CROSSWALK_CANDIDATES_COMPLETED`；本步骤只识别候选变量，不自动把候选变量纳入正式分析。",
        "",
        f"候选变量清单：`{out_csv}`",
        f"机器可读摘要：`{OUT_ROOT / 'variable_crosswalk_candidates_v1.json'}`",
        "",
        "扫描范围包括 CHARLS 各波次健康、人口、Sample/Weights、PSU、Exit、Cognition、Biomarker、Community 和 Harmonized 文件，以及三个 CLHLS longitudinal SAV 的既有官方字典。",
        "",
        "下一步：阶段 A 第 3 步，逐题审查 FI、睡眠、当前运动和协变量候选条目，记录变量标签、值标签、特殊缺失、跨波可用性、与暴露重叠和是否进入分析数据。",
    ])
    (AUDIT_ROOT / "stageA_step02_variable_crosswalk_20261010.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
