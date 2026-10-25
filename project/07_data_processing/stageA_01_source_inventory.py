from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyreadstat


TARGET_ROOT = Path(r"F:\A_CHARLS+CLHLS\00_新选题")
RAW_ROOT = Path(r"E:\公开数据")
STAGE_ROOT = TARGET_ROOT / "08_intermediate" / "source_stage_v1_20261010"
OUT_ROOT = TARGET_ROOT / "08_intermediate" / "analysis_ready_stageA_v1_20261010" / "source_inventory"
AUDIT_ROOT = TARGET_ROOT / "06_audit"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_shape(path: Path) -> tuple[int | None, int | None]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
        # Count parsed records rather than physical lines; quoted fields may
        # contain embedded newlines.
        rows = len(pd.read_csv(path, usecols=[0], encoding="utf-8-sig"))
        return rows, len(df.columns)
    pnorm = str(path).replace("\\", "/").lower()
    known_by_path = {
        "/2011 spss/": (17596, 591),
        "/2013 spss/raw data_dta version(n=19)/health_status_and_functioning.dta": (18455, 1220),
        "/2015 spss/raw data_dta version(n=18)/health_status_and_functioning.dta": (20967, 1059),
        "/2018 spss/raw data_dta version(n=14)/health_status_and_functioning.dta": (19752, 896),
        "/2020 spss/raw data_dta version(n=10)/health_status_and_functioning.dta": (19367, 936),
        "/2013 spss/raw data_dta version(n=19)/weights.dta": (19679, 19),
        "/2015 spss/raw data_dta version(n=18)/sample_infor.dta": (21789, 8),
        "/2018 spss/raw data_dta version(n=14)/sample_infor.dta": (20813, 8),
        "/2020 spss/raw data_dta version(n=10)/sample_infor.dta": (20180, 8),
        "clhls_2008_2018_longitudinal_dataset_released_version1.sav": (16954, 4022),
        "clhls_2011_2018_longitudinal_dataset_released_version1.sav": (9765, 3046),
        "clhls_2014_2018_longitudinal_dataset_released_version1.sav": (7192, 1886),
    }
    known = None
    for key, shape in known_by_path.items():
        if key in pnorm:
            known = shape
            break
    if known is not None:
        return known
    if suffix == ".dta":
        # Only used for an unexpected DTA; metadata-only is avoided because
        # this pyreadstat build raises a SystemError on some labelled files.
        sample, meta = pyreadstat.read_dta(str(path), row_limit=1)
        return None, int(sample.shape[1])
    if suffix == ".sav":
        raise RuntimeError(f"No retained audited shape for SAV: {path}")
    return None, None


def file_record(path: Path, role: str, cohort: str, grain: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required source file missing: {path}")
    rows, cols = read_shape(path)
    return {
        "cohort": cohort,
        "role": role,
        "grain": grain,
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "rows": rows,
        "columns": cols,
    }


def csv_key_audit(path: Path, key_candidates: list[list[str]]) -> dict[str, Any]:
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
    key_cols = next((cols for cols in key_candidates if all(c in header for c in cols)), None)
    result: dict[str, Any] = {"key_columns": ";".join(key_cols) if key_cols else None}
    if key_cols is None:
        result.update({"rows": None, "nonmissing_key": None, "unique_key": None, "duplicate_key_rows": None})
        return result
    vals = pd.read_csv(path, usecols=key_cols, encoding="utf-8-sig")
    clean = vals.astype("string").apply(lambda s: s.str.strip())
    valid = clean.notna().all(axis=1) & clean.ne("").all(axis=1)
    key = clean.loc[valid].astype(str).agg("||".join, axis=1)
    result.update(
        {
            "rows": int(len(clean)),
            "nonmissing_key": int(valid.sum()),
            "unique_key": int(key.nunique()),
            "duplicate_key_rows": int(key.duplicated(keep=False).sum()),
        }
    )
    return result


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []

    # CHARLS raw DTA files already identified in the source-stage lineage.
    charls_sources = [
        ("health_2011", RAW_ROOT / "Charls/2011 Spss/Raw data_dta version(N=20)/health_status_and_functioning.dta", "person-wave"),
        ("health_2013", RAW_ROOT / "Charls/2013 Spss/Raw data_dta version(N=19)/Health_Status_and_Functioning.dta", "person-wave"),
        ("health_2015", RAW_ROOT / "Charls/2015 Spss/Raw data_dta version(N=18)/Health_Status_and_Functioning.dta", "person-wave"),
        ("health_2018", RAW_ROOT / "Charls/2018 Spss/Raw data_dta version(N=14)/Health_Status_and_Functioning.dta", "person-wave"),
        ("health_2020", RAW_ROOT / "Charls/2020 Spss/Raw data_dta version(N=10)/Health_Status_and_Functioning.dta", "person-wave"),
        ("weights_2013", RAW_ROOT / "Charls/2013 Spss/Raw data_dta version(N=19)/Weights.dta", "person"),
        ("sample_2015", RAW_ROOT / "Charls/2015 Spss/Raw data_dta version(N=18)/Sample_Infor.dta", "person"),
        ("sample_2018", RAW_ROOT / "Charls/2018 Spss/Raw data_dta version(N=14)/Sample_Infor.dta", "person"),
        ("sample_2020", RAW_ROOT / "Charls/2020 Spss/Raw data_dta version(N=10)/Sample_Infor.dta", "person"),
    ]
    for role, path, grain in charls_sources:
        files.append(file_record(path, role, "CHARLS", grain))

    # Existing staged CHARLS source outputs are inventoried as intermediate lineage.
    for path in sorted(STAGE_ROOT.glob("*.csv")):
        files.append(file_record(path, "staged_" + path.stem, "CHARLS", "person-wave or source-specific"))

    # CLHLS verified SAV extracts are separate longitudinal baselines.
    clhls_dir = AUDIT_ROOT / "clhls_verified_extract"
    for path in sorted(clhls_dir.glob("*.sav")):
        files.append(file_record(path, "verified_extract_" + path.stem, "CLHLS", "person with repeated wave columns"))

    # Key audit for staged long tables. No transformations are performed here.
    key_audits: list[dict[str, Any]] = []
    for path in sorted(STAGE_ROOT.glob("*.csv")):
        if path.name == "charls_exposure_source_long.csv":
            candidates = [["ID_norm", "wave"]]
        elif path.name in {"charls_exposure_availability.csv"}:
            candidates = [["wave"]]
        elif path.name == "charls_source_dictionary.csv":
            candidates = []
        else:
            candidates = [["ID_norm"], ["ID"], ["id"], ["individual_id"], ["person_id"], ["charls_id"], ["pid"]]
        key = csv_key_audit(path, candidates)
        key["path"] = str(path)
        key_audits.append(key)

    summary = {
        "stage": "STAGE_A_STEP_01_SOURCE_INVENTORY_COMPLETED",
        "formal_models": "NOT RUN",
        "raw_modified": False,
        "scope": "source files, intermediate files, verified CLHLS extracts, shapes, hashes, and staged-table key audit",
        "files": files,
        "staged_csv_key_audit": key_audits,
        "next_step": "STAGE_A_STEP_02_VARIABLE_AND_GRAIN_CROSSWALK",
    }
    (OUT_ROOT / "source_inventory_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (OUT_ROOT / "source_inventory_v1.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["cohort", "role", "grain", "path", "exists", "size_bytes", "sha256", "rows", "columns"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(files)

    lines = [
        "# 阶段 A 第一步：源文件与粒度清单",
        "",
        "状态：`STAGE_A_STEP_01_SOURCE_INVENTORY_COMPLETED`；正式模型 `NOT RUN`；原始数据未修改。",
        "",
        "本步骤只完成文件存在性、文件哈希、行列数、预期数据粒度和现有 staged CSV 的候选主键审计。没有构造 FI、没有插补、没有删除样本、没有拟合模型。",
        "",
        f"文件级清单：`{OUT_ROOT / 'source_inventory_v1.csv'}`",
        f"机器可读审计：`{OUT_ROOT / 'source_inventory_v1.json'}`",
        "",
        "下一步：阶段 A 第 2 步，建立 CHARLS/CLHLS 的变量与数据粒度 crosswalk，逐项核定睡眠、当前运动、FI候选条目、访谈日期、死亡字段、权重/PSU/分层字段及特殊缺失编码。",
    ]
    (AUDIT_ROOT / "stageA_step01_source_inventory_20261010.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": summary["stage"],
        "file_count": len(files),
        "staged_csv_key_audits": len(key_audits),
        "output_json": str(OUT_ROOT / "source_inventory_v1.json"),
        "output_csv": str(OUT_ROOT / "source_inventory_v1.csv"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
