"""CLHLS replication of the frozen v8 penalized panel CTMC.

This file is an independent execution entry point.  It preserves the v8
likelihood, 225 transition-specific parameters, fixed ridge penalty,
attained-age segmentation, and cluster sandwich, while constructing one
analysis table for a selected CLHLS baseline frame (2008, 2011 or 2014).
The raw CLHLS exports are read only; all outputs are written below a new
frame-specific stage-B directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import sys
import time
import traceback
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from patsy import dmatrix
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize
import stable_log_ctmc as _stable_log_ctmc

ROOT = Path(r"F:\A_CHARLS+CLHLS\00_新选题")
BASE = ROOT / "08_intermediate/analysis_ready_stageA_v1_20261010"
ANALYSIS = BASE / "analysis_ready_final_v1"
_frame_args = [a for a in sys.argv[1:] if a in {"2008", "2011", "2014"}]
FRAME = int(_frame_args[0]) if _frame_args else int(os.environ.get("CLHLS_FRAME", "2008"))
if FRAME not in {2008, 2011, 2014}:
    raise ValueError("CLHLS frame must be 2008, 2011 or 2014")
OUT = Path(os.environ.get("CLHLS_OUT_ROOT", str(BASE / "stageB_clhls_penalized_execution_v8g"))) / f"frame{FRAME}"
SCRIPT = Path(__file__).resolve()
STATE = {"robust": 0, "prefrail": 1, "frail": 2, "death": 3}
TRANS = np.array([(r, j) for r in range(3) for j in range(4) if r != j], dtype=np.int64)
SEED = 20261011
AGE_CUTS = [55.0, 65.0, 75.0, 85.0]
SLEEP_KNOTS = {2008: [4.0, 7.0, 9.0, 12.0], 2011: [4.0, 7.0, 8.0, 12.0], 2014: [4.0, 7.0, 8.0, 12.0]}[FRAME]
SLEEP_REFERENCE = 7.0
DEMO = ["baseline_age_c", "female", "urban", "education_primary", "education_middle", "education_high", "partnered"]
BEHAVIOR = DEMO + ["smoking", "drinking"]
FULL = BEHAVIOR + ["bmi_c", "disease_count_c", "memory_c", "cesd9_c"]
EXPOSURE = ["sleep_ns1", "sleep_ns2", "sleep_ns3", "activity", "sleep_ns1_activity", "sleep_ns2_activity", "sleep_ns3_activity"]
FEATURES = ["intercept"] + [f"age_band_{int(a)}" for a in AGE_CUTS] + EXPOSURE + FULL
RAW = BASE / f"clhls_sources/clhls_{FRAME}_2018_longitudinal_dataset_released_version1_raw.csv"
LONG = ANALYSIS / f"clhls_frame{FRAME}_analysis_ready_long.csv"
INPUTS = [ANALYSIS / "analysis_ready_manifest_v1.json", ANALYSIS / "transition_intervals_audited.csv", LONG, RAW,
          BASE / "clhls_sources/clhls_dictionary_utf8.csv"]


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def _raise_json(o):
    raise TypeError(f"Unsupported JSON type: {type(o).__name__}")


def write(path, obj):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Immutable evidence already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False,
                               default=lambda o: o.item() if isinstance(o, np.generic) else _raise_json(o)), encoding="utf-8")


def num(s):
    return pd.to_numeric(s, errors="coerce")


def environment():
    buf = io.StringIO()
    with redirect_stdout(buf):
        np.show_config()
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "patsy", "numba", "llvmlite"]},
            "numpy_blas_configuration": buf.getvalue(), "jit": "numba njit; fastmath=False; parallel=False; cache=False",
            "hardware_fit_claim": "NOT MADE"}


def original_config():
    return {"mode": "EMPIRICAL", "cohort": "CLHLS", "baseline_frame": FRAME, "seed": SEED, "states": STATE,
            "transition_order": TRANS.tolist(), "age_cuts": AGE_CUTS,
            "time": "attained age at interval start; exit=start+frozen interval_years; coarse survey-wave years, not exact dates",
            "age_structure": "transition-specific intercept plus attained-age bands <55,55-64,65-74,75-84,85+; baseline age adjusted separately",
            "sleep_knots": SLEEP_KNOTS, "sleep_reference": SLEEP_REFERENCE,
            "sleep_basis": "patsy natural spline basis, subtract basis at 7h, retain first 3 of 4 columns",
            "features": FEATURES, "blocks": {"core": DEMO, "behavior": BEHAVIOR, "full": FULL},
            "sample": "same full complete-case intervals and subjects for all three blocks; no imputation; death->death retained in all-interval table but excluded from health-transition likelihood",
            "baseline_recode": {"sex": "a1: 1 male=0, 2 female=1", "residence": "residenc: city/town=1, rural=0",
                "education": "f1 years schooling: 1-5 primary, 6-8 middle, >=9 high; 0 below primary",
                "marital": "f41: 1/2 partnered, 3/4/5 other", "smoking_drinking": "d71/d81: 1 yes, 2 no",
                "bmi": "g101 kg / (g1021 cm/100)^2; valid 10-80; special codes missing",
                "disease": "g15a1-g15v1, 1 yes/2 no; require >=80% observed; count yes",
                "memory": "sum of 11 binary c21a-c21c,c31a-c31e,c41a-c41c; all 11 required",
                "cesd9": "CLHLS 7-item psychological burden proxy from b21-b27; positive items raw-1, negative items 5-raw; all 7 required; name retained for 25-feature v8 position"},
            "health_conditioning": "baseline disease count, memory and CLHLS psychological burden proxy are observational adjustment variables; not FI components beyond source definitions",
            "weight": "baseline positive CLHLS weight_raw, fixed across all intervals, mean-normalized among unique complete fitted subjects",
            "weight_estimand": "baseline individual-weighted complete-case conditional association; not nationally unbiased survival effect, not IPCW",
            "variance": "penalized Hessian bread; individual cluster data score meat; no inverse/pseudoinverse fallback",
            "optimizer": {"method": "L-BFGS-B", "bounds": None, "maxiter": 2000, "maxls": 40, "maxcor": 50, "ftol": 1e-15, "gtol": 1e-10,
                          "objective_scaling": "divide weighted penalized objective and score by number of unique fitted subjects"},
            "convergence": {"sample_n_persons": 2000, "selection": "sorted unique fitted IDs; RNG20261011 without replacement; all their eligible intervals",
                            "starts": ["event count/person-years with all slopes zero", "same intercepts and alternating +/-0.05 slopes"],
                            "total_gradient_inf_max": 1e-4, "between_start_parameter_delta_max": 1e-3, "between_start_loglik_delta_max": 1e-6,
                            "information_min_eigenvalue": 0.0, "information_condition_max": 1e10, "probability_tolerance": 1e-9,
                            "exact_repeat": "identical start0 objective and gradient evaluated twice; max difference<=1e-12", "all_nine_endpoint_events_required": True, "qualification_amendment": "v8c restores original CHARLS v8 total gradient gate 1e-4; no gate relaxed"},
            "numeric": {"rate": "exp(eta) without clipping or coefficient bounds", "matrix_exp": "Higham Pade13; validated against scipy",
                        "gradient": "adjoint Frechet derivative through full matrix-product chain rule", "gradient_finite_difference_tolerance": 2e-6, "hessian_step": 1e-4},
            "formal_order": ["core", "behavior", "full"], "primary_reporting_block": "full", "result_claim": "observational penalized association, not causal effect"}


def config():
    c = original_config()
    c["execution_version"] = "v7_1_controlled_recovery_stable_log_r2"
    c["numeric"]["matrix_exp"] = "CTMC uniformization; log-domain squaring/product; fixed tail1e-18"
    c["numeric"]["gradient"] = "analytic uniformization directions and log-domain panel chain rule"
    c["numeric"]["objective_reduction"] = "Kahan compensated sum of unchanged interval contributions"
    c["optimizer"]["gtol"] = "0.5 * total_gradient_inf_max / fitted_person_count; solver gradient is subject-scaled, formal gate is total unscaled"
    c["convergence"]["qualification_rule"] = "fixed two starts are fully retained; local objective differences are reported; qualification selects the lowest penalized objective among starts satisfying optimizer, total gradient, information, and probability gates; no global optimum claim"
    c["optimizer"]["continuation"] = {"fixed_preliminary_strengths": [8.0, 4.0, 2.0], "target": "lambda1 primary; fixed lambda0.5 and lambda2 sensitivity", "strategy": "same fixed initial vector through decreasing ridge strengths strictly greater than final target, then final target; both independent qualification starts follow identical fixed schedule", "limitation": "numerical continuation is a heuristic; does not prove global optimum; original two-start and mathematical gates retained", "source": "https://proceedings.mlr.press/v48/hazanb16.html; general continuation principle only, not a theorem for this CTMC"}
    c["continuous_scaling"] = "unique complete-case persons, unweighted mean and population SD (ddof=0), exactly as CHARLS v8"
    c["optimizer"]["refinement"] = {"method": "explicit Newton numerical polishing of the same penalized objective", "max_steps": 4, "hessian_step": 1e-4, "line_search_halvings": 20, "roundoff_objective_tolerance": 1e-9, "acceptance": "positive definite Hessian; objective nonincrease within roundoff AND lower total gradient; no parameter/model/seed/gate changes"}
    c["parameterization"] = "original 9 x 25 = 225 transition-specific parameters; same feature positions as CHARLS v8"
    c["estimator"] = "fixed Gaussian ridge penalized weighted panel likelihood; not ordinary maximum likelihood"
    c["penalty"] = {"definition": "0.5*sum precision*(theta-center)^2 on TOTAL weighted negative loglik before subject scaling",
                     "intercept_center": float(np.log(0.1)), "intercept_precision": 0.25, "slope_center": 0.0, "slope_precision": 1.0,
                     "justification": "same fixed v8 shrinkage assumptions; CLHLS age bands below 65 are unsupported but retained",
                     "sensitivity_strengths": [0.5, 2.0], "primary_strength": 1.0}
    c["references"] = [{"doi": "10.1016/j.csda.2020.107057", "use": "penalized panel multistate likelihood foundation"},
                        {"doi": "10.1214/08-AOAS191", "use": "finite coefficient regularization principle"},
                        {"url": "https://chjackson.github.io/msm/reference/msm.html", "use": "intermittent panel probabilities and absorbing death"}]
    c["eligibility_design"] = "mathematics + fixed-person convergence + formal-fit numerical qualification; no rule relaxed"
    return c


PENALTY_STRENGTH = 1.0


def penalty_terms(theta):
    p = theta.size // 9
    center = np.zeros_like(theta); center[::p] = np.log(0.1)
    precision = np.ones_like(theta) * PENALTY_STRENGTH; precision[::p] *= 0.25
    diff = theta - center
    return float(0.5 * np.dot(precision * diff, diff)), precision * diff, precision


def _read_raw_covariates():
    cols = pd.read_csv(RAW, nrows=0).columns.tolist()
    disease = [f"g15{chr(97+i)}1" for i in range(22)]
    memory = ["c21a", "c21b", "c21c", "c31a", "c31b", "c31c", "c31d", "c31e", "c41a", "c41b", "c41c"]
    psych = [f"b2{i}" for i in range(1, 8)]
    use = [c for c in ["id", "a1", "residenc", "f1", "f41", "d71", "d81", "g101", "g1021"] + disease + memory + psych if c in cols]
    miss = sorted(set(["id", "a1", "residenc", "f1", "f41", "d71", "d81", "g101", "g1021"] + disease + memory + psych) - set(use))
    if miss:
        raise RuntimeError("Required audited CLHLS baseline variables missing: " + ",".join(miss))
    raw = pd.read_csv(RAW, usecols=use, dtype=str, keep_default_na=False)
    raw = raw.rename(columns={"id": "person_id"}); raw["person_id"] = raw.person_id.astype(str)
    out = pd.DataFrame({"person_id": raw.person_id})
    out["female"] = num(raw.a1).map({1: 0, 2: 1})
    out["urban"] = num(raw.residenc).map({1: 1, 2: 1, 3: 0})
    edu = num(raw.f1).where(~num(raw.f1).isin([88, 99]))
    out["education_primary"] = edu.between(1, 5).astype(float).where(edu.notna())
    out["education_middle"] = edu.between(6, 8).astype(float).where(edu.notna())
    out["education_high"] = (edu >= 9).astype(float).where(edu.notna())
    out["partnered"] = num(raw.f41).isin([1, 2]).astype(float).where(num(raw.f41).notna() & ~num(raw.f41).isin([8, 9]))
    out["smoking"] = num(raw.d71).map({1: 1, 2: 0})
    out["drinking"] = num(raw.d81).map({1: 1, 2: 0})
    wt = num(raw.g101).where(~num(raw.g101).isin([888, 999])); ht = num(raw.g1021).where(~num(raw.g1021).isin([888, 999]))
    bmi = wt / (ht / 100.0) ** 2
    out["bmi"] = bmi.where(bmi.between(10, 80))
    dmat = pd.DataFrame({c: num(raw[c]).where(num(raw[c]).isin([1, 2])) for c in disease})
    out["disease_count"] = dmat.eq(1).sum(axis=1).astype(float).where(dmat.notna().sum(axis=1) >= int(np.ceil(0.8 * len(disease))))
    mmat = pd.DataFrame({c: num(raw[c]).where(num(raw[c]).isin([0, 1])) for c in memory})
    out["memory"] = mmat.sum(axis=1).astype(float).where(mmat.notna().all(axis=1))
    pmat = pd.DataFrame({c: num(raw[c]).where(num(raw[c]).isin([1, 2, 3, 4, 5])) for c in psych})
    positive = ["b21", "b22", "b25", "b27"]; negative = ["b23", "b24", "b26"]
    score = sum((pmat[c] - 1) if c in positive else (5 - pmat[c]) for c in psych)
    out["cesd9"] = score.astype(float).where(pmat.notna().all(axis=1))
    if out.person_id.duplicated().any():
        raise RuntimeError("CLHLS raw baseline covariate export has duplicate person_id")
    return out


def _build_model_inputs():
    long = pd.read_csv(LONG, dtype={"person_id": str})
    long.person_id = long.person_id.astype(str)
    trans = pd.read_csv(ANALYSIS / "transition_intervals_audited.csv", dtype={"person_id": str})
    trans = trans[(trans.cohort == "CLHLS") & (trans.baseline_frame == FRAME)].copy()
    raw = _read_raw_covariates()
    base = long[(long.baseline_frame == FRAME) & (long.wave == FRAME)].copy()
    base = base.sort_values(["person_id"]).drop_duplicates("person_id")
    base = base[["person_id", "age_years", "weight_raw"]].merge(raw, on="person_id", how="left", validate="one_to_one")
    base = base.rename(columns={"age_years": "baseline_age", "weight_raw": "subject_weight_raw"})
    # covariates are baseline-fixed; age/exposure are interval-start variables.
    start = long[["person_id", "wave", "age_years", "sleep_hours_valid_physical", "activity_binary", "exposure_eligible"]].copy()
    start = start.rename(columns={"wave": "wave_start", "age_years": "age_enter", "sleep_hours_valid_physical": "sleep", "activity_binary": "activity"})
    d = trans.merge(start, on=["person_id", "wave_start"], how="left", validate="one_to_one").merge(base, on="person_id", how="left", validate="many_to_one")
    d["i"] = d.state_start.map(STATE); d["j"] = d.state_end.map(STATE)
    d["age_enter"] = num(d.age_enter); d["age_exit"] = d.age_enter + num(d.interval_years)
    d["model_eligible"] = (d.i.isin([0, 1, 2]) & d.j.isin([0, 1, 2, 3]) & num(d.interval_years).gt(0) &
                            d.age_enter.notna() & num(d.sleep).between(0, 24) & d.activity.isin([0, 1]) &
                            num(d.baseline_age).notna() & num(d.subject_weight_raw).gt(0))
    for c in ["female", "urban", "education_primary", "education_middle", "education_high", "partnered", "smoking", "drinking", "bmi", "disease_count", "memory", "cesd9"]:
        d["model_eligible"] &= num(d[c]).notna()
    d.to_csv(OUT / f"clhls_frame{FRAME}_model_input_all_intervals_v2.csv", index=False, encoding="utf-8-sig")
    fit = d[d.model_eligible].copy()
    if fit.empty:
        raise RuntimeError("No CLHLS complete-case model intervals")
    people = fit.person_id.unique(); wmean = num(fit.drop_duplicates("person_id").subject_weight_raw).mean()
    fit["weight"] = num(fit.subject_weight_raw) / wmean
    scaling = {}
    for c in ["baseline_age", "bmi", "disease_count", "memory", "cesd9"]:
        values = num(fit.drop_duplicates("person_id")[c])
        mean, sd = float(values.mean()), float(values.std(ddof=0))
        if not sd > 0:
            raise ValueError("Constant continuous covariate: " + c)
        scaling[c] = {"mean": mean, "sd": sd}
        fit[c + "_c"] = (num(fit[c]) - mean) / sd
    sp = spline(fit.sleep.to_numpy(float))
    for k in range(3): fit[f"sleep_ns{k+1}"] = sp[:, k]
    for k in range(3): fit[f"sleep_ns{k+1}_activity"] = sp[:, k] * fit.activity.to_numpy(float)
    fit["baseline_age_c"] = fit["baseline_age_c"].astype(float)
    fit.to_csv(OUT / "clhls_model_ready_v2.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"frame": FRAME, "all_intervals": len(d), "all_persons": d.person_id.nunique(), "eligible_intervals": len(fit),
                   "eligible_persons": fit.person_id.nunique(), "death_to_death_all_intervals": int(((d.i == 3) & (d.j == 3)).sum()),
                   "excluded_missing_or_invalid": int((~d.model_eligible).sum()), "sleep_knots": SLEEP_KNOTS}]).to_csv(OUT / "model_input_flow_v2.csv", index=False)
    write(OUT / "model_input_assembly_v2.json", {"status": "PASS", "cohort": "CLHLS", "baseline_frame": FRAME,
          "all_intervals": len(d), "eligible_intervals": len(fit), "all_persons": int(d.person_id.nunique()), "eligible_persons": int(fit.person_id.nunique()),
          "death_to_death_retained": int(((d.i == 3) & (d.j == 3)).sum()), "covariates": FULL, "raw_covariate_source": str(RAW),
          "scaling": scaling, "psychological_variable_note": "cesd9_c column is a CLHLS 7-item psychological burden proxy; no claim of equivalence to CHARLS CESD-9"})


def prepare():
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    _build_model_inputs()
    pd.DataFrame([{"analysis_variable": c, "definition": config()["baseline_recode"].get(c, "v8 feature position") } for c in ["female","urban","education_primary","education_middle","education_high","partnered","smoking","drinking","bmi","disease_count","memory","cesd9"]]).to_csv(OUT / "selected_source_dictionary_v2.csv", index=False, encoding="utf-8-sig")
    write(OUT / "run_definition_v2.json", config())
    write(OUT / "environment_v2.json", environment())
    write(OUT / "authorized_amendment_v8.json", {"authority": "user requested CLHLS replication under CHARLS v8 protocol",
          "previous_charls_checksum": "3bdd91b2a49aa409f671c39d929edfcf96903c6d6533283152e73ffceecffc14",
          "v8g_qualification_definition": "user-authorized outcome-oriented qualification redesign: fixed two-start local solutions retained; lowest admissible penalized objective selected; no global convergence claim; v8c-v8f failures retained",
          "stageA_reeaudited": False, "sample_change": "CLHLS frame-specific complete-case derivation; raw source unchanged",
          "adaptations": ["frame-specific knots", "CLHLS 7-item psychological burden proxy in the pre-specified health covariate position",
                          "CLHLS raw measured weight/height BMI and baseline covariates"], "primary_estimator": "fixed Gaussian ridge penalized weighted panel CTMC"})
    (OUT / "run_definition_v2.md").write_text("# CLHLS v8 replication execution definition\n\n" + json.dumps(config(), ensure_ascii=False, indent=2), encoding="utf-8")


def spline(values):
    k1, k2, k3, k4 = SLEEP_KNOTS
    x = np.r_[np.asarray(values, float), SLEEP_REFERENCE]
    a = np.asarray(dmatrix(f"cr(x, knots=({k2},{k3}), lower_bound={k1}, upper_bound={k4})-1", {"x": x}), float)
    return (a[:-1] - a[-1])[:, :3]

@njit(cache=False)
def matrix_exp(a):
    """Higham's Pade13, fixed theta13=5.371920351148152."""
    n = a.shape[0]
    norm = np.max(np.sum(np.abs(a),axis=0))
    scale = max(0,int(np.ceil(np.log2(norm/5.371920351148152)))) if norm>5.371920351148152 else 0
    z = a/(2.**scale)
    eye = np.eye(n)
    z2 = z@z
    z4 = z2@z2
    z6 = z4@z2
    b = np.array([64764752532480000.,32382376266240000.,7771770303897600.,1187353796428800.,129060195264000.,10559470521600.,670442572800.,33522128640.,1323241920.,40840800.,960960.,16380.,182.,1.])
    u = z@(z6@(b[13]*z6+b[11]*z4+b[9]*z2)+b[7]*z6+b[5]*z4+b[3]*z2+b[1]*eye)
    v = z6@(b[12]*z6+b[10]*z4+b[8]*z2)+b[6]*z6+b[4]*z4+b[2]*z2+b[0]*eye
    result = np.linalg.solve(v-u,v+u)
    for _ in range(scale):
        result = result@result
    return result

def arrays(df, block='full'):
    names = ['intercept']+[f'age_band_{int(a)}' for a in AGE_CUTS]+EXPOSURE+{'core':DEMO,'behavior':BEHAVIOR,'full':FULL}[block]
    # Age dummies will be replaced separately for each segment.
    x = np.column_stack([np.ones(len(df)),np.zeros((len(df),4)),df[names[5:]].to_numpy(float)])
    segments = np.zeros((len(df),5))
    bands = np.zeros((len(df),5),dtype=np.int64)
    for n,(start,end) in enumerate(zip(df.age_enter,df.age_exit)):
        breaks = [float(start)]+[a for a in AGE_CUTS if start<a<end]+[float(end)]
        for k,(a,b) in enumerate(zip(breaks[:-1],breaks[1:])):
            segments[n,k] = b-a
            bands[n,k] = np.searchsorted(np.asarray(AGE_CUTS), (a+b)/2,side='right')
    return np.ascontiguousarray(x),segments,bands,df.i.to_numpy(np.int64),df.j.to_numpy(np.int64),df.weight.to_numpy(float),names

@njit(cache=False)
def legacy_data_evaluate(theta,x,dt,bands,si,sj,weights):
    beta = theta.reshape((9,x.shape[1]))
    value = 0.
    scores = np.zeros((len(x),9,x.shape[1]))
    minp,maxp,rowsum,minall,maxall = 1.,0.,0.,1.,0.
    for n in range(len(x)):
        ns = np.sum(dt[n]>0)
        xs = np.zeros((ns,x.shape[1]))
        aa = np.zeros((ns,4,4))
        pp = np.zeros((ns,4,4))
        qq = np.zeros((ns,9))
        prefix = np.zeros((ns+1,4,4))
        suffix = np.zeros((ns+1,4,4))
        prefix[0] = np.eye(4)
        suffix[ns] = np.eye(4)
        for s in range(ns):
            xs[s] = x[n]
            if bands[n,s]>0:
                xs[s,bands[n,s]] = 1.
            qq[s] = np.exp(beta@xs[s])
            if not np.all(np.isfinite(qq[s])):
                raise ValueError('Nonfinite unclipped transition intensity')
            for k in range(9):
                r,j = TRANS[k]
                aa[s,r,j] = qq[s,k]*dt[n,s]
                aa[s,r,r] -= qq[s,k]*dt[n,s]
            pp[s] = matrix_exp(aa[s])
            prefix[s+1] = prefix[s]@pp[s]
        for s in range(ns-1,-1,-1):
            suffix[s] = pp[s]@suffix[s+1]
        pmat = prefix[ns]
        p = pmat[si[n],sj[n]]
        if not np.isfinite(p) or p<=0:
            raise ValueError('Nonpositive/nonfinite observed panel probability')
        minp,maxp = min(minp,p),max(maxp,p)
        rowsum = max(rowsum,np.max(np.abs(np.sum(pmat,axis=1)-1)))
        minall,maxall = min(minall,np.min(pmat)),max(maxall,np.max(pmat))
        value -= weights[n]*np.log(p)
        for s in range(ns):
            e = np.outer(prefix[s,si[n],:],suffix[s+1,:,sj[n]])
            block = np.zeros((8,8))
            block[:4,:4] = aa[s].T
            block[4:,4:] = aa[s].T
            block[:4,4:] = e
            frechet = matrix_exp(block)[:4,4:]
            for k in range(9):
                r,j = TRANS[k]
                de = -weights[n]/p*qq[s,k]*dt[n,s]*(frechet[r,j]-frechet[r,r])
                scores[n,k] += de*xs[s]
    return value,np.sum(scores,axis=0).ravel(),scores.reshape((len(x),theta.size)),np.array([minp,maxp,rowsum,minall,maxall])

def data_evaluate(theta,x,dt,bands,si,sj,weights):
    # Authorized log-likelihood adapter; exp(logp) is for diagnostics only.
    value,gradient,scores,logp,observed,_ = _stable_log_ctmc.data_evaluate(
        theta,x,dt,bands,si,sj,weights,True)
    probability = np.exp(logp)
    diag = np.array([np.exp(np.min(observed)),np.exp(np.max(observed)),
                     np.max(np.abs(np.sum(probability,axis=2)-1.)),
                     np.min(probability),np.max(probability)])
    if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
        raise ValueError('Nonfinite stable log-likelihood or gradient')
    if diag[2]>1e-9 or diag[3]<-1e-9 or diag[4]>1.+1e-9:
        raise ValueError('Stable log-CTMC probability diagnostics failed')
    return value,gradient,scores,diag


def evaluate(theta,x,dt,bands,si,sj,weights):
    value,grad,scores,diag = data_evaluate(theta,x,dt,bands,si,sj,weights)
    penalty,pg,_ = penalty_terms(theta)
    # scores are deliberately DATA-ONLY for the subject-cluster sandwich meat.
    return value+penalty,grad+pg,scores,diag


def initial(df, p, perturb=False):
    b = np.zeros((9,p))
    for k,(r,j) in enumerate(TRANS):
        n = int(((df.i==r)&(df.j==j)).sum())
        denom = float(df.loc[df.i==r,'interval_years'].sum())
        if n==0 or denom<=0:
            raise ValueError(f'No empirical start support for endpoint {r}->{j}; cannot invent start')
        b[k,0] = np.log(n/denom)
        if perturb:
            b[k,1:] = .05*(2*((k+np.arange(p-1))%2)-1)
    return b.ravel()

def scipy_reference(th,args):
    x,dt,bands,si,sj,w,_ = args
    beta = th.reshape(9,-1)
    out = 0.
    for n in range(len(x)):
        p = np.eye(4)
        for s in np.flatnonzero(dt[n]>0):
            v = x[n].copy()
            if bands[n,s]>0:
                v[bands[n,s]] = 1.
            rates = np.exp(beta@v)
            q = np.zeros((4,4))
            for rate,(r,j) in zip(rates,TRANS):
                q[r,j] = rate
                q[r,r] -= rate
            p = p@expm(q*dt[n,s])
        out -= w[n]*np.log(p[si[n],sj[n]])
    return out + penalty_terms(th)[0]

def validate():
    df = pd.read_csv(OUT/'clhls_model_ready_v2.csv', dtype={'person_id': str})
    groups = [df.head(12)]
    for a in AGE_CUTS:
        groups += [df[(df.age_enter < a) & (df.age_exit > a)].head(3), df[df.age_enter >= a].head(2)]
    probe = pd.concat(groups).drop_duplicates(['person_id', 'wave_start']).reset_index(drop=True)
    args = arrays(probe)
    th = initial(df, args[0].shape[1], True)
    v, g, _, pdg = evaluate(th, *args[:-1])
    ref = scipy_reference(th, args)
    h = 1e-5; fd = np.zeros_like(th)
    for k in range(len(th)):
        hi, lo = th.copy(), th.copy(); hi[k] += h; lo[k] -= h
        fd[k] = (evaluate(hi, *args[:-1])[0] - evaluate(lo, *args[:-1])[0]) / (2*h)
    relative = float(np.max(np.abs(g-fd)/(1+np.abs(fd))))
    exp_error, frechet_error = 0., 0.
    for mult in [.01, 1., 10., 100.]:
        q = np.zeros((4,4))
        for k,(r,j) in enumerate(TRANS):
            q[r,j] = mult*(k+1)*.025; q[r,r] -= q[r,j]
        e = np.outer(np.array([.2,.3,.5,0.]), np.array([.4,.1,.3,.2]))
        block = np.zeros((8,8)); block[:4,:4] = q.T; block[4:,4:] = q.T; block[:4,4:] = e
        exp_error = max(exp_error, float(np.max(np.abs(matrix_exp(q)-expm(q)))))
        frechet_error = max(frechet_error, float(np.max(np.abs(matrix_exp(block)[:4,4:]-expm_frechet(q.T,e,compute_expm=False)))))
    ranks = {}
    for block_name in ['core','behavior','full']:
        a = arrays(df, block_name); actual = []
        for n in range(len(df)):
            for ss in np.flatnonzero(a[1][n] > 0):
                xx = a[0][n].copy()
                if a[2][n,ss] > 0: xx[a[2][n,ss]] = 1.
                actual.append(xx)
        actual = np.asarray(actual)
        ranks[block_name] = {'columns': actual.shape[1], 'rank': int(np.linalg.matrix_rank(actual)), 'parameters': 9*actual.shape[1],
                             'segment_count': len(actual), 'age_band_segment_counts': np.bincount(a[2][a[1] > 0], minlength=5).tolist()}
    checks = {'scipy_likelihood_agreement': abs(v-ref) <= 1e-9, 'exponential_agreement': exp_error <= 1e-10,
              'frechet_agreement': frechet_error <= 1e-9, 'all_parameter_gradient_agreement': relative <= 2e-6,
              'deterministic_repeat': bool(v == evaluate(th,*args[:-1])[0]), 'actual_cross_age_intervals_tested': bool(np.any((args[1] > 0).sum(axis=1) > 1)),
              'matrix_probability_legal': bool(pdg[2] <= 1e-9 and pdg[3] >= -1e-9 and pdg[4] <= 1+1e-9)}
    report = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': {k: bool(vv) for k,vv in checks.items()},
              'n_real_test_intervals': len(probe), 'parameter_count': len(th), 'weighted_penalized_nll': v,
              'gradient_relative_error': relative, 'expm_max_error': exp_error, 'frechet_max_error': frechet_error,
              'design': ranks, 'age_band_support_limitation': 'CLHLS baseline ages are 65+; <55 and 55-64 attained-age bands have zero observed support and remain ridge-regularized',
              'probability_diagnostics': pdg.tolist()}
    write(OUT/'mathematical_validation_v8.json', report)
    if report['status'] != 'PASS': raise RuntimeError('STOP: CLHLS mathematics validation failed')

def lock():
    val = json.loads((OUT/'mathematical_validation_v8.json').read_text(encoding='utf-8'))
    if val['status']!='PASS':raise RuntimeError('STOP: mathematics gate is not PASS')
    local = [f'clhls_frame{FRAME}_model_input_all_intervals_v2.csv','clhls_model_ready_v2.csv','model_input_assembly_v2.json','run_definition_v2.json','run_definition_v2.md','environment_v2.json','mathematical_validation_v8.json','authorized_amendment_v8.json']
    paths = INPUTS+[SCRIPT]+[OUT/n for n in local]
    payload = {'execution_id':f'CLHLS_{FRAME}_PANEL_CTM_PENALIZED_v8g_20261011','status':'LOCKED_CONVERGENCE_PENDING','created_utc':utc(),
               'hash_basis':'canonical UTF8 JSON payload; sorted keys; compact separators; per-file raw bytes SHA256; manifest/checksum/logs excluded to avoid self reference',
               'configuration':config(),'files':[{'path':str(p),'size_bytes':p.stat().st_size,'sha256':sha(p)} for p in paths]}
    digest = hashlib.sha256(canonical(payload)).hexdigest()
    write(OUT/'execution_manifest_v2.json',{'payload':payload,'execution_checksum':digest})
    (OUT/'execution_manifest_v2.sha256').write_text(digest+'  execution_manifest_v2.json\n',encoding='utf-8')
    print('LOCKED execution checksum: '+digest,flush=True)

def verify():
    m = json.loads((OUT/'execution_manifest_v2.json').read_text(encoding='utf-8'))
    if hashlib.sha256(canonical(m['payload'])).hexdigest()!=m['execution_checksum']:
        raise RuntimeError('STOP: manifest checksum mismatch')
    for f in m['payload']['files']:
        p = Path(f['path'])
        if p.stat().st_size!=f['size_bytes'] or sha(p)!=f['sha256']:
            raise RuntimeError('STOP: locked file changed: '+str(p))
    if environment()!=json.loads((OUT/'environment_v2.json').read_text(encoding='utf-8')):
        raise RuntimeError('STOP: current runtime differs from locked environment')
    if config()!=m['payload']['configuration']:raise RuntimeError('STOP: configuration changed')
    return m

def fit(df,block,start,prefix):
    global PENALTY_STRENGTH
    args = arrays(df,block)
    nsub = df.person_id.nunique()
    logpath = OUT/(prefix+'_iterations.jsonl')
    if logpath.exists():raise FileExistsError(logpath)
    t0 = time.perf_counter()
    started = utc()
    log = logpath.open('x',encoding='utf-8')
    last = {}
    def fun(th):
        val,grad,_,diag = evaluate(th,*args[:-1])
        last.update(value=val,grad_inf=float(np.max(np.abs(grad))),diag=diag.tolist())
        return val/nsub,grad/nsub
    it = 0
    def callback(th):
        nonlocal it
        it += 1
        rec = {'iteration':it,'seconds':time.perf_counter()-t0,**last}
        log.write(json.dumps(rec,allow_nan=False)+'\n');log.flush()
        if it==1 or it%25==0:print(prefix,json.dumps(rec),flush=True)
    cfg = config()['optimizer']
    solver_options = {k:cfg[k] for k in ['maxiter','maxls','maxcor','ftol']}
    solver_options['gtol'] = 0.5*config()['convergence']['total_gradient_inf_max']/nsub
    try:
        target_strength = PENALTY_STRENGTH
        continuation = []
        for preliminary_strength in cfg['continuation']['fixed_preliminary_strengths']:
            if preliminary_strength <= target_strength:
                continue
            PENALTY_STRENGTH = preliminary_strength
            preliminary = minimize(fun,start,method='L-BFGS-B',jac=True,callback=callback,
                                   options=solver_options)
            pv,pg,_,_ = evaluate(preliminary.x,*args[:-1])
            entry = {'continuation_strength':preliminary_strength,'target_strength':target_strength,
                     'objective':pv,'gradient_inf_total':float(np.max(np.abs(pg))),
                     'optimizer_success':bool(preliminary.success),'optimizer_message':str(preliminary.message),
                     'nit':int(preliminary.nit),'nfev':int(preliminary.nfev),'seconds':time.perf_counter()-t0}
            continuation.append(entry);log.write(json.dumps(entry,allow_nan=False)+'\n');log.flush()
            print(prefix+' continuation '+json.dumps(entry),flush=True)
            if not preliminary.success:
                raise RuntimeError('STOP: fixed continuation numerical stage failed')
            start = preliminary.x.copy()
        PENALTY_STRENGTH = target_strength
        result = minimize(fun,start,method='L-BFGS-B',jac=True,callback=callback,
                          options=solver_options)
        lbfgs_gradient = float(np.max(np.abs(evaluate(result.x,*args[:-1])[1])))
        polishing = []
        refine = cfg['refinement']
        for step_id in range(refine['max_steps']):
            value0, gradient0, _, _ = evaluate(result.x,*args[:-1])
            norm0 = float(np.max(np.abs(gradient0)))
            if norm0 <= config()['convergence']['total_gradient_inf_max']:
                break
            hess = np.empty((len(result.x),len(result.x)))
            for k in range(len(result.x)):
                dh = refine['hessian_step']*max(1.,abs(result.x[k]))
                hi,lo = result.x.copy(),result.x.copy(); hi[k]+=dh; lo[k]-=dh
                hess[:,k] = (evaluate(hi,*args[:-1])[1]-evaluate(lo,*args[:-1])[1])/(2*dh)
            hess = (hess+hess.T)/2
            mineig = float(np.linalg.eigvalsh(hess)[0])
            if mineig <= 0:
                raise RuntimeError('STOP: numerical polishing Hessian not positive definite')
            direction = np.linalg.solve(hess,gradient0)
            accepted = False
            for half in range(refine['line_search_halvings']+1):
                alpha = 2.**(-half)
                candidate = result.x-alpha*direction
                value1,gradient1,_,_ = evaluate(candidate,*args[:-1])
                norm1 = float(np.max(np.abs(gradient1)))
                if value1 <= value0+refine['roundoff_objective_tolerance'] and norm1 < norm0:
                    result.x = candidate
                    accepted = True
                    break
            entry = {'numerical_polishing_step':step_id+1,'accepted':accepted,'gradient_before':norm0,
                     'gradient_after':norm1,'objective_before':value0,'objective_after':value1,
                     'alpha':alpha,'minimum_hessian_eigenvalue':mineig,'seconds':time.perf_counter()-t0}
            polishing.append(entry); log.write(json.dumps(entry,allow_nan=False)+'\n');log.flush()
            print(prefix+' polishing '+json.dumps(entry),flush=True)
            if not accepted:
                raise RuntimeError('STOP: explicit numerical polishing line search failed')
        val,grad,scores,diag = evaluate(result.x,*args[:-1])
        record = {'run_id':prefix,'block':block,'status':'OPTIMIZER_COMPLETED','started_utc':started,'completed_utc':utc(),
                  'lbfgs_gradient_inf_total':lbfgs_gradient,'numerical_polishing':polishing,'continuation':continuation,
                  'n_intervals':len(df),'n_persons':int(nsub),'parameter_count':len(result.x),'penalized_objective':val,'nll':float(data_evaluate(result.x,*args[:-1])[0]),'penalty':penalty_terms(result.x)[0],'penalty_strength':PENALTY_STRENGTH,
                  'gradient_inf_total':float(np.max(np.abs(grad))),'data_score_inf':float(np.max(np.abs(data_evaluate(result.x,*args[:-1])[1]))),'optimizer_success':bool(result.success),
                  'optimizer_message':str(result.message),'nit':int(result.nit),'nfev':int(result.nfev),'seconds':time.perf_counter()-t0,
                  'probability_diagnostics':diag.tolist(),'theta':result.x.tolist(),'parameter_names':args[-1]}
        write(OUT/(prefix+'_fit.json'),record)
        print(prefix+' completed '+json.dumps({k:v for k,v in record.items() if k not in ['theta','parameter_names']},ensure_ascii=False),flush=True)
        return record,result.x,args,scores
    finally:
        log.close()

def information(theta,args,df,prefix):
    h = config()['numeric']['hessian_step']
    size = len(theta)
    hess = np.empty((size,size))
    print(prefix+' observed-information calculation '+str(size)+' columns',flush=True)
    for k in range(size):
        step = h*max(1.,abs(theta[k]))
        hi,lo = theta.copy(),theta.copy();hi[k]+=step;lo[k]-=step
        hess[:,k] = (evaluate(hi,*args[:-1])[1]-evaluate(lo,*args[:-1])[1])/(2*step)
        if (k+1)%50==0:print(prefix+' information columns '+str(k+1)+'/'+str(size),flush=True)
    asym = float(np.max(np.abs(hess-hess.T)))
    hess = (hess+hess.T)/2
    eigen = np.linalg.eigvalsh(hess)
    data_info = hess-np.diag(penalty_terms(theta)[2])
    data_eigen = np.linalg.eigvalsh(data_info)
    np.save(OUT/(prefix+'_data_information.npy'),data_info)
    result = {'minimum_eigenvalue':float(eigen[0]),'maximum_eigenvalue':float(eigen[-1]),'max_unsymmetrized_asymmetry':asym,
              'condition_number':float(np.linalg.cond(hess)), 'step_relative':h, 'penalty_strength':PENALTY_STRENGTH, 'data_information_min_eigenvalue':float(data_eigen[0]), 'data_information_condition_number':float(np.linalg.cond(data_info)), 'information_basis':'penalized information; data-only matrix separately retained'}
    result['qualified'] = bool(eigen[0]>0 and result['condition_number']<=1e10)
    np.save(OUT/(prefix+'_observed_information.npy'),hess)
    if result['qualified']:
        _,_,scores,_ = evaluate(theta,*args[:-1])
        grouped = pd.DataFrame(scores).groupby(df.person_id.to_numpy(),sort=True).sum().to_numpy()
        bread = np.linalg.inv(hess)
        sandwich = bread@(grouped.T@grouped)@bread.T
        np.save(OUT/(prefix+'_cluster_sandwich.npy'),sandwich)
        result['cluster_count'] = len(grouped)
        result['effective_parameter_dimension'] = float(np.trace(bread@data_info))
        result['regularization_information_fraction'] = float(np.trace(bread@np.diag(penalty_terms(theta)[2]))/size)
        result['covariance_interpretation'] = config()['variance']
        result['sandwich_min_diagonal'] = float(np.diag(sandwich).min())
        result['sandwich_status'] = 'COMPUTED'
    else:
        result['sandwich_status'] = 'NOT RUN: information gate failed; no pseudoinverse'
    write(OUT/(prefix+'_information.json'),result)
    print(prefix+' information '+json.dumps(result),flush=True)
    return result

def convergence():
    m = verify()
    if (OUT/'convergence_v2.json').exists():raise FileExistsError('Immutable convergence report already exists')
    df = pd.read_csv(OUT/'clhls_model_ready_v2.csv',dtype={'person_id':str})
    people = np.sort(df.person_id.unique())
    selected = np.random.default_rng(SEED).choice(people,size=min(2000,len(people)),replace=False)
    d = df[df.person_id.isin(selected)].reset_index(drop=True)
    d.to_csv(OUT/'convergence_person_sample_v2.csv',index=False,encoding='utf-8-sig')
    a = arrays(d)
    counts = {f'{int(i)}->{int(j)}':int(n) for (i,j),n in d.groupby(['i','j']).size().items()}
    support = all(counts.get(f'{r}->{j}',0)>0 for r,j in TRANS)
    if not support:
        write(OUT/'convergence_v2.json',{'status':'FAIL','execution_checksum':m['execution_checksum'],'reason':'fixed person sample has absent endpoint event support','counts':counts,'formal_fit':'NOT RUN'})
        raise RuntimeError('STOP: fixed small-sample event support failed')
    start0 = initial(d,a[0].shape[1]);start1 = initial(d,a[0].shape[1],True)
    v1,g1,_,_ = evaluate(start0,*a[:-1]);v2,g2,_,_ = evaluate(start0,*a[:-1])
    repeat = abs(v1-v2)<=1e-12 and np.max(np.abs(g1-g2))<=1e-12
    fits = [fit(d,'full',s,f'convergence_start{k}_v2') for k,s in enumerate([start0,start1])]
    infos = [information(f[1],f[2],d,f'convergence_start{k}_v2') for k,f in enumerate(fits)]
    delta = float(np.max(np.abs(fits[0][1]-fits[1][1])))
    ld = abs(fits[0][0]['penalized_objective']-fits[1][0]['penalized_objective'])
    admissible=[k for k,(fit_tuple,inf) in enumerate(zip(fits,infos)) if fit_tuple[0]['optimizer_success'] and fit_tuple[0]['gradient_inf_total']<=config()['convergence']['total_gradient_inf_max'] and inf['qualified'] and fit_tuple[0]['probability_diagnostics'][2]<=1e-9 and fit_tuple[0]['probability_diagnostics'][3]>=-1e-9 and fit_tuple[0]['probability_diagnostics'][4]<=1+1e-9]
    if not admissible:
        raise RuntimeError('STOP: no admissible local solution among fixed qualification starts')
    selected=min(admissible,key=lambda k:fits[k][0]['penalized_objective'])
    np.save(OUT/'convergence_selected_theta_v2.npy',fits[selected][1])
    checks = {'all_nine_endpoint_events':support,'exact_repeat':bool(repeat),'both_optimizers_success':all(f[0]['optimizer_success'] for f in fits),
              'all_starts_gradient_gate':all(f[0]['gradient_inf_total']<=config()['convergence']['total_gradient_inf_max'] for f in fits),
              'selected_local_solution_admissible':selected in admissible,
              'both_information_qualified':all(i['qualified'] for i in infos),
              'probability_legal':all(f[0]['probability_diagnostics'][2]<=1e-9 and f[0]['probability_diagnostics'][3]>=-1e-9 and f[0]['probability_diagnostics'][4]<=1+1e-9 for f in fits)}
    report = {'status':'PASS' if all(checks.values()) else 'FAIL','execution_checksum':m['execution_checksum'],
              'seed':SEED,'sample_intervals':len(d),'sample_persons':int(d.person_id.nunique()),'parameter_count':len(start0),
              'counts':counts,'checks':checks,'parameter_delta':delta,'loglik_delta':ld,'information':infos,
              'admissible_starts':admissible,'selected_start':selected,'qualification_rule':config()['convergence']['qualification_rule'],
              'formal_fit':'NOT RUN','completed_utc':utc()}
    write(OUT/'convergence_v2.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    if not all(checks.values()):raise RuntimeError('STOP: complete-parameter convergence gate failed; formal run prohibited')

def main():
    m = verify()
    c = json.loads((OUT/'convergence_v2.json').read_text(encoding='utf-8'))
    if c['status']!='PASS' or c['execution_checksum']!=m['execution_checksum']:
        raise RuntimeError('STOP: convergence gate not PASS for current checksum')
    df = pd.read_csv(OUT/'clhls_model_ready_v2.csv',dtype={'person_id':str})
    for block in ['core','behavior','full']:
        a = arrays(df,block)
        if block == 'full' and (OUT/'convergence_selected_theta_v2.npy').exists():
            starting = np.load(OUT/'convergence_selected_theta_v2.npy')
        else:
            starting = initial(df,a[0].shape[1])
        record,theta,args,_ = fit(df,block,starting,f'formal_{block}_v2')
        inf = information(theta,args,df,f'formal_{block}_v2')
        if not record['optimizer_success'] or record['gradient_inf_total']>config()['convergence']['total_gradient_inf_max'] or not inf['qualified'] or not (record['probability_diagnostics'][2]<=1e-9 and record['probability_diagnostics'][3]>=-1e-9 and record['probability_diagnostics'][4]<=1+1e-9):
            write(OUT/(f'formal_{block}_status_v2.json'),{'status':'FAIL','execution_checksum':m['execution_checksum'],'fit':record,'information':inf})
            raise RuntimeError('STOP: formal '+block+' model qualification failed')
        se = np.sqrt(np.diag(np.load(OUT/(f'formal_{block}_v2_cluster_sandwich.npy'))))
        rows = []
        for k,(r,j) in enumerate(TRANS):
            for z,name in enumerate(args[-1]):
                index = k*len(args[-1])+z
                rows.append({'transition':f'{r}->{j}','parameter':name,'estimate':theta[index],'penalized_cluster_se':se[index],'estimator':'fixed_ridge_panel_likelihood',
                             'ci_lower':theta[index]-1.96*se[index],'ci_upper':theta[index]+1.96*se[index],'execution_checksum':m['execution_checksum']})
        pd.DataFrame(rows).to_csv(OUT/(f'formal_{block}_parameters_v2.csv'),index=False,encoding='utf-8-sig')
        write(OUT/(f'formal_{block}_status_v2.json'),{'status':'PASS','execution_checksum':m['execution_checksum'],'fit':record,'information':inf})
    write(OUT/'formal_clhls_completed_v2.json',{'status':'PASS','execution_checksum':m['execution_checksum'],'completed_utc':utc()})

def sensitivity():
    global PENALTY_STRENGTH
    m = verify()
    if json.loads((OUT/'formal_clhls_completed_v2.json').read_text(encoding='utf-8'))['status']!='PASS':
        raise RuntimeError('Formal primary fit must PASS before penalty sensitivity')
    df = pd.read_csv(OUT/'clhls_model_ready_v2.csv',dtype={'person_id':str})
    args = arrays(df,'full')
    primary = json.loads((OUT/'formal_full_v2_fit.json').read_text(encoding='utf-8'))
    results = []
    for strength in config()['penalty']['sensitivity_strengths']:
        PENALTY_STRENGTH = strength
        prefix = 'sensitivity_lambda_'+str(strength).replace('.','p')
        f,th,a,_ = fit(df,'full',initial(df,args[0].shape[1]),prefix)
        inf = information(th,a,df,prefix)
        status = 'PASS' if f['optimizer_success'] and f['gradient_inf_total']<=config()['convergence']['total_gradient_inf_max'] and inf['qualified'] and f['probability_diagnostics'][2]<=1e-9 else 'FAIL'
        delta = float(np.max(np.abs(th-np.asarray(primary['theta']))))
        result = {'status':status,'penalty_strength':strength,'fit':f,'information':inf,'max_parameter_delta_from_primary':delta,'execution_checksum':m['execution_checksum']}
        write(OUT/(prefix+'_status.json'),result)
        results.append({k:v for k,v in result.items() if k not in ['fit','information']})
        if status!='PASS':raise RuntimeError('Prespecified penalty sensitivity failed: '+str(strength))
    PENALTY_STRENGTH = 1.
    write(OUT/'penalty_sensitivity_completed_v8.json',{'status':'PASS','results':results,'primary_selection':'fixed lambda1; never chosen for significance','execution_checksum':m['execution_checksum']})


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('phase',choices=['prepare','validate','lock','convergence','main','sensitivity'])
    ap.add_argument('frame', nargs='?', choices=['2008','2011','2014'])
    options = ap.parse_args()
    try:
        globals()[options.phase]()
    except Exception as e:
        trace = traceback.format_exc()
        print(trace,file=sys.stderr,flush=True)
        if OUT.exists():
            failure = OUT/(options.phase+'_failure_'+datetime.now().strftime('%Y%m%dT%H%M%S')+'.json')
            write(failure,{'status':'FAIL','phase':options.phase,'type':type(e).__name__,'error':str(e),'traceback':trace,'utc':utc(),'formal_fit':'NOT RUN' if options.phase!='main' else 'FAILED'})
        sys.exit(1)



