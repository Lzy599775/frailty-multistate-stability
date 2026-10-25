"""CHARLS panel CTMC: explicit covariates, attained age, subject weights.

New implementation; never writes Stage A or v1 evidence. CLI phases:
prepare, validate, lock, convergence, main. All formal inputs are hash checked.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import io
import json
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

ROOT = Path(r'F:\A_CHARLS+CLHLS\00_新选题')
BASE = ROOT / '08_intermediate/analysis_ready_stageA_v1_20261010'
ANALYSIS = BASE / 'analysis_ready_final_v1'
OUT = BASE / 'stageB_charls_penalized_execution_v8'
SCRIPT = Path(__file__).resolve()
STATE = {'robust': 0, 'prefrail': 1, 'frail': 2, 'death': 3}
TRANS = np.array([(r, j) for r in range(3) for j in range(4) if r != j], dtype=np.int64)
SEED = 20261011
AGE_CUTS = [55., 65., 75., 85.]
DEMO = ['baseline_age_c', 'female', 'urban', 'education_primary', 'education_middle', 'education_high', 'partnered']
BEHAVIOR = DEMO + ['smoking', 'drinking']
FULL = BEHAVIOR + ['bmi_c', 'disease_count_c', 'memory_c', 'cesd9_c']
EXPOSURE = ['sleep_ns1', 'sleep_ns2', 'sleep_ns3', 'activity', 'sleep_ns1_activity', 'sleep_ns2_activity', 'sleep_ns3_activity']
FEATURES = ['intercept'] + [f'age_band_{int(a)}' for a in AGE_CUTS] + EXPOSURE + FULL
INPUTS = [ANALYSIS/'analysis_ready_manifest_v1.json',
          BASE/'stageB_model_qualification/stageB_formal_execution_manifest_draft_v1.json',
          ANALYSIS/'charls_analysis_ready_long.csv', ANALYSIS/'transition_intervals_audited.csv',
          BASE/'longitudinal_sources/charls_2011_source_linked.csv',
          BASE/'longitudinal_sources/charls_2011_FI_candidate_raw.csv',
          BASE/'charls_sources/2011_psu_raw.csv',
          BASE/'charls_sources/harmonized_H_CHARLS_D_Data_raw.csv',
          BASE/'charls_sources/charls_dictionary_complete.csv']

def utc():
    return datetime.now(timezone.utc).isoformat()

def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()

def _raise_json(o):
    raise TypeError(f'Unsupported JSON type: {type(o).__name__}')

def write(path, obj):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Immutable evidence already exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=lambda o: o.item() if isinstance(o, np.generic) else (_raise_json(o))), encoding='utf-8')

def read(path, usecols=None):
    return pd.read_csv(path, dtype=str, keep_default_na=False, usecols=usecols)

def num(s):
    return pd.to_numeric(s, errors='coerce')

def valid(s, codes):
    a = num(s)
    return a.where(a.isin(codes))

def environment():
    buf = io.StringIO()
    with redirect_stdout(buf):
        np.show_config()
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'patsy', 'numba', 'llvmlite']},
            'numpy_blas_configuration': buf.getvalue(), 'jit': 'numba njit; fastmath=False; parallel=False; cache=False',
            'hardware_fit_claim': 'NOT MADE'}

def original_config():
    return {'mode': 'EMPIRICAL', 'seed': SEED, 'states': STATE,
            'transition_order': TRANS.tolist(), 'age_cuts': AGE_CUTS,
            'time': 'attained age at interval start; exit=start+frozen interval_years; coarse survey-wave years, not exact dates',
            'age_structure': 'transition-specific intercept plus attained-age bands <55,55-64,65-74,75-84,85+; baseline age adjusted separately',
            'sleep_knots': [3., 6., 7., 9.], 'sleep_reference': 7.,
            'sleep_basis': 'patsy natural cubic cardinal basis, subtract basis at 7h, retain first 3 of 4 columns; exactly same full spline space with intercept',
            'features': FEATURES, 'blocks': {'core': DEMO, 'behavior': BEHAVIOR, 'full': FULL},
            'sample': 'same full complete-case intervals and subjects for all three blocks; no imputation; healthy-start and self intervals retained',
            'baseline_recode': {'sex': '1 male=0,2 female=1', 'residence': 'health__communityID joins official 2011 PSU urban_nbs; 0 rural,1 urban; not hukou or interviewer ID',
                'education': 'valid 1-3 below primary(reference),4 primary,5 middle,6-10 high+',
                'marital': 'valid 1/3 partnered=1,4/5/7/8 other=0',
                'smoking_drinking': 'valid 0/1 only', 'bmi': 'numeric harmonized r1mbmi; Stata special missing remains missing',
                'disease': 'all 14 da007_ codes valid 1/2; sum yes(1), otherwise missing',
                'memory': 'harmonized r1tr20 in [0,20]',
                'cesd9': 'dc009-018 excluding sleep dc015; all 9 valid1-4; negative raw-1; positive dc013/dc016 4-raw; sum0-27; not CESD10'},
            'health_conditioning': 'baseline diseases overlap baseline FI by prespecified adjustment; not time-updated FI components; memory and CESD9 not among 26 FI items; estimates conditional observational associations',
            'weight': '2011 weight__ind_weight_ad2 >0; fixed subject weight across all intervals, divide by mean among unique full-complete fitted subjects; L=sum_i w_i sum_j log(P_ij)',
            'weight_estimand': 'baseline individual-weighted complete-case conditional association; not nationally unbiased survival effect, not IPCW, not product of weights',
            'variance': 'observed information from central differences of exact analytic score; subject-cluster sandwich H^-1 sum_i (w_i score_i)(w_i score_i)^T H^-1; no inverse/pseudoinverse fallback',
            'optimizer': {'method': 'L-BFGS-B', 'bounds': None, 'maxiter': 1500, 'maxls': 40, 'maxcor': 30, 'ftol': 1e-14, 'gtol': 1e-8, 'objective_scaling': 'divide weighted negative loglik and score by number of unique fitted subjects; report total score'},
            'convergence': {'sample_n_persons': 2000, 'selection': 'sorted unique fitted IDs; RNG20261011 choice without replacement; retain all their eligible intervals; no event balancing/resampling',
                'starts': ['event count/person-years, same intercept in all age bands, all slopes0', 'same intercepts, alternating deterministic +/-0.05 slopes'],
                'total_gradient_inf_max': 1e-4, 'between_start_parameter_delta_max': 1e-3, 'between_start_loglik_delta_max': 1e-6,
                'information_min_eigenvalue': 0., 'information_condition_max': 1e10, 'probability_tolerance': 1e-9,
                'exact_repeat': 'identical start0 objective and gradient evaluated twice; max difference<=1e-12',
                'all_nine_endpoint_events_required': True},
            'numeric': {'rate': 'exp(eta) without clipping or coefficient bounds', 'matrix_exp': 'Higham Pade13 scaling/squaring; validated against scipy; no fastmath',
                'gradient': 'adjoint Frechet derivative via block exponential, full matrix-product chain rule across age cuts',
                'gradient_finite_difference_tolerance': 2e-6, 'hessian_step': 1e-4},
            'formal_order': ['core', 'behavior', 'full'], 'primary_reporting_block': 'full', 'result_claim': 'observational association, not causal effect'}

def config():
    c = original_config()
    c['execution_version'] = 'v8_penalized_full_transition_specific'
    c['parameterization'] = 'original 9 x 25 = 225 transition-specific parameters; all covariates and age bands retained; no sharing across improvement/worsening/death'
    c['baseline_recode']['bmi'] = 'harmonized r1mbmi numeric within [10,80]; outside missing; same fixed 5720-person model table as v2b'
    c['estimator'] = 'fixed Gaussian ridge penalized weighted panel likelihood; not ordinary maximum likelihood; no significance-based tuning'
    c['penalty'] = {'definition':'0.5*sum precision*(theta-center)^2 on TOTAL weighted negative loglik, before dividing by unique-subject count',
                    'intercept_center': float(np.log(.1)), 'intercept_precision': .25,
                    'slope_center':0., 'slope_precision':1.,
                    'justification':'baseline annual rates weakly anchored at 0.1; log-rate baseline SD2; coefficient SD1. Fixed regularization is a new assumption, not inferred from convergence or p-values.',
                    'sensitivity_strengths':[.5,2.],
                    'primary_strength':1.}
    c['convergence']['score'] = 'penalized estimating score; data-only score separately retained and need not be zero'
    c['convergence']['information'] = 'penalized Hessian must be positive definite and condition<=1e10; unpenalized Hessian reported separately; penalization does not prove data-only identifiability'
    c['optimizer'].update(maxiter=2000,maxcor=50,gtol=1e-10,ftol=1e-15)
    c['variance'] = 'Hpen^-1 sum_i (weighted data-only score_i)(weighted data-only score_i)^T Hpen^-1; no penalty in meat; approximate sampling variance of shrinkage estimator. Not unpenalized-MLE confidence intervals, not full Bayesian credible intervals.'
    c['references'] = [{'doi':'10.1016/j.csda.2020.107057','url':'https://discovery.ucl.ac.uk/id/eprint/10107405/','use':'penalized panel multistate likelihood foundation; their spline smoothing/automatic tuning is NOT copied as our fixed ridge'},
                       {'doi':'10.1214/08-AOAS191','url':'https://doi.org/10.1214/08-AOAS191','use':'finite coefficient regularization for sparse likelihood; their Cauchy logistic prior is NOT claimed as our Gaussian CTMC prior'},
                       {'url':'https://chjackson.github.io/msm/reference/msm.html','use':'subject weights, intermittent panel probabilities and constraints'}]
    c['eligibility_design'] = 'mathematics+fixed-person convergence+formal-fit numerical qualification; no rule relaxed to obtain significance'
    return c

PENALTY_STRENGTH = 1.

def penalty_terms(theta):
    p = theta.size//9
    center = np.zeros_like(theta)
    center[::p] = np.log(.1)
    precision = np.ones_like(theta)*PENALTY_STRENGTH
    precision[::p] *= .25
    diff = theta-center
    return float(.5*np.dot(precision*diff,diff)), precision*diff, precision


V8_DEFINITION = '# CHARLS v8 惩罚估计执行定义（2026-10-11）\n\n授权：用户明确允许调整参数化或收敛资格设计。所有 v2/v3-v7 失败证据保留，原始数据与阶段 A manifest 不变。\n\n纠正诊断：v7b 基线设计秩为5/5，并非编码别名。系数[-29.48,28.97,1.45,27.97,-0.195]将多条直接强度推向约1e-13/年；两个起点概率/似然一致但系数继续沿边界漂移。因此不能从接近零信息特征值断言年龄共线或矩阵编码错误。\n\n恢复原9条转移各25参数，合计225；截距、4年龄段哑变量、3睡眠自然样条、活动、3交互、7人口学、2行为、4完整健康协变量。年龄边界55/65/75/85与基线年龄项原样保留。跨年龄段用矩阵乘积积分未知状态，死亡吸收；睡眠节点3/6/7/9、参考7h；无强度clip、无参数bounds、无插补。\n\n数据：直接复用 v2b 的已冻结完整案例建模表，5720人/5720条区间，其中2011起始61条、2018起始5659条。因此此表主要支撑2018-2020面板转移与2011健康条件调整，不应宣称完整2011-2020多波动态效应。这里不重开阶段A审计，也不填造缺失波次。\n\n估计量：最小化 -sum_i w_i log L_i + 0.5 sum_j tau_j(theta_j-m_j)^2。个人2011基线正权重均值归一化并固定跨区间。截距 m=log(0.1/年), tau=0.25 (SD2); 其余坐标 m=0, tau=1 (SD1)。惩罚施加于总目标后再数值除以个人数，避免样本大小隐式改变lambda。lambda=1主分析；0.5与2作为预定敏感性，不能挑最显著或最收敛的版本当主结果。Gaussian penalty 有明确收缩假设，不是临床已知先验；参考文献的惩罚框架不等于使用其相同penalty。\n\n资格：原梯度<=1e-4、两起点参数差<=1e-3、目标差<=1e-6、概率合法<=1e-9、信息正定与条件数<=1e10不放宽；梯度/信息对应新的惩罚目标。原始数据似然、原始分数和原始Hessian另报，不因惩罚曲率通过就声称数据本身充分识别。种子20261011、人级2000人固定，不重抽。数学测试包括data+penalty解析梯度与独立SciPy对照。\n\n方差：Hpen^-1 Bdata Hpen^-1，Bdata由个人加权DATA分数外积组成，惩罚不加入meat。报告为收缩估计的近似抽样SE/区间；不是无惩罚MLE的置信区间，也不是MCMC Bayesian可信区间。原始信息最小特征值、惩罚信息比例与有效维度必须报告。惩罚依赖大或敏感性差异明显的转移不能宣称稳健剂量效应。\n\n方法来源：原5篇Nature系列参考文献仍提供衰弱面板、睡眠剂量和联合暴露框架；额外数值方法参考10.1016/j.csda.2020.107057的惩罚面板多状态模型，以及10.1214/08-AOAS191的稀疏似然正则化原理。前者为样条平滑、后者为Cauchy logistic prior，均不能声称等同于此Gaussian ridge。官方msm定义：https://chjackson.github.io/msm/reference/msm.html 。\n\nchecksum：canonical UTF8 JSON载荷含脚本、运行配置、源输入、阶段A manifest、复用建模表、环境及数学验证各文件原始SHA与字节数；不含自身与可变运行日志；运行状态另存。formal入口仅由此checksum的收敛PASS放行。\n'

def prepare():
    import shutil
    prior = BASE/'stageB_charls_execution_v2b'
    if OUT.exists():raise FileExistsError(OUT)
    old = json.loads((prior/'execution_manifest_v2.json').read_text(encoding='utf-8'))
    if hashlib.sha256(canonical(old['payload'])).hexdigest()!=old['execution_checksum']:
        raise RuntimeError('Old frozen input manifest digest mismatch')
    reused = ['charls_model_input_all_intervals_v2.csv','charls_model_ready_v2.csv','model_input_assembly_v2.json','selected_source_dictionary_v2.csv']
    expected = {f['path']:f for f in old['payload']['files']}
    for name in reused:
        source = prior/name
        f = expected[str(source)]
        if source.stat().st_size!=f['size_bytes'] or sha(source)!=f['sha256']:
            raise RuntimeError('Reused locked model input changed: '+str(source))
    OUT.mkdir(parents=True)
    for name in reused:shutil.copy2(prior/name,OUT/name)
    write(OUT/'run_definition_v2.json',config())
    write(OUT/'environment_v2.json',environment())
    (OUT/'run_definition_v2.md').write_text(V8_DEFINITION,encoding='utf-8')
    write(OUT/'authorized_amendment_v8.json',{'authority':'user explicitly authorizes parameterization/qualification adjustment',
          'previous_manifest':str(prior/'execution_manifest_v2.json'),'reused_model_files':reused,
          'sample_change':False,'stageA_reaudited':False,'primary_estimator':'fixed Gaussian ridge penalized weighted panel CTMC',
          'prior_attempts_not_promoted':['v3','v3b','v4','v5','v5b','v5c','v5d','v6','v7','v7b'],
          'correction':'Earlier age-confounding/reference-group alias diagnoses not established; v7b baseline design is full rank, fitted rates approach boundary.'})
    print('REUSED original v2b model table: 5720 persons/intervals. No Stage A changes.',flush=True)


def spline(values):
    x = np.r_[np.asarray(values,float),7.]
    a = np.asarray(dmatrix('cr(x, knots=(6.,7.), lower_bound=3., upper_bound=9.)-1',{'x':x}),float)
    return (a[:-1]-a[-1])[:,:3]

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
def data_evaluate(theta,x,dt,bands,si,sj,weights):
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
    df = pd.read_csv(OUT/'charls_model_ready_v2.csv',dtype={'person_id':str})
    # Deterministically include all attained-age bands and actual crossings for mathematical tests only.
    groups = [df.head(12)]
    for a in AGE_CUTS:
        groups += [df[(df.age_enter<a)&(df.age_exit>a)].head(3), df[df.age_enter>=a].head(2)]
    probe = pd.concat(groups).drop_duplicates(['person_id','wave_start']).copy()
    args = arrays(probe)
    # Initialization for validation uses whole data endpoint counts, not invented numerical outcomes.
    th = initial(df,args[0].shape[1],True)
    t0 = time.perf_counter()
    v,g,_,pdg = evaluate(th,*args[:-1])
    elapsed = time.perf_counter()-t0
    reference = scipy_reference(th,args)
    fd = np.zeros_like(th)
    h = 1e-5
    for k in range(len(th)):
        hi,lo = th.copy(),th.copy()
        hi[k] += h
        lo[k] -= h
        fd[k] = (evaluate(hi,*args[:-1])[0]-evaluate(lo,*args[:-1])[0])/(2*h)
    relative = float(np.max(np.abs(g-fd)/(1+np.abs(fd))))
    # Validate exponential and adjoint Frechet over prespecified rate multipliers.
    exp_error,frechet_error = 0.,0.
    for mult in [.01,1.,10.,100.]:
        q = np.zeros((4,4))
        for k,(r,j) in enumerate(TRANS):
            q[r,j] = mult*(k+1)*.025
            q[r,r] -= q[r,j]
        e = np.outer(np.array([.2,.3,.5,0.]),np.array([.4,.1,.3,.2]))
        block = np.zeros((8,8));block[:4,:4]=q.T;block[4:,4:]=q.T;block[:4,4:]=e
        exp_error = max(exp_error,float(np.max(np.abs(matrix_exp(q)-expm(q)))))
        frechet_error = max(frechet_error,float(np.max(np.abs(matrix_exp(block)[:4,4:]-expm_frechet(q.T,e,compute_expm=False)))))
    ranks = {}
    for block in ['core','behavior','full']:
        a = arrays(df,block)
        actual = []
        for n in range(len(df)):
            for s in np.flatnonzero(a[1][n]>0):
                xx = a[0][n].copy()
                if a[2][n,s]>0:xx[a[2][n,s]]=1
                actual.append(xx)
        actual = np.asarray(actual)
        ranks[block] = {'columns':actual.shape[1],'rank':int(np.linalg.matrix_rank(actual)),'parameters':9*actual.shape[1],
                        'segment_count':len(actual),'age_band_segment_counts':np.bincount(a[2][a[1]>0],minlength=5).tolist()}
    raw = np.asarray(dmatrix('cr(x, knots=(6.,7.), lower_bound=3., upper_bound=9.)-1',{'x':np.r_[df.sleep.to_numpy(float),7.]}),float)
    ref = raw[-1];raw=raw[:-1]-ref
    basis_error = float(np.max(np.abs(raw[:,3]+raw[:,:3].sum(axis=1))))
    repeat = evaluate(th,*args[:-1])
    checks = {'scipy_likelihood_agreement':abs(v-reference)<=1e-9,'exponential_agreement':exp_error<=1e-10,
              'frechet_agreement':frechet_error<=1e-9,'all_parameter_gradient_agreement':relative<=2e-6,
              'spline_space_preserved':basis_error<=1e-10,'three_blocks_full_rank':all(a['columns']==a['rank'] for a in ranks.values()),
              'deterministic_repeat':bool(v==repeat[0] and np.array_equal(g,repeat[1])),'actual_cross_age_intervals_tested':bool(np.any((args[1]>0).sum(axis=1)>1)),
              'matrix_probability_legal':bool(pdg[2]<=1e-9 and pdg[3]>=-1e-9 and pdg[4]<=1+1e-9)}
    checks = {k:bool(v) for k,v in checks.items()}
    report = {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'n_real_test_intervals':len(probe),
              'parameter_count':len(th),'weighted_nll':v,'scipy_reference_nll':reference,'gradient_relative_error':relative,
              'expm_max_error':exp_error,'frechet_max_error':frechet_error,'spline_equivalence_error':basis_error,
              'design':ranks,'first_compilation_and_evaluation_seconds':elapsed,'probability_diagnostics':pdg.tolist()}
    write(OUT/'mathematical_validation_v8.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    if not all(checks.values()):raise RuntimeError('STOP: mathematics/design validation failed')

def lock():
    val = json.loads((OUT/'mathematical_validation_v8.json').read_text(encoding='utf-8'))
    if val['status']!='PASS':raise RuntimeError('STOP: mathematics gate is not PASS')
    local = ['charls_model_input_all_intervals_v2.csv','charls_model_ready_v2.csv','model_input_assembly_v2.json','run_definition_v2.json','run_definition_v2.md','environment_v2.json','selected_source_dictionary_v2.csv','mathematical_validation_v8.json','authorized_amendment_v8.json']
    paths = INPUTS+[SCRIPT]+[OUT/n for n in local]
    payload = {'execution_id':'CHARLS_2011_PANEL_CTM_PENALIZED_v8_20261011','status':'LOCKED_CONVERGENCE_PENDING','created_utc':utc(),
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
    args = arrays(df,block)
    nsub = df.person_id.nunique()
    logpath = OUT/(prefix+'_iterations.jsonl')
    if logpath.exists():raise FileExistsError(logpath)
    t0 = time.perf_counter()
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
    try:
        result = minimize(fun,start,method='L-BFGS-B',jac=True,callback=callback,
                          options={k:cfg[k] for k in ['maxiter','maxls','maxcor','ftol','gtol']})
        val,grad,scores,diag = evaluate(result.x,*args[:-1])
        record = {'run_id':prefix,'block':block,'status':'OPTIMIZER_COMPLETED','started_utc':utc(),
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
    df = pd.read_csv(OUT/'charls_model_ready_v2.csv',dtype={'person_id':str})
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
    checks = {'all_nine_endpoint_events':support,'exact_repeat':bool(repeat),'both_optimizers_success':all(f[0]['optimizer_success'] for f in fits),
              'both_total_gradient_le_1e4':all(f[0]['gradient_inf_total']<=1e-4 for f in fits),
              'between_start_parameter_delta_le_1e3':delta<=1e-3,'between_start_loglik_delta_le_1e6':ld<=1e-6,
              'both_information_qualified':all(i['qualified'] for i in infos),
              'probability_legal':all(f[0]['probability_diagnostics'][2]<=1e-9 and f[0]['probability_diagnostics'][3]>=-1e-9 and f[0]['probability_diagnostics'][4]<=1+1e-9 for f in fits)}
    report = {'status':'PASS' if all(checks.values()) else 'FAIL','execution_checksum':m['execution_checksum'],
              'seed':SEED,'sample_intervals':len(d),'sample_persons':int(d.person_id.nunique()),'parameter_count':len(start0),
              'counts':counts,'checks':checks,'parameter_delta':delta,'loglik_delta':ld,'information':infos,
              'formal_fit':'NOT RUN','completed_utc':utc()}
    write(OUT/'convergence_v2.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    if not all(checks.values()):raise RuntimeError('STOP: complete-parameter convergence gate failed; formal run prohibited')

def main():
    m = verify()
    c = json.loads((OUT/'convergence_v2.json').read_text(encoding='utf-8'))
    if c['status']!='PASS' or c['execution_checksum']!=m['execution_checksum']:
        raise RuntimeError('STOP: convergence gate not PASS for current checksum')
    df = pd.read_csv(OUT/'charls_model_ready_v2.csv',dtype={'person_id':str})
    for block in ['core','behavior','full']:
        a = arrays(df,block)
        record,theta,args,_ = fit(df,block,initial(df,a[0].shape[1]),f'formal_{block}_v2')
        inf = information(theta,args,df,f'formal_{block}_v2')
        if not record['optimizer_success'] or record['gradient_inf_total']>1e-4 or not inf['qualified'] or not (record['probability_diagnostics'][2]<=1e-9 and record['probability_diagnostics'][3]>=-1e-9 and record['probability_diagnostics'][4]<=1+1e-9):
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
    write(OUT/'formal_charls_completed_v2.json',{'status':'PASS','execution_checksum':m['execution_checksum'],'completed_utc':utc()})

def sensitivity():
    global PENALTY_STRENGTH
    m = verify()
    if json.loads((OUT/'formal_charls_completed_v2.json').read_text(encoding='utf-8'))['status']!='PASS':
        raise RuntimeError('Formal primary fit must PASS before penalty sensitivity')
    df = pd.read_csv(OUT/'charls_model_ready_v2.csv',dtype={'person_id':str})
    args = arrays(df,'full')
    primary = json.loads((OUT/'formal_full_v2_fit.json').read_text(encoding='utf-8'))
    results = []
    for strength in config()['penalty']['sensitivity_strengths']:
        PENALTY_STRENGTH = strength
        prefix = 'sensitivity_lambda_'+str(strength).replace('.','p')
        f,th,a,_ = fit(df,'full',initial(df,args[0].shape[1]),prefix)
        inf = information(th,a,df,prefix)
        status = 'PASS' if f['optimizer_success'] and f['gradient_inf_total']<=1e-4 and inf['qualified'] and f['probability_diagnostics'][2]<=1e-9 else 'FAIL'
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



