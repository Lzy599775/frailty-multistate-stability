"""Freeze all design/start/selection rules before any v8 formal optimization."""
import ast,shutil,sys
import numpy as np
import study_v8 as s

def main():
    if (s.OUT/'batch_registration.json').exists():raise FileExistsError('already registered')
    integrity=[]
    for r in s.js(s.G4/'v7_2_full_delivery_manifest.json')['files']:
        p=s.G4/r['path'];assert p.stat().st_size==r['bytes'] and s.sha(p)==r['sha256']
        integrity.append(dict(path=str(p),bytes=r['bytes'],sha256=r['sha256']))
    # Verify the original files bound at the end of G4; do not repeat its arithmetic.
    import pandas as pd
    for r in pd.read_csv(s.G4/'final_original_input_integrity_checks.csv').to_dict('records'):
        p=s.Path(r['path']);assert p.stat().st_size==r['bytes'] and s.sha(p)==r['sha256'];integrity.append(r)
    for n in ['v7_2_G4_submission_gate.md','v7_2_G5_read_only_gate.md','g5_required_manuscript_edits.md','v7_2_unresolved_scientific_questions.md']:(s.G4/n).read_text(encoding='utf-8')
    for n in ['JCM_manuscript.docx','JCM_Supplementary_Material.docx','JCM_v8_one_pass_resolution_for_Codex.md']:
        p=s.SUB/n;integrity.append(dict(path=str(p),bytes=p.stat().st_size,sha256=s.sha(p)))
    shutil.copyfile(s.REC/'stable_log_ctmc.py',s.OUT/'stable_log_ctmc.py');assert s.sha(s.OUT/'stable_log_ctmc.py')==s.sha(s.REC/'stable_log_ctmc.py')
    env=s.old.environment();previous=s.js(s.REC/'environment.json')
    for k in ['python','executable','platform']:assert env[k]==previous[k]
    for k,v in previous['packages'].items():assert env['packages'][k]==v
    s.write(s.OUT/'environment.json',env)
    text='''# Compact model and bounded scientific adjudication: frozen before fitting

POST-HOC RE-SPECIFICATION after v7.2, not an originally prespecified analysis or independent validation.

Clinical target, samples, four states, nine ordered transitions, FI, weights, death absorbing and two-year mortality estimands remain fixed. Frozen v7/G4 inputs are read-only. Original data are not recoded. All v8 files are new.

Compact model: each of nine paths has independent intercept, activity, z=(sleep−7)/2, z² and activity×z coefficients:45. All13 original baseline nuisance covariates retained, with their coefficients shared only within predefined recovery(1→0,2→0,2→1), deterioration(0→1,0→2,1→2), death(0/1/2→3) categories. No p-value deletion. Attained-age bands share the same category structure: CHARLS uses supported55/65/75/85 cut-point indicators (reference<55); CLHLS reference65–74 and supported75/85 indicators, removes unsupported<55 and55–64 effects. Frozen attained-age segment durations stay unchanged; no continuous-time age approximation added. Baseline age remains the frozen standardized adjustment variable; all design ranks and support are checked before fit. No medical covariate is excluded. Dimensions: CHARLS96; CLHLS90. This stronger shared-adjustment assumption is explicit, not automatically more correct.

Coefficient mapping is linear into9×22 path designs. The exact validated stable_log_ctmc data probability/derivative engine is byte-copied unchanged. Score chain rule Mᵀg, Hessian differentiation in free coordinates, subject-cluster data meat. Each free coefficient penalized once, not once per duplicated expanded coordinate. Intercept centerlog(.1), precision.25; all free slopes center0 precision1. Primaryλ1; sensitivity.5/2; no lambda choice by significance. Sleep scaling2h fixed before fitting.

A: original225 model,2014 only, direct fixed-targetλ (no continuation, to test same-objective basins). Everyλ evaluates frozen solution, optimizes two otherλ frozen warm starts plus2 independently generated starts. Seeds2026102401/2026102402, RNG normal slopesSD.15 and intercepts log(.1)+N(0,.5); identical seeds/rule for eachλ. Exactly12 new attempts. Frozen solution participates as immutable diagnostic candidate. Select lowest sameλ penalized objective among numerically qualified candidates only; never by TIR sign. Preserve every attempt, including failures and any new lower-objective candidate. No automatic expansion to other frames without direct conflicting sameλ evidence.

B: same threeλ, four frames;2 fixed independent starts per frame/λ using the same seeds/rule:24 full-sample attempts. No further model candidate or lucky restarts. Shared-design matrices and full analytic gradient verified before optimization. Original/compact rank, constant/unsupported columns and observed endpoint support recorded. Observed2→0 endpoints are not hidden CTMC jump counts.

Optimizer/qualification unchanged: L-BFGS-B unbounded maxiter2000,maxls40,maxcor50,ftol1e−15; subject-scaled gtol=.5×1e−4/Npeople. Total unscaled gradient≤1e−4; optimizer success required. Existing≤4 Newton polishing, centered Hessian step1e−4×max(1,|θ|), positive Hessian,≤20 halvings, objective nonincrease tolerance1e−9 AND improved gradient. No fallback/clipping. Information positive definite and condition≤1e10; probability tolerance1e−9. Every accepted fit saves full gradients, Hessian/data-Hessian and matched sandwich. Trial exception preserves true theta/input and stops that independent attempt; other registered attempts remain independent, never replace a failure invisibly.

Sameλ solution closeness thresholds retained: parameter maxdifference≤1e−3, objective difference≤1e−6. Larger objective differences label MULTIMODAL_LOCAL_SOLUTIONS. Objective≤1e−6 with keylogTIR difference>1e−3 flags candidate practical nonidentification. These are numerical diagnostic thresholds, not clinical/significance tests. For key quantities report direction, log-range, max/min, CI and predictions. Direction reversal=DIRECTION_SENSITIVE; different sameλ basins=SOLUTION_DEPENDENT. Stable direction with substantial variation reported MAGNITUDE_SENSITIVE, not confirmatory evidence. No invented clinical magnitude cutoff.

Prediction comparison: one fixed internal subject-grouped80/20 hash split: SHA256('20261011|frame|person_id')first16hex mod5==0 held-out. All intervals/person stay together; same split for both models; no overlapping frames called external validation. Train-only mean/SD for5 continuous covariates and mean person baseline weight; transform validation using train values. Fixed original spline knots and compact polynomial unchanged. Each model/frame trains one independent seed2026102401 atλ1 only (8 fits). No model tuning from held-out predictions. Report weighted held-out endpoint logscore, person-cluster pairedSE; deemed clearly unacceptable compact deterioration only if mean compact−original<−.02nats/interval AND upper95% paired normal interval<0. This pragmatic prospective-for-this-batch threshold is not a validated equivalence margin. Failure/unsupported training rank is honestly NOT_ESTIMABLE; no replacement split. Training normalization and test IDs sealed before fit. These additional8fits only assess predictive comparison, never replace full-sample parameters. If a valid comparison cannot be computed, do not claim noninferiority; retain the protocol's qualitative assessment boundary.

Selection: do not compare penalized objectives between models/lambda. Compact can be recommended as a post-hoc candidate only with materially reduced dimension/full free-design support, sameλ qualified start agreement and stable central recovery association directions acrossλ, no clearly unacceptable predictive deterioration. Original may support bounded descriptive conclusions only if sameλ qualified solutions do not give contradictory central recovery associations. Assess all declared central recovery and four mortality contrasts, not select only CI-excluding1/0 results. CHARLS path5vs7 direct recovery contrast need not be confirmatory; calibrated difference testing is not central. If both model families cannot support stable recovery associations, SCIENTIFIC_CLAIM_NOT_SUPPORTED/BLOCK_SUBMISSION; stop after A/B adjudication, no submission-candidate Word/figures, no additional model search. If a scientifically bounded association narrative passes, freeze selected exact model version and propagate all outputs before Word/figures. Do not force manuscripts when science fails.

Expected registered optimization attempts44=12original diagnostics+24compact full-sample+8subject-split prediction. No bootstrap/IPCW/new dataset. Original7PASS and recovered2PASS are not refit as replacements. Old Word remains read-only; authorized manuscript replacements only new version after scientific gate.
'''
    s.md(s.OUT/'compact_model_spec_frozen.md',text)
    rows=[];support=[];splits=[]
    for frame in s.FRAMES:
        df=s.old.sample(frame);stem='charls_model_ready_v2.csv' if frame=='CHARLS' else 'clhls_model_ready_v2.csv';source=s.old.directory(frame)/stem
        integrity.append(dict(path=str(source),bytes=source.stat().st_size,sha256=s.sha(source)))
        train,test,transforms=s.cv_frames(df,frame)
        for part,dd in [('train',train),('test',test)]:
            for pid in sorted(dd.person_id.unique()):splits.append(dict(frame=frame,person_id=pid,partition=part))
        s.write(s.OUT/f'inputs/{frame}_cv_transform.json',transforms)
        for kind in ['original','compact']:
            mat,par,names=s.mapping(frame,kind)
            sd=s.OUT/'inputs';sd.mkdir(exist_ok=True)
            np.save(sd/f'{frame}_{kind}_mapping.npy',mat);s.write(sd/f'{frame}_{kind}_parameters.json',dict(parameters=par,expanded_features=names))
            args=s.arrays(df,frame,kind);design=[];counts=np.zeros(5,int)
            for n in range(len(df)):
                for q in np.flatnonzero(args[1][n]>0):
                    xx=args[0][n].copy();band=args[2][n,q];counts[band]+=1
                    if band>0:xx[band]=1
                    # Parametric support independent of transition endpoint frequency.
                    design.append(xx)
            design=np.asarray(design);free=np.vstack([design@mat[k*len(names):(k+1)*len(names)] for k in range(9)])
            freerank=int(np.linalg.matrix_rank(free))
            support.append(dict(frame=frame,model=kind,n_intervals=len(df),n_persons=df.person_id.nunique(),parameters=len(par),free_design_rank=freerank,expanded_design_rank=int(np.linalg.matrix_rank(design)),expanded_columns=len(names),zero_expanded_columns='|'.join(names[k] for k in np.flatnonzero(np.max(abs(design),axis=0)==0)),age_band_segment_counts=counts.tolist(),observed_frail_robust_pairs=int((df.i.eq(2)&df.j.eq(0)).sum())))
            if kind=='compact':assert freerank==len(par),'STOP compact structural rank deficiency'
            for lam,_ in s.RUNS:
                if kind=='original' and frame!='2014':continue
                if kind=='original':
                    for olam,_ in s.RUNS:
                        if olam==lam:continue
                        th=s.old.model(frame,olam)[1];name=f'warm_lambda{olam:g}';p=sd/f'{kind}_{frame}_lambda{lam:g}_{name}.npy';np.save(p,th)
                        rows.append(dict(model=kind,frame=frame,lambda_value=lam,start=name,start_path=str(p),start_sha256=s.sha(p),partition='full',formal_role='same_lambda_diagnostic'))
                for seed in s.SEEDS:
                    th=s.independent_start(par,seed);p=sd/f'{kind}_{frame}_lambda{lam:g}_seed{seed}.npy';np.save(p,th)
                    rows.append(dict(model=kind,frame=frame,lambda_value=lam,start=f'seed{seed}',start_path=str(p),start_sha256=s.sha(p),partition='full',formal_role='same_lambda_diagnostic' if kind=='original' else 'posthoc_compact_candidate'))
            seed=s.SEEDS[0];p=sd/f'{kind}_{frame}_cv_seed{seed}.npy';np.save(p,s.independent_start(par,seed))
            rows.append(dict(model=kind,frame=frame,lambda_value=1.,start=f'cv_seed{seed}',start_path=str(p),start_sha256=s.sha(p),partition='train',formal_role='heldout_prediction_comparison'))
    # Put original target diagnostics first in A lane, compact in separate B lane.
    s.save(s.OUT/'fit_plan.csv',rows);assert len(rows)==44
    s.save(s.OUT/'inputs/subject_holdout_split.csv',splits);s.save(s.OUT/'inputs/design_support.csv',support)
    s.save(s.OUT/'inputs/frozen_input_integrity.csv',integrity)
    files=[p for p in s.OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='batch_registration.json']
    s.write(s.OUT/'batch_registration.json',dict(utc=s.utc(),status='FROZEN_BEFORE_FORMAL_FIT',spec_sha256=s.sha(s.OUT/'compact_model_spec_frozen.md'),expected_attempts=44,seeds=s.SEEDS,engine_sha256=s.sha(s.OUT/'stable_log_ctmc.py'),files=[dict(path=str(p.relative_to(s.OUT)),bytes=p.stat().st_size,sha256=s.sha(p)) for p in sorted(files)]))
    print('BATCH_FROZEN',len(rows),support,flush=True)

if __name__=='__main__':main()
