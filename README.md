# Supplementary Code

This archive supplies the analytical source code, fixed specifications, environment records and aggregate parameter and covariance evidence for the accompanying manuscript. It covers model construction, stable continuous-time Markov probabilities and analytic scores, penalized fitting, comparisons of fixed starts, conditional covariance, mortality scenarios and internal endpoint prediction. The 178 frozen source and evidence files retain their original bytes; source_manifest.csv records their provenance. Only this README has been updated from the v3 archive.

## Entry points and source lineage

Under project/00_投稿/3-Journal of Clinical Medicine/JCM_v8_one_pass_resolution:

- study_v8.py defines the original and compact models, penalties, registered seeds and participant split.
- stable_log_ctmc.py calculates stable log probabilities and analytic scores.
- fit_runner.py implements fitting, qualification, the fixed 44-attempt batch and prediction-training fits.
- evidence_v8.py and adjudicate_v8.py implement matched covariance, mortality, same-penalty comparisons, curvature and held-out evaluation.
- rebuild_selected_results.py generates selected aggregate results.
- repair_reporting_records.py reconstructs reporting records from saved numerical evidence; it does not refit models.

The audit_v7_2_g4_g5/audit_common.py and controlled_recovery_v7_1 directories supply original-model access and numerical code. The audit_v7_fi_corrected/authorized_repair_20261024 and 07_data_processing directories supply the corrected frailty-index construction and sample lineage. The serious-illness rule in stageA_06_final_recode.py is superseded by the corrected builder. Do not reconstruct CLHLS with that older builder alone.

## Data access and environment

Participant-level source data, model-ready intervals, individual scores and resampling identifiers are excluded. Obtain CHARLS and CLHLS data directly from their providers under the applicable agreements. Restoring the exact frozen fits requires the licensed model-ready inputs referenced by audit_common.sample() under 08_intermediate/analysis_ready_stageA_v1_20261010 and the corrected and controlled model directories. Source-to-intermediate construction also requires the inventories, questionnaires, codebooks and intermediate inputs identified by the scripts.

environment.json in the v8 directory records the analytical environment and exact package versions. static_import_index.csv distinguishes bundled modules from third-party dependencies, which must be obtained under their own licences. No third-party packages or participant-level survey files are distributed here.

Some original scripts contain absolute Windows F: drive paths. In a separate working copy, explicitly map the study and provider-data roots and review the path-only changes before execution. This source archive is not a turnkey reproduction environment. The included aggregate parameter and covariance evidence can be inspected without participant-level data.

## Execution and verification

Do not execute scripts inside the frozen archive. Some scripts create output directories on import, and formal entry points write research artifacts. Inspect the entry points and fit_plan.csv before running in an independent analysis copy. The code supplement does not grant permission to redistribute restricted survey data.

All 30 Python files were statically parsed; the ZIP CRC and every one of the 178 manifest entries were checked against the v3 archive. The editorial preparation did not execute imports, reconstruct participant-level inputs, refit models or test execution on another machine. Those operations have not been independently reproduced in this revision.
