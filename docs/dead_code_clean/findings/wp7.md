# WP7 raw return (gpt-5.6-luna, read-only sweep, 2026-08-01)

## 1. CLUSTER SHAPE

This cluster contains live CI, data-preparation, fixture, annotation, BRIC diagnostic and BST validation tools, plus finished-pass analysis and parity gates. Every listed file is either a live tool or a one-shot gate; no D-unreach finding survived the root-manifest sweep.

## 2. LEDGER

WP7-1 | C | scripts/__init__.py:module | LIVE TOOL: package initialisation is required by documented `scripts.build_shots_master` and test imports. | evidence: scripts/build_shots_master.py:39-40, tests/test_first_last_stroke_buffered_search.py:9; roots: audit_plan.md:25-36 | leave as package support | confidence high

WP7-2 | D-prod | scripts/analyse_first_last_stroke_buffered_search.py:main | ONE-SHOT GATE: post-hoc analysis for the completed buffered-search measurement, retained for report regeneration and tests rather than production execution. | evidence: experiments/annotator/first_last_stroke_buffered_search_20260730/README.md:68-72, tests/test_first_last_stroke_buffered_search.py:9; roots: audit_plan.md:25-36; no production caller | archive candidate with its test evidence | confidence high

WP7-3 | C | scripts/api_fixtures/build_mock_artifacts.py:main | LIVE TOOL: rebuilds API mock artefacts and has a tracked direct invocation. | evidence: scripts/api_fixtures/build_mock_artifacts.py:1-9, docs/architecture_notes/collation_taxon_pin_w_preds_refactor_log.md:46; roots: audit_plan.md:25-36 | leave as fixture tooling | confidence high

WP7-4 | C | scripts/api_fixtures/e2e.py:main | LIVE TOOL: the API fixture handover documents this as the programmatic end-to-end check. | evidence: scripts/api_fixtures/handoff_report.md:181-182, :894; roots: audit_plan.md:25-36 | leave as fixture tooling | confidence high

WP7-5 | C | scripts/api_fixtures/rebuild_real.py:main | LIVE TOOL: the API inference path documents rerunning this helper to rebuild real fixtures. | evidence: src/api/bst_x_inference.py:20-21, :225; scripts/api_fixtures/handoff_report.md:1093, :1374; roots: audit_plan.md:25-36 | leave as fixture tooling | confidence high

WP7-6 | C | scripts/build_extract_stems.py:main | LIVE TOOL: the refactor runbook documents this command for selecting and re-extracting pose stems. | evidence: docs/architecture_notes/collation_taxon_pin_w_preds_refactor.md:405-431, :1210; roots: audit_plan.md:25-36 | leave as data-preparation tooling | confidence high

WP7-7 | C | scripts/build_shots_master.py:main | LIVE TOOL: the repository README and module docstring provide the `python -m scripts.build_shots_master` entry point. | evidence: README.md:76, scripts/build_shots_master.py:39-40; src/annotator/calibration/scoring.py:63; roots: audit_plan.md:25-36 | leave as live data-preparation tooling | confidence high

WP7-8 | C | scripts/estimate_shuttle_oob_rate.py:main | LIVE TOOL: the augmentation documentation names this script as the run method for the shuttle out-of-bounds check. | evidence: docs/architecture_notes/augmentation_framework.md:383-389, scripts/estimate_shuttle_oob_rate.py:11-16; roots: audit_plan.md:25-36 | leave as analysis tooling | confidence high

WP7-9 | D-prod | scripts/plots/bar_chart_overall_shuttleset_comparison.py:main | ONE-SHOT GATE: the script hardcodes completed run IDs and writes the committed headline chart. | evidence: scripts/plots/bar_chart_overall_shuttleset_comparison.py:8-12, :24-27; README.md:7; roots: audit_plan.md:25-36; no production caller | archive candidate after retaining the committed chart | confidence high

WP7-10 | D-prod | scripts/plots/bar_chart_per_class_f1_final.py:main | ONE-SHOT GATE: the script compares two fixed completed runs and writes a presentation artefact. | evidence: scripts/plots/bar_chart_per_class_f1_final.py:1-8, :15-23; roots: audit_plan.md:25-36; no production caller | archive candidate after retaining the committed chart | confidence high

WP7-11 | D-prod | scripts/plots/class_size_vs_val_f1.py:main | ONE-SHOT GATE: the script reads one fixed run and produces a presentation-only class-size chart. | evidence: scripts/plots/class_size_vs_val_f1.py:1-9, :20-24; roots: audit_plan.md:25-36; no production caller | archive candidate with its generated evidence | confidence high

WP7-12 | D-prod | scripts/plots/confusion_matrix.py:main | ONE-SHOT GATE: this is a presentation-polish confusion-matrix script referenced by the shared plotting helper, not a production path. | evidence: src/shared/eval_plots.py:1-7, docs/architecture_notes/collation_taxon_pin_w_preds_refactor_log.md:32; roots: audit_plan.md:25-36; no production caller | archive candidate with the shared helper reference retained | confidence high

WP7-13 | D-prod | scripts/plots/f1_runs_bar_charts.py:module | ONE-SHOT GATE: the script hardcodes historical Series H runs and is retained as a comparison-plot recipe. | evidence: scripts/plots/f1_runs_bar_charts.py:1-16, docs/architecture_notes/focal_alpha_revert_sketch.md:957; roots: audit_plan.md:25-36; no production caller | archive candidate | confidence high

WP7-14 | D-prod | scripts/plots/plot_stage8_tradeoffs.py:main | ONE-SHOT GATE: this renders the completed stage-8 sweep menu from fixed local defaults and is analysis-only. | evidence: scripts/plots/plot_stage8_tradeoffs.py:1-21, :34-41; roots: audit_plan.md:25-36; no production caller | archive candidate | confidence med

WP7-15 | D-prod | scripts/plots/trajectory_chart_macro_and_min.py:main | ONE-SHOT GATE: this is a presentation chart backed by hand-transcribed historical results. | evidence: scripts/plots/trajectory_chart_macro_and_min.py:1-13, :19-20; roots: audit_plan.md:25-36; no production caller | archive candidate with the presentation output | confidence high

WP7-16 | C | scripts/pr_advisory.py:main | LIVE TOOL: CI invokes the advisory check directly. | evidence: .github/workflows/pr-quality.yml:105; roots: audit_plan.md:25-36 | leave as CI tooling | confidence high

WP7-17 | C | scripts/pr_main_files.py:main | LIVE TOOL: CI invokes the main-files check directly and docs describe its output. | evidence: .github/workflows/pr-quality.yml:76, docs/ci.md:23; roots: audit_plan.md:25-36 | leave as CI tooling | confidence high

WP7-18 | C | scripts/rename_videos.py:main | LIVE TOOL: the repository review explicitly retains this for the open video-download task. | evidence: docs/architecture_notes/pre_phase_2_review_2026-04-26.md:109; scripts/rename_videos.py:8-10; roots: audit_plan.md:25-36 | leave as data-preparation tooling | confidence high

WP7-19 | C | scripts/validate_videos.py:main | LIVE TOOL: the repository review explicitly retains this for the open video-download task. | evidence: docs/architecture_notes/pre_phase_2_review_2026-04-26.md:109; scripts/validate_videos.py:6-8; roots: audit_plan.md:25-36 | leave as data-preparation tooling | confidence high

WP7-20 | D-prod | src/bst_x/validation_scripts/calibration_ece.py:main | ONE-SHOT GATE: calibration analysis is pinned to completed run artefacts and is referenced as a finished downstream check. | evidence: src/bst_x/validation_scripts/calibration_ece.py:1-16, src/bst_x/validation_scripts/refactoring/README.md:19-20; roots: audit_plan.md:25-36; no production caller | archive candidate with generated calibration evidence | confidence high

WP7-21 | D-prod | src/bst_x/validation_scripts/collation_fulldiff.py:main | ONE-SHOT GATE: this is a completed refactor parity utility retained as evidence of byte-identical collation. | evidence: src/bst_x/validation_scripts/refactoring/README.md:26-37, docs/architecture_notes/completed_general_refactors/simplification_pass/refactor_worklog.md:376-387; roots: audit_plan.md:25-36; no production caller | archive candidate | confidence high

WP7-22 | D-prod | src/bst_x/validation_scripts/compute_clip_length_stats.py:main | ONE-SHOT GATE: the CSV-only clip-length calculation is cited as the extraction used for the completed EDA notebook. | evidence: notebooks/01_shuttleset_eda_v3.ipynb:85, src/bst_x/validation_scripts/hit_frame_lookup.py:9; roots: audit_plan.md:25-36; no production caller | archive candidate | confidence med

WP7-23 | C | src/bst_x/validation_scripts/fail_rate_per_class.py:main | LIVE TOOL: the validation README documents two current CLI forms for per-class fail-rate analysis. | evidence: src/bst_x/validation_scripts/README.md:114-155, docs/architecture_notes/completed_general_refactors/dir_flatten_refactor.md:100; roots: audit_plan.md:25-36 | leave as reusable validation tooling | confidence high

WP7-24 | D-prod | src/bst_x/validation_scripts/failsafe_bst_mmpose_zeroing_check_equivalence.py:main | ONE-SHOT GATE: the completed MMPose heuristic pass used this byte-identity gate as critical verification infrastructure. | evidence: docs/architecture_notes/mmpose_heuristic/mmpose_heuristic.md:103-165, :394; docs/architecture_notes/pre_phase_2_review_2026-04-26.md:112; roots: audit_plan.md:25-36; no production caller | archive candidate with the parity result | confidence high

WP7-25 | C | src/bst_x/validation_scripts/hit_frame_lookup.py:build_hit_frame_lookup | LIVE TOOL: the README and X3D plan identify this as the existing Method A hit-frame entry point. | evidence: src/bst_x/validation_scripts/README.md:182-194, docs/architecture_notes/x3d_integration_macro_plan/x3d_integration_macro_plan.md:27, :54; roots: audit_plan.md:25-36; callers: find_busted_clips and validate_zeroed_frames | leave as reusable library tooling | confidence high

WP7-26 | C | src/bst_x/validation_scripts/keypoint_lr_interframe_diagnostic.py:main | LIVE TOOL: the Stage 2 plan records this diagnostic as written, reviewed and scheduled for execution. | evidence: docs/architecture_notes/x3d_integration_macro_plan/stage_2_wrist_loss_assessment.md:12, :30, :38; roots: audit_plan.md:25-36 | leave as planned validation tooling | confidence high

WP7-27 | D-prod | src/bst_x/validation_scripts/mmpose_heuristic_investigation/diagnose_top_k_capture.py:main | ONE-SHOT GATE: this was added during the completed sticky-anchor selector investigation. | evidence: docs/architecture_notes/mmpose_heuristic/mmpose_phase1_extraction_plan.md:387-390, src/bst_x/validation_scripts/mmpose_heuristic_investigation/diagnose_top_k_capture.py:13-15; roots: audit_plan.md:25-36; no production caller | archive candidate | confidence high

WP7-28 | D-prod | src/bst_x/validation_scripts/mmpose_heuristic_investigation/find_busted_clips.py:main | ONE-SHOT GATE: this generated the completed Phase 1 whole-clip and hit-zone busted-stem lists. | evidence: docs/architecture_notes/mmpose_heuristic/mmpose_phase1_extraction_plan.md:354-356, :387; src/bst_x/preparing_data/raw_extract.py:6; roots: audit_plan.md:25-36; no production caller | archive candidate with canonical stem lists retained | confidence high

WP7-29 | D-prod | src/bst_x/validation_scripts/mmpose_heuristic_investigation/render_detection_overlays.py:main | ONE-SHOT GATE: this overlay renderer belongs to the completed sticky-anchor investigation and is retained as an XAI reference. | evidence: docs/architecture_notes/mmpose_heuristic/mmpose_phase1_extraction_plan.md:390, src/xai/bst_x_fe_keypoint_overlay.md:99; roots: audit_plan.md:25-36; no production caller | archive candidate | confidence high

WP7-30 | D-prod | src/bst_x/validation_scripts/mmpose_heuristic_investigation/render_sticky_anchor_overlays.py:main | ONE-SHOT GATE: this renderer is documented as a completed historical overlay pass and an XAI sibling variant. | evidence: docs/architecture_notes/mmpose_heuristic/historical_mmpose_heuristic_investigation.md:602-605, src/xai/bst_x_fe_keypoint_overlay.md:100; roots: audit_plan.md:25-36; no production caller | archive candidate | confidence high

WP7-31 | D-prod | src/bst_x/validation_scripts/mmpose_heuristic_investigation/summarise_raw_ndet.py:main | ONE-SHOT GATE: this diagnostic was added to quantify the completed raw-extraction over-detection picture. | evidence: docs/architecture_notes/mmpose_heuristic/mmpose_phase1_extraction_plan.md:362, :389; roots: audit_plan.md:25-36; no production caller | archive candidate | confidence high

WP7-32 | D-prod | src/bst_x/validation_scripts/mmpose_heuristic_investigation/zeroed_frames_class_audit.py:main | ONE-SHOT GATE: the per-class zeroing audit is explicitly recorded as completed with committed output evidence. | evidence: docs/architecture_notes/mmpose_heuristic/historical_mmpose_heuristic_investigation.md:70-72, :602-605; roots: audit_plan.md:25-36; no production caller | archive candidate with committed audit outputs retained | confidence high

WP7-33 | C | src/bst_x/validation_scripts/perclass_clip_miss_rate.py:main | LIVE TOOL: the module documents a reusable `python -m validation_scripts.perclass_clip_miss_rate` entry point for clip-level analysis. | evidence: src/bst_x/validation_scripts/perclass_clip_miss_rate.py:21-23, docs/architecture_notes/frame_zeroing.md:290; roots: audit_plan.md:25-36 | leave as reusable validation tooling | confidence high

WP7-34 | C | src/bst_x/validation_scripts/perclass_shuttle_miss_vs_f1.py:main | LIVE TOOL: the validation README documents this correlation analysis as a reusable current tool. | evidence: src/bst_x/validation_scripts/README.md:172-180, src/bst_x/validation_scripts/perclass_shuttle_miss_vs_f1.py:31-34; roots: audit_plan.md:25-36 | leave as reusable validation tooling | confidence high

WP7-35 | C | src/bst_x/validation_scripts/raw_ndet_stats.py:main | LIVE TOOL: the README provides canonical `python -m validation_scripts.raw_ndet_stats` invocations before heuristic processing. | evidence: src/bst_x/validation_scripts/README.md:9-41, src/bst_x/validation_scripts/raw_ndet_stats.py:29-35; roots: audit_plan.md:25-36 | leave as reusable validation tooling | confidence high

WP7-36 | D-prod | src/bst_x/validation_scripts/refactoring/compare_b7_real_runs.py:main | ONE-SHOT GATE: this is the completed consumer half of the seeded B7 real-run comparison. | evidence: src/bst_x/validation_scripts/refactoring/README.md:22-23, :61-65; roots: audit_plan.md:25-36; no production caller | archive candidate | confidence high

WP7-37 | D-prod | src/bst_x/validation_scripts/refactoring/seed_and_run_bst_x_train.py:main | ONE-SHOT GATE: this is the completed launcher for pinned-RNG refactor verification. | evidence: src/bst_x/validation_scripts/refactoring/README.md:22, :61-63; roots: audit_plan.md:25-36; no production caller | archive candidate | confidence high

WP7-38 | D-prod | src/bst_x/validation_scripts/refactoring/smoke_b1_validate_gpu.py:main | ONE-SHOT GATE: this is the GPU-only validation gate from the completed simplification pass. | evidence: src/bst_x/validation_scripts/refactoring/README.md:15-20; roots: audit_plan.md:25-36; no production caller | archive candidate | confidence high

WP7-39 | D-prod | src/bst_x/validation_scripts/refactoring/smoke_b6_npz_writer.py:main | ONE-SHOT GATE: this protects the completed prediction-NPZ schema refactor and is not a production consumer. | evidence: src/bst_x/validation_scripts/refactoring/README.md:19-20; roots: audit_plan.md:25-36; no production caller | archive candidate | confidence high

WP7-40 | D-prod | src/bst_x/validation_scripts/refactoring/smoke_b7_seeded_train.py:main | ONE-SHOT GATE: this is the completed seeded end-to-end training equivalence smoke. | evidence: src/bst_x/validation_scripts/refactoring/README.md:20-21, :67-70; roots: audit_plan.md:25-36; no production caller | archive candidate | confidence high

WP7-41 | D-prod | src/bst_x/validation_scripts/refactoring/smoke_infer_bit_exact.py:main | ONE-SHOT GATE: this is the completed inference bit-exact smoke from the pre-Phase-2 tidy. | evidence: src/bst_x/validation_scripts/refactoring/README.md:8-13; roots: audit_plan.md:25-36; no production caller | archive candidate | confidence high

WP7-42 | D-prod | src/bst_x/validation_scripts/refactoring/smoke_prepare_2d_bit_exact.py:main | ONE-SHOT GATE: this is the completed 2D pose-extraction bit-exact smoke from the pre-Phase-2 tidy. | evidence: src/bst_x/validation_scripts/refactoring/README.md:8-13; roots: audit_plan.md:25-36; no production caller | archive candidate | confidence high

WP7-43 | D-prod | src/bst_x/validation_scripts/refactoring/tier1_comment_check.py:main | ONE-SHOT GATE: this is the completed mechanical comment-only verification gate. | evidence: src/bst_x/validation_scripts/refactoring/README.md:22-24; roots: audit_plan.md:25-36; no production caller | archive candidate | confidence high

WP7-44 | C | src/bst_x/validation_scripts/render_anchor_and_dets_overlay.py:main | LIVE TOOL: the module documents the manual overlay invocation for inspecting raw detections against sticky-anchor picks. | evidence: src/bst_x/validation_scripts/render_anchor_and_dets_overlay.py:37-50; roots: audit_plan.md:25-36 | leave as manual diagnostic tooling | confidence med

WP7-45 | D-prod | src/bst_x/validation_scripts/rtmlib_migration/_common.py:module | ONE-SHOT GATE: this shared helper is coupled only to the completed RTMLib parity gate family. | evidence: src/bst_x/validation_scripts/rtmlib_migration/_common.py:1-12, :94-96; callers: adapter_contract_test, bench_detector_pose_configs, gate_cpu_determinism, gate_cpu_downstream_byteeq, gate_cuda_selfvariance, gate_deployed_parity, gate_dtype_parity, gate_gpu_parity, gate_keypoint_value; roots: audit_plan.md:25-36 | archive with the migration gate bundle | confidence high

WP7-46 | D-prod | src/bst_x/validation_scripts/rtmlib_migration/adapter_contract_test.py:main | ONE-SHOT GATE: the RTMLib adapter contract check is explicitly part of the completed migration verification sequence. | evidence: docs/architecture_notes/rtmlib_migration/README.md:65-75; src/bst_x/validation_scripts/rtmlib_migration/adapter_contract_test.py:28-30; roots: audit_plan.md:25-36; no production caller | archive with the migration gate bundle | confidence high

WP7-47 | D-prod | src/bst_x/validation_scripts/rtmlib_migration/bench_detector_pose_configs.py:main | ONE-SHOT GATE: the timed detector and pose configuration benchmark belongs to the completed migration comparison. | evidence: docs/architecture_notes/rtmlib_migration/README.md:41, :80-81; src/bst_x/validation_scripts/rtmlib_migration/bench_detector_pose_configs.py:35-36; roots: audit_plan.md:25-36; no production caller | archive with the migration benchmark evidence | confidence high

WP7-48 | D-prod | src/bst_x/validation_scripts/rtmlib_migration/classify_phase_a_fails.py:module | ONE-SHOT GATE: this helper classifies the completed G8 parity output and is only coupled to migration gate constants. | evidence: src/bst_x/validation_scripts/rtmlib_migration/classify_phase_a_fails.py:1-17; docs/architecture_notes/rtmlib_migration/README.md:76-79; roots: audit_plan.md:25-36; no production caller | archive with the migration gate bundle | confidence high

WP7-49 | D-prod | src/bst_x/validation_scripts/rtmlib_migration/download_and_verify_models.py:main | ONE-SHOT GATE: model download and SHA verification is explicitly listed as migration verification tooling. | evidence: docs/architecture_notes/rtmlib_migration/README.md:65-69; src/bst_x/validation_scripts/rtmlib_migration/download_and_verify_models.py:22-23; roots: audit_plan.md:25-36; no production caller | archive with the migration provenance | confidence high

WP7-50 | D-prod | src/bst_x/validation_scripts/rtmlib_migration/gate_cpu_determinism.py:main | ONE-SHOT GATE: CPU determinism is one of the completed migration CPU checks. | evidence: docs/architecture_notes/rtmlib_migration/README.md:72-75; src/bst_x/validation_scripts/rtmlib_migration/gate_cpu_determinism.py:20-21; roots: audit_plan.md:25-36; no production caller | archive with the migration gate bundle | confidence high

WP7-51 | D-prod | src/bst_x/validation_scripts/rtmlib_migration/gate_cpu_downstream_byteeq.py:main | ONE-SHOT GATE: downstream byte equality is one of the completed migration CPU checks. | evidence: docs/architecture_notes/rtmlib_migration/README.md:72-75; src/bst_x/validation_scripts/rtmlib_migration/gate_cpu_downstream_byteeq.py:28-29; roots: audit_plan.md:25-36; no production caller | archive with the migration gate bundle | confidence high

WP7-52 | D-prod | src/bst_x/validation_scripts/rtmlib_migration/gate_cuda_selfvariance.py:main | ONE-SHOT GATE: CUDA self-variance is explicitly the completed G7 migration check. | evidence: docs/architecture_notes/rtmlib_migration/README.md:76-79; src/bst_x/validation_scripts/rtmlib_migration/gate_cuda_selfvariance.py:33-34; roots: audit_plan.md:25-36; no production caller | archive with the migration gate bundle | confidence high

WP7-53 | D-prod | src/bst_x/validation_scripts/rtmlib_migration/gate_deployed_parity.py:main | ONE-SHOT GATE: deployed parity is explicitly part of the completed RTMLib CPU verification sequence. | evidence: docs/architecture_notes/rtmlib_migration/README.md:72-75; src/bst_x/validation_scripts/rtmlib_migration/gate_deployed_parity.py:42-43; roots: audit_plan.md:25-36; no production caller | archive with the migration gate bundle | confidence high

WP7-54 | D-prod | src/bst_x/validation_scripts/rtmlib_migration/gate_dtype_parity.py:main | ONE-SHOT GATE: dtype parity is explicitly part of the completed RTMLib CPU verification sequence. | evidence: docs/architecture_notes/rtmlib_migration/README.md:72-75; src/bst_x/validation_scripts/rtmlib_migration/gate_dtype_parity.py:26-27; roots: audit_plan.md:25-36; no production caller | archive with the migration gate bundle | confidence high

WP7-55 | D-prod | src/bst_x/validation_scripts/rtmlib_migration/gate_gpu_parity.py:main | ONE-SHOT GATE: GPU parity is explicitly part of the completed migration sequence after CUDA self-variance. | evidence: docs/architecture_notes/rtmlib_migration/README.md:76-79; src/bst_x/validation_scripts/rtmlib_migration/gate_gpu_parity.py:36-37; roots: audit_plan.md:25-36; no production caller | archive with the migration gate bundle | confidence high

WP7-56 | D-prod | src/bst_x/validation_scripts/rtmlib_migration/gate_keypoint_value.py:main | ONE-SHOT GATE: keypoint-value parity is explicitly part of the completed RTMLib CPU verification sequence. | evidence: docs/architecture_notes/rtmlib_migration/README.md:72-75; src/bst_x/validation_scripts/rtmlib_migration/gate_keypoint_value.py:43-57; roots: audit_plan.md:25-36; no production caller | archive with the migration gate bundle | confidence high

WP7-57 | D-prod | src/bst_x/validation_scripts/rtmlib_migration/make_phase_a_sample.py:main | ONE-SHOT GATE: this creates the fixed sample consumed by the completed keypoint-value migration gate. | evidence: src/bst_x/validation_scripts/rtmlib_migration/gate_keypoint_value.py:50-53; src/bst_x/validation_scripts/rtmlib_migration/make_phase_a_sample.py:18-19; roots: audit_plan.md:25-36; no production caller | archive with the migration gate bundle | confidence high

WP7-58 | D-prod | src/bst_x/validation_scripts/rtmlib_migration/phase_a_decision.py:main | ONE-SHOT GATE: this is the completed final Phase-A migration decision gate. | evidence: docs/architecture_notes/rtmlib_migration/README.md:76-79; src/bst_x/validation_scripts/rtmlib_migration/phase_a_decision.py:40-43; roots: audit_plan.md:25-36; no production caller | archive with the migration decision evidence | confidence high

WP7-59 | C | src/bst_x/validation_scripts/shuttle_gap_length_distribution.py:main | LIVE TOOL: the validation README documents the gap-length distribution CLI and its reusable output. | evidence: src/bst_x/validation_scripts/README.md:172-180, src/bst_x/validation_scripts/shuttle_gap_length_distribution.py:19-22; roots: audit_plan.md:25-36 | leave as reusable validation tooling | confidence high

WP7-60 | C | src/bst_x/validation_scripts/shuttle_gap_y_distribution.py:main | LIVE TOOL: the validation README documents the gap-boundary y-distribution CLI and its reusable output. | evidence: src/bst_x/validation_scripts/README.md:172-180, src/bst_x/validation_scripts/shuttle_gap_y_distribution.py:20-22; roots: audit_plan.md:25-36 | leave as reusable validation tooling | confidence high

WP7-61 | C | src/bst_x/validation_scripts/validate_zeroed_frames.py:main | LIVE TOOL: the data-pipeline guide and validation README document this pre-training data-quality CLI. | evidence: src/bst_x/validation_scripts/README.md:43-112, src/bst_x/data_pipeline_to_model_train.md:257-292; roots: audit_plan.md:25-36 | leave as reusable validation tooling | confidence high

WP7-62 | C | src/bst_x/validation_scripts/verify_bst_x_train_target.py:main | LIVE TOOL: the actual filename is referenced by the structure-and-guards plan, although its module docstring still shows the stale shorter filename. | evidence: docs/architecture_notes/completed_general_refactors/structure_and_guards_pass/pose_3d_stream_design.md:172, src/bst_x/validation_scripts/verify_bst_x_train_target.py:8-14; roots: audit_plan.md:25-36 | leave live, with the stale invocation treated as documentation drift | confidence high

WP7-63 | C | src/bst_x/validation_scripts/verify_env_paths.py:main | LIVE TOOL: the validation README documents this as a pre-flight guard for the BST environment paths. | evidence: src/bst_x/validation_scripts/README.md:157-170, src/bst_x/validation_scripts/verify_env_paths.py:8-13; roots: audit_plan.md:25-36 | leave as pre-flight tooling | confidence high

WP7-64 | C | src/courtkeynet/validation_scripts/annotate_court_corners.py:module | LIVE TOOL SUPPORT: the retired GUI module remains live because the canonical off-frame annotator imports its helpers and tests import the module. | evidence: docs/annotator_unification_brief.md:3, :42; tests/test_courtkeynet_annotation.py:42-43, :569; roots: audit_plan.md:25-36; callers: annotate_court_corners_offframe and tests | leave helpers; retire only proven-unused GUI code in the owning package audit | confidence high

WP7-65 | C | src/courtkeynet/validation_scripts/annotate_court_corners_offframe.py:main | LIVE TOOL: the data README and manual-labelling documentation identify this as the canonical off-frame annotator. | evidence: data/amateur_court_corners/README.md:3-6, docs/scraper_pipeline/broadcast_nonstandard_camera_id/non_play_manual_labelling_20260731-095201.md:311-313; src/courtkeynet/validation_scripts/annotate.sh:71-76; roots: audit_plan.md:25-36 | leave as canonical annotation tooling | confidence high

WP7-66 | C | src/courtkeynet/validation_scripts/check_extrapolation.py:main | LIVE TOOL: the annotation runner dispatches its `check` command to this script and the module documents the direct invocation. | evidence: src/courtkeynet/validation_scripts/annotate.sh:38-41, src/courtkeynet/validation_scripts/check_extrapolation.py:16-18; roots: audit_plan.md:25-36 | leave as annotation validation tooling | confidence high

WP7-67 | C | src/courtkeynet/validation_scripts/court_landmarks.py:module | LIVE TOOL SUPPORT: the off-frame annotator, extrapolation checker, renderer and scorer import this court-landmark model. | evidence: src/courtkeynet/validation_scripts/annotate_court_corners_offframe.py:91, src/courtkeynet/validation_scripts/check_extrapolation.py:34, src/courtkeynet/validation_scripts/score_hand_corners.py:47; roots: audit_plan.md:25-36; callers checked | leave as shared annotation support | confidence high

WP7-68 | C | src/courtkeynet/validation_scripts/make_landmark_key.py:main | LIVE TOOL: the annotation runner dispatches its `key` command to this renderer and the module documents the direct invocation. | evidence: src/courtkeynet/validation_scripts/annotate.sh:42-44, src/courtkeynet/validation_scripts/make_landmark_key.py:10-12; roots: audit_plan.md:25-36 | leave as annotation tooling | confidence high

WP7-69 | C | src/courtkeynet/validation_scripts/render_court_overlay.py:main | LIVE TOOL: the module provides a documented manual CourtKeyNet overlay command for detector inspection. | evidence: src/courtkeynet/validation_scripts/render_court_overlay.py:20-22, src/courtkeynet/validation_scripts/annotate_court_corners.py:39; roots: audit_plan.md:25-36 | leave as manual diagnostic tooling | confidence med

WP7-70 | C | src/courtkeynet/validation_scripts/render_ground_truth.py:main | LIVE TOOL: the amateur-court README documents this renderer as the way to regenerate visual ground-truth audits. | evidence: data/amateur_court_corners/README.md:32-34, src/courtkeynet/validation_scripts/render_ground_truth.py:9-11; roots: audit_plan.md:25-36 | leave as annotation validation tooling | confidence high

WP7-71 | C | src/courtkeynet/validation_scripts/score_hand_corners.py:main | LIVE TOOL SUPPORT: tests load this scorer and the off-frame annotator documents its CSV contract. | evidence: tests/test_courtkeynet_annotation.py:46-47, src/courtkeynet/validation_scripts/annotate_court_corners_offframe.py:21; roots: audit_plan.md:25-36; callers checked | leave as hand-corner scoring tooling | confidence high

WP7-72 | C | src/bric/diagnostics/__init__.py:module | LIVE TOOL SUPPORT: package initialisation is required by the documented `bric.diagnostics.*` module entry points. | evidence: src/bric/README.md:24-29, audit_plan.md:32-35; roots: audit_plan.md:25-36 | leave as package support | confidence high

WP7-73 | C | src/bric/diagnostics/debug_court_bias.py:main | LIVE TOOL: BRIC documents this as a per-stroke court-coordinate diagnostic with a module entry point. | evidence: src/bric/README.md:24-29, src/bric/diagnostics/debug_court_bias.py:13; roots: audit_plan.md:25-36 | leave as BRIC diagnostic tooling | confidence high

WP7-74 | C | src/bric/diagnostics/evaluate_players.py:main | LIVE TOOL: BRIC documents this as the player-cache visibility and outlier diagnostic. | evidence: src/bric/README.md:24-29, src/bric/diagnostics/evaluate_players.py:15; roots: audit_plan.md:25-36 | leave as BRIC diagnostic tooling | confidence high

WP7-75 | C | src/bric/diagnostics/evaluate_shuttle.py:main | LIVE TOOL: the BRIC module documents per-video and aggregate quality-verdict invocations. | evidence: src/bric/README.md:24-29, src/bric/diagnostics/evaluate_shuttle.py:15-21; roots: audit_plan.md:25-36 | leave as BRIC diagnostic tooling | confidence high

WP7-76 | C | src/bric/diagnostics/validate_court_positions.py:main | LIVE TOOL: BRIC documents this as the court-position validation entry point. | evidence: src/bric/README.md:24-29, src/bric/diagnostics/validate_court_positions.py:14; roots: audit_plan.md:25-36 | leave as BRIC diagnostic tooling | confidence high

WP7-77 | C | src/bric/diagnostics/validate_rgb.py:main | LIVE TOOL: BRIC documents this as the RGB-cache contact-sheet validator and the root manifest lists its module entry point. | evidence: src/bric/README.md:24-29, src/bric/diagnostics/validate_rgb.py:9-20, audit_plan.md:32-35; roots: audit_plan.md:25-36 | leave as BRIC diagnostic tooling | confidence high

## 3. OUTWARD NOTES

`src/api/bst_x_inference.py:20,225` references `scripts/api_fixtures/rebuild_real.py`; confirm the API-owned fixture dependency remains intentional.

`src/shared/eval_plots.py:3-7` references `scripts/plots/confusion_matrix.py`; the shared plotting owner should decide whether the presentation script remains separately needed.

`scripts/archive` boundary: tracked hits are historical documentation and Ruff exclusions at `docs/architecture_notes/collation_taxon_pin_w_preds_refactor_log.md:61`, `docs/architecture_notes/completed_general_refactors/annotator_cleanup/README.md:27`, `pyproject.toml:154,178-179`, plus archive self-usage; no non-archive live-code import, path load or subprocess call into `scripts/archive` was found.

## 4. NOT CHECKED

Unlisted non-Python files such as `src/courtkeynet/validation_scripts/annotate.sh`, READMEs and committed output artefacts were used as evidence but not separately classified. Ignored, untracked and external scratch callers were excluded by the tracked-content rubric. No scripts or tests were executed, and no function-level U/P/S/O audit was performed.
