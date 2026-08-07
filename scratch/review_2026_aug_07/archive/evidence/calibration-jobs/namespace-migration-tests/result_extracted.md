I’ll inspect the target test file, its follow-history, the imported production symbols, and the permitted pytest collection command. I’ll keep the work read-only and report evidence only.
The target file contains the retained T6/T8/T10/T11 families described in its module docstring. The exact collection command could not start because pytest’s capture setup found no writable temporary directory in this read-only worker; I’ll retry collection with capture disabled and retain the failed command as a counterevidence item.
Collection now succeeds with capture disabled: 38 items, made up of 18 test functions plus four six-case `fe_dir` parametrisations. I’m doing the final reference and history checks, including the exact requested command’s failure mode and the read-only diff/status check.
**Question**

`tests/test_namespace_migration.py` contains durable BST-X artefact and namespace-regression guards.

**Scope inspected**

Read-only inspection of the test file, its git history, `conftest.py`, relevant `src/bst_x` modules, retained artefacts, and text references. No files changed; no diff exists. No `.env` or credentials were read.

**Evidence**

Test inventory: 18 functions, 38 collected cases.

- `test_t6_six_retained_weights_resolve` (`:91`, 0): asserts six retained run weights exist and are files.
- `test_t6_retained_weights_match_run_manifests` (`:97`, 0): each weight basename appears in its run manifest.
- `test_t6_post_switchover_weights_prefixed_bst_x` (`:111`, 0): post-switch retained weights use the `bst_x_` prefix.
- `test_t8_fe_jsons_exact_file_set` (`:165`, 1 parameter, 6 cases): each sidecar directory contains exactly five expected gzipped JSON files.
- `test_t8_clips_split_schema` (`:173`, 1 parameter, 6 cases): test/val clip JSONs have exact keys, matching metadata, non-empty clips, and no `model_name` key.
- `test_t8_perclass_stats_schema` (`:189`, 1 parameter, 6 cases): per-class JSONs have exact keys and non-empty per-class statistics.
- `test_t8_clip_index_schema` (`:203`, 1 parameter, 6 cases): clip-index JSON has the expected top-level and entry keys.
- `test_t8_retained_weight_dirs_have_fe_jsons` (`:216`, 0): every retained run has the complete sidecar set.
- `test_t10_baseline_dir_has_at_least_one_tracked_weight` (`:250`, 0): the Chang baseline contains a tracked weight.
- `test_t10_baseline_dir_never_carries_bst_x_prefix` (`:259`, 0): baseline weights never use `bst_x_`.
- `test_t10_baseline_weight_files_match_expected_prefix` (`:265`, 0): baseline filenames match the expected mixed- or lower-case triple.
- `test_t10_baseline_manifest_declares_the_full_triple` (`:274`, 0): the baseline manifest declares exactly that triple.
- `test_t11_stage1_bst_refactor` (`:371`, 0): tracked text contains no stale `bst_refactor` references.
- `test_t11_stage2_module_paths` (`:390`, 0): tracked text contains no old module paths or `build_bst_network`.
- `test_t11_stage4_extras_group` (`:407`, 0): tracked text contains no legacy `bst-runtime` extra.
- `test_t11_stage5_legacy_env_vars` (`:417`, 0): tracked text contains none of the listed legacy `BST_*` environment names.
- `test_t11_stage6_bst_cg_ap_filename_prose` (`:460`, 0): scoped tracked text contains no mixed-case `bst_CG_AP`.
- `test_t11_stage7_legacy_venv_name` (`:479`, 0): tracked text contains no legacy `venv-bst` name when enabled.

All current assertions are artefact, schema, or text-scan observations. No current test asserts import identity or re-export equality.

History:

- Introducing commit: `12b30c9` — “BST-X rebrand: namespace + env vars + weights + extras + docs”. It added the file with 1,034 lines. The commit describes T1-T12 as gates for the BST-to-BST-X migration.
- `2369971` later pruned the one-shot migration checks, leaving T6, T8, T10 and T11 as durable guards.
- Later modifying commits: `e5e3cb3` “BST-X rebrand: post-merge sweep fixups”; `b5440cf` “Pre-restructure tidy: drop dead files, presentation plots tracked”; `fc6aa62` “Plan 3 restructure: dissolve nested layout, hoist runs + data to root”; `2369971` “Plan 3 restructure: scratch/ consolidation + validation_scripts + tests prune + audit catch-ups”; `7d6203a` “Archive completed web demo”; `4a8945f` “Align project configuration with CV annotator”; `b4bdda7` “Correct classifier archive boundaries”.

Pinned production surface:

- `bst_x_common` must be importable during collection.
- `bst_x_common.MODELS` must exist; its `BST_X` key controls several skip gates.
- `pipeline.data_access` is probed for the stage-5 gate; `ENV_VAR_RENAMES` is expected to be absent.
- The tests also pin the retained-weight tree, run manifests, sidecar schemas, Chang baseline filenames, `src/bst_x` versus `src/bst_refactor`, the `bst-x-runtime` extra, legacy environment names, and legacy virtual-environment text.

**Production references**

- `bst_x_common`: yes, imported by `src/bst_x/bst_x_train.py:47` and `src/bst_x/bst_x_infer.py:38`.
- `MODELS`: no direct production import found; defined and used in `src/bst_x/bst_x_common.py:34` and `:76`.
- `BST_X`: no import; production registry key at `src/bst_x/bst_x_common.py:36`, with train/infer defaults also using it.
- `pipeline.data_access`: yes, imported by `src/bst_x/bst_x_train.py:45` and `src/bst_x/bst_x_infer.py:37`.
- `ENV_VAR_RENAMES`: no current production reference found; the test checks its absence at `tests/test_namespace_migration.py:361`.

**Test references**

Dynamic production probes are at `tests/test_namespace_migration.py:39`, `:68`, `:73`, `:342`, and `:355`. Artefact discovery is at `:58`, `:83`, and `:152`. The staged scans are implemented from `:371` through `:486`.

**Counterevidence**

The requested collection command exited 1 before collection because pytest could not create a temporary capture file. With capture disabled, `~/.venvs/badminton-cicd/bin/python -m pytest tests/test_namespace_migration.py --collect-only -q -s` succeeded: `38 tests collected`. The empty git status and diff confirm no changes.

**Unresolved/dynamic surfaces**

The six `fe_dir` cases are generated from the current retained run weights. Several tests can skip based on `MODELS`, filesystem layout, `pipeline.data_access`, `pyproject.toml`, `.env.example`, or `RENAME_SCAN_VENV=1`; collection count does not indicate execution count.
