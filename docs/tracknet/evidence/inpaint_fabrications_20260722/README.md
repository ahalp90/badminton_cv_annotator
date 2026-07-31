# Inpaint fabrications evidence (2026-07-22)

Curated public evidence pack for the InpaintNet fabrication investigation
that ran on 2026-07-22. It combines the verified mechanism, the machine
artefacts that support it, and the useful analytical scripts.

The producer contract for the sidecar this investigation motivated is
`docs/tracknet/inpaint_sidecar.md`. The consumer state and open work sit
at `docs/tracknet/inpaint_sidecar_consumption.md`. This pack is the
underlying evidence.

## Top-level layout

- `inpaint_fabrications_investigation.md` — the main investigation
  report. Verified mechanism from the saved arrays through to the model
  weights, plus the weight-mode re-track findings folded into the body.
- `detector_options.md` — the decision sheet for flagging invented content
  in tracks that already exist, with the three detector options and their
  measured performance.
- `c11_landing_bisect/` — the landing-collapse investigation that
  discovered the fabrications, plus the code-level source trace and the
  full mechanism reproduction.
- `stride1_retrack/` — the WEIGHT-mode re-track outputs on `sset_01`
  (pilot), the per-mode fill / recurrence masks, the generated
  ShuttleTrack CSV, the sidecar manifest, and the analytical scripts.

## Source-disposition table

Copy = byte-exact copy of the source under
`local_scratch/autograder_architecture/now_tracked/inpaint_fabrications_20260722/source_tree/inpaint_fabrications_investigation/`.
Edited copy = a source file retained with private process details removed.
Distilled = the useful content lives in a public tracked doc; the source
remains local. Omitted = source not copied to Git and no public
distillation carries its distinct content (either superseded or entirely
private / conversational). Excluded bytecode = never copied.

| Source (relative to `inpaint_fabrications_investigation/`) | Disposition | Notes |
|---|---|---|
| `inpaint_fabrications_investigation.md` | Distilled | Verified mechanism report. A former private session appendix was folded into an ordinary `## Weight-mode re-track` section and the session framing removed. Source preserved locally. |
| `detector_options.md` | Copy | Detector decision sheet, public-facing already. |
| `c11_landing_bisect/c11_landing_report.md` | Copy | Plain-language landing collapse report. |
| `c11_landing_bisect/findings.txt` | Edited copy | Verified evidence ledger; supports the report. Personal and agent-process references were removed; the technical findings are unchanged. |
| `c11_landing_bisect/inpaint_source_findings.md` | Edited copy | Source-level code investigation with file:line citations; trailing Markdown padding removed. |
| `c11_landing_bisect/inpaint_source_worklog.md` | Omitted | Process worklog with no distinct technical evidence. Its verified facts (window length 16, the phase-5 pixel flip) are already public in `inpaint_source_findings.md` question 1 and question 8. |
| `c11_landing_bisect/inpaint_flag_writeout_recipe.md` | Edited copy | Recipe for the write-out change. Two stale relative links were corrected; the shipped sidecar (`docs/tracknet/inpaint_sidecar.md`) supersedes the design. |
| `c11_landing_bisect/upstream_issue_draft.md` | Edited copy | Draft GitHub issue for the upstream TrackNetV3 repo, with agent-process framing removed. |
| `c11_landing_bisect/summary.txt` | Edited copy | Bisect outcome; redundant final blank line removed. |
| `c11_landing_bisect/gt_join_summary.txt` | Edited copy | GT-join outcome; redundant final blank line removed. |
| `c11_landing_bisect/bisect_per_rally.csv` | Copy | Per-rally bisect result. |
| `c11_landing_bisect/gt_join_per_rally.csv` | Copy | Per-rally GT join result. |
| `c11_landing_bisect/loop_flag_diag.csv` | Copy | Loop-flag diagnostic. |
| `c11_landing_bisect/probe_inpaint_cycle.py` | Copy | Checkpoint probe (reproduces the loop from InpaintNet weights). |
| `c11_landing_bisect/probe_vs_track.py` | Copy | Diffs the probe against saved tracks. |
| `c11_landing_bisect/instrument_bisect.py` | Copy | Landing bisect instrumentation. |
| `c11_landing_bisect/gt_join.py` | Copy | GT-join scoring driver. |
| `c11_landing_bisect/sol_redteam.txt` | Distilled | External red-team transcript. Verified conclusions are folded into `c11_landing_report.md` / `findings.txt`. Source preserved locally. |
| `briefs/inpaintnet_bobbing_source_spec.md` | Distilled | Private commission brief. Its questions and hypothesis drove `inpaint_source_findings.md`, which supersedes it. |
| `briefs/stride1_retrack_commission.md` | Omitted | Private remote-launch details. No public information beyond what `stride1_retrack/summary.txt` already carries. |
| `stride1_retrack/summary.txt` | Copy | WEIGHT-mode re-track outcome, corrections folded in. |
| `stride1_retrack/pilot_nonoverlap_fillmask.npy` | Copy | Coordinate-match fill mask, non-overlap track. |
| `stride1_retrack/pilot_nonoverlap_recurrence_mask.npy` | Copy | Recurrence-v1 mask, non-overlap track. |
| `stride1_retrack/pilot_nonoverlap_recurrence_v2.npy` | Copy | Recurrence-v2 mask, non-overlap track. |
| `stride1_retrack/pilot_nonoverlap_recurrence_v3.npy` | Copy | Recurrence-v3 mask, non-overlap track (current detector). |
| `stride1_retrack/pilot_weight.npy` | Copy | The WEIGHT-mode saved track (x, y, visibility). |
| `stride1_retrack/pilot_weight_fillmask.npy` | Copy | Coordinate-match fill mask, WEIGHT track. |
| `stride1_retrack/pilot_weight_recurrence_mask.npy` | Copy | Recurrence-v1 mask, WEIGHT track. |
| `stride1_retrack/pilot_weight_recurrence_v2.npy` | Copy | Recurrence-v2 mask, WEIGHT track. |
| `stride1_retrack/pilot_weight_recurrence_v3.npy` | Copy | Recurrence-v3 mask, WEIGHT track. |
| `stride1_retrack/save_dir/pilot_288p_ball.csv` | Copy | Generated ShuttleTrack CSV from the WEIGHT-mode re-track. |
| `stride1_retrack/fill_sidecar_manifest.json` | Copy | Provenance manifest for the Option 3 sidecar run. |
| `stride1_retrack/exit_code.txt` | Copy | Exit code of the WEIGHT-mode inference. |
| `stride1_retrack/*.py` (analyse, condensed_fabrication, contacts_on_constant, diagnose, make_fill_sidecar, rule_recurrence, rule_recurrence_v2, rule_recurrence_v3, shapes_and_fate, verify_redteam) | Copy | Analytical scripts producing the outputs above. |
| `stride1_retrack/*_console.txt` | Copy | Console output beside each analytical script. |
| `stride1_retrack/launch_state.md` | Omitted | Private HPC ops state (SSH master socket, tmux session id, remote paths). No public information beyond what `summary.txt` records. |
| `stride1_retrack/{launch.sh, poll_remote.sh, pull.sh, run_predict.sh, run_predict_remote_copy.sh, setup_and_dryrun.sh, watcher.sh}` | Omitted | HPC-specific orchestration scripts hardcoded to bourbaki paths. The exact `predict.py` invocation is quoted in `summary.txt` § RUN DETAILS. |
| `stride1_retrack/logs/predict.log` | Omitted | ~266 kB of tqdm progress bars from the WEIGHT-mode inference. |
| `stride1_retrack/logs/tmux_console.log` | Omitted | Tmux console dump duplicating `predict.log`. |
| `stride1_retrack/__pycache__/`, all `*.pyc` | Excluded bytecode | Never copied. |

## Historical paths in the exact bytes

Several tracked evidence files still carry pre-cleanup absolute paths:

- Console transcripts in `stride1_retrack/*_console.txt` print the
  pre-consolidation working directory. They were left verbatim on
  purpose, since the numbers in them are the evidence the investigation
  rests on.
- `inpaint_source_findings.md` and `inpaint_flag_writeout_recipe.md` pin
  file:line citations to the `wt_annotator` worktree at commit `d04a789`.
  That anchor lets a reader run `git log -S` / `git grep` at that commit
  to follow any symbol that has since moved. Treat the anchors as
  historical, not current.
- `inpaint_flag_writeout_recipe.md` predates the shipped sidecar. Its
  proposed pseudocode has been superseded by `9475036`; keep the recipe
  as design archaeology only. `docs/tracknet/inpaint_sidecar.md` is the
  current contract.
- Several scripts and prose files load or cite `pilot_track_npy/1.npy`
  (the raw TrackNetV3 + InpaintNet track cache for pilot) and its
  sibling `pilot_track_npy/README.md`. That cache is bulky raw model
  output and is not part of this pack. The derived `.npy` fill and
  recurrence masks under `stride1_retrack/` are the public-facing
  artefacts computed from it and are what a reader can verify against.
- `stride1_retrack/summary.txt` inventories launch scripts and raw progress
  logs that this curated pack omits. Two bisect scripts also retain their
  historical agent-job output paths. These references describe the original
  run; neither set is required to read or reuse the retained evidence.

## Errata

- `c11_landing_bisect/findings.txt` says "256x288 tracking grid" once. The
  source carries the same slip; the correct TrackNet grid is 512x288, as the
  surrounding arithmetic and other reports state.

## Cross-links

- Producer contract: `docs/tracknet/inpaint_sidecar.md`.
- Consumer state and open work: `docs/tracknet/inpaint_sidecar_consumption.md`.
- Serve-prepend lookback (a downstream consumer that the sidecar informs):
  `docs/archive/serve_prepend_lookback.md`.
