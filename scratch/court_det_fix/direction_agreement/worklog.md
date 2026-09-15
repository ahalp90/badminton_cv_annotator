# Direction-agreement experiments: worklog

## Resume

- Stage: E0-E2 (membership recovery, baseline replay, anchor comparison, four frozen arms) launching on the remote host.
- Next action: poll `bash sync.sh status`; when `arms_exit_code.txt` reads 0, `bash sync.sh pull` and inspect `runs/<run>/e0/*.json.gz` checks before E3.
- Active job: label `arms` (script `run_arms`), PID in `runs/<run>/receipts/arms.pid` on the remote. No connection is held between polls.
- Last passed gate: 30 synthetic tests and Ruff on the experiment code, exit 0 (local).
- Blockers: none.
- Paths: local `scratch/court_det_fix/direction_agreement/`; remote mirror `R/direction_agreement/` (R is the packet's remote root); run name in `run_name.txt`; outputs under `runs/<run>/`.

## Scope and overrides

The packet is `scratch/court_det_fix/worklog/CLAUDE_DIRECTION_EXPERIMENTS.md`. The user's session instructions override it where they clash:

- Session model is Claude Fable 5.1, not Opus 5 xhigh. The manifest records the actual model.
- Local commits on `fix/court-det` are allowed. Audits by Opus 5 (medium) run at stage checkpoints via the CLI, read-only.
- Local outputs live in this tracked folder, not under the earlier 20260914 tree. The remote outputs live in the mirror folder, not under `automatic_axes_20260914/<run>`.
- The packet's provider-launch metadata folder was for a Codex-to-Claude handoff and is not used.
- Committed files use the aliases `R` (remote root) and `L` (local source tree at `scratch/court_det_fix/worklog/checks/independent/player_guided/20260914`). The gitignored `paths.local.sh` maps them to real locations.

## Log

- 2026-09-15 14:49. Folder, `.gitignore`, alias file and run name created. Read the packet, the shared modules it instruments (`detector._merge_lines`, `assignment.prepare_observations`, `vp_pruning.retain_pencils`), the matcher entry points (`run_automatic.generate`, `rescore_camera_pool.rescore`), the SVD helpers and the previous gallery renderer.
- 2026-09-15 14:52. All 43 shared modules and every saved input are byte-identical between `L` and `R` (MD5). The remote has no running jobs. Baseline checkpoint `9fcec94`; HEAD `d0c9a12` differs only by documentation.
- 2026-09-15 15:20. Wrote `common.py`, `membership.py`, `selection.py`, `run_arms.py`, `run_fits.py`, `run_matcher.py`, `diagnose_matrix.py`, `manifest.py`, `run_remote.sh`, `sync.sh` and three test modules. 30 synthetic tests pass; Ruff clean after auto-fix (import order, one unused import, two redundant casts, `datetime.UTC`).
