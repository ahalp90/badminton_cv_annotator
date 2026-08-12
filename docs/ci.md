# CI / CD

CI lives in `.github/workflows/`. The retired web demo and its deployment
workflow are no longer tested or deployed.

## What runs

**`ci.yml`** (PRs + pushes to `main`), all blocking:
`lint` (ruff and pyrefly) · `test` (pytest).

**`pr-quality.yml`** (PRs, all non-blocking):
`commit-lint` (compatibility status for existing branch protection) · `pr-body`
(optional template-section suggestions) · `main-files` (deterministic; inserts
a short **Main files changed** block into the PR body) · `advisory` (AI quick
read based on the PR text and implementation diff).

The pull request template still provides **What / Why / Testing / Reviewer
focus** sections. PR content does not make these jobs fail. Gitlint remains
available as an optional local hook. It does not enforce title or body length.

`main-files` (`scripts/pr_main_files.py`) lists the most-impactful changed files
(up to 8), ranked by churn × path relevance (`src/`, `training/` outrank config;
the `data/` `experiments/` `notebooks/` trees score 0 and never show), skipping
trivial (<3-line) and noise files (lockfiles, generated/minified, binary + model
blobs). Knobs are constants at the top of the script. It edits the PR body
between `<!-- main-files-start/end -->` markers
and only PATCHes when the block actually changes, so its own edit can't retrigger
the `edited` run. No key needed; on fork PRs the token is read-only so it no-ops.
Don't mark it required (it edits, doesn't gate).

## Enable the AI quick read (optional, free)

Off until you add a key; without one it skips silently. With one, it posts a
short explanation based mainly on a ranked sample of the implementation diff.
The sample takes up to six meaningful files, with per-file and total size
limits. Rate limits and outages only produce warnings, so the quick read never
blocks a PR. A cut-off or malformed model response also produces a warning and
does not replace the existing PR comment.

1. Free key: <https://aistudio.google.com/app/apikey>
2. Add it as repo secret **`PR_MESSAGE_BOT_KEY`** (Settings → Secrets and variables → Actions).
3. Optional: set repo variable `PR_MESSAGE_BOT_MODEL` to override the default
   `gemini-2.5-flash` (e.g. a Gemma model id).

Called once per PR, so the ~1,500/day free tier is plenty. Fork PRs don't get
secrets, so it runs on in-repo branches only.

## Dependencies

`requirements.txt` is pinned from `uv.lock` so CI uses a repeatable dependency
set. `torch` and `torchvision` are unpinned because CI installs their CPU builds
from the PyTorch index first. After changing dependencies, run
`./scripts/gen-requirements.sh --check` and update any drifted pins.

## Branch protection

Existing required-check settings can keep their current status names. PR
content cannot fail these checks.

## Local hooks (optional)

`pre-commit install --hook-type commit-msg --hook-type pre-commit` runs the same
gitlint + ruff before you push.
