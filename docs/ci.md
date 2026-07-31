# CI / CD

CI lives in `.github/workflows/` — the YAML is commented. No auto-deploy: prod is
still a manual `docker compose -f docker-compose.prod.yml up --build -d` (see
`DEPLOYMENT.md`). This doc covers only what you can't get from reading the YAML.

## What runs

**`ci.yml`** (PRs + pushes to `main`), all blocking:
`lint` (ruff) · `test` (pytest) · `frontend` (npm build; self-skips if `frontend/`
is removed) · `docker-build` (builds the images, no push).

**`pr-quality.yml`** (PRs, all non-blocking):
`commit-lint` (compatibility status for existing branch protection) · `pr-body`
(optional template-section suggestions) · `main-files` (deterministic; inserts
a short **Main files changed** block into the PR body) · `advisory` (AI review,
including commit-message feedback).

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

## Enable the AI advisory (optional, free)

Off until you add a key; without one it skips silently. With one, it comments on
commit/PR legibility and only ever *warns* on rate limits or outages — never blocks.

1. Free key: <https://aistudio.google.com/app/apikey>
2. Add it as repo secret **`PR_MESSAGE_BOT_KEY`** (Settings → Secrets and variables → Actions).
3. Optional: set repo variable `PR_MESSAGE_BOT_MODEL` to override the default
   `gemini-2.5-flash` (e.g. a Gemma model id).

Called once per PR, so the ~1,500/day free tier is plenty. Fork PRs don't get
secrets, so it runs on in-repo branches only.

## Dependencies

`requirements.txt` is pinned from `uv.lock` so CI installs what the image ships.
`torch`/`torchvision` are unpinned — CI and the Dockerfile install them from an
explicit PyTorch index first. After changing deps, run
`./scripts/gen-requirements.sh --check` and update any drifted pins.

## Branch protection

Existing required-check settings can keep their current status names. PR
content cannot fail these checks.

## Local hooks (optional)

`pre-commit install --hook-type commit-msg --hook-type pre-commit` runs the same
gitlint + ruff before you push.
