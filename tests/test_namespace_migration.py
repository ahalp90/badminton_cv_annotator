"""Durable post-rebrand guards: retained weights + sidecars + Chang baseline + scans.

Pruned 2026-06-16 from the original T1-T12 namespace migration suite (full
design at docs/architecture_notes/namespace_migration_test_design.md). The
one-shot rebrand mechanics (T1/T2/T3/T4/T5/T7/T9/T12) all landed and were
trimmed once stable; the four families that remain catch ongoing regressions:

  T6  retained-weight integrity (run manifests agree, naming pins)
  T8  classifier sidecar JSON schemas (clip splits, perclass stats, clip index)
  T10 Chang baseline run untouched (weight filenames keep bst_cg_ap_ per the contract)
  T11 staged tree scans (no regression to legacy modules / dirs / extras / env vars / venv name)

CPU-only; runs in the laptop ``badminton-cicd`` venv and CI. Asserts only
against git-tracked artefacts so a fresh CI clone is complete.
"""

from __future__ import annotations

import gzip
import importlib
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Shared scaffolding
# ---------------------------------------------------------------------------

def _first_importable(*candidates: str):
    """Return the first importable module from an ordered candidate list.

    Lets every test that touches a renamed module survive Step 6 / Step 8 by
    asking for the new name first and falling back to the old one. Raises
    rather than skipping if neither imports: that state is a broken tree, not
    a not-yet state.
    """
    last_err: Exception | None = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            last_err = exc
    raise RuntimeError(
        f'None of {candidates} importable in this tree state; last error: {last_err}'
    )


def _experiments_dir() -> Path:
    candidate = (
        REPO_ROOT / 'experiments' / 'bst_x' / 'shuttleset'
    )
    if candidate.is_dir():
        return candidate
    raise RuntimeError(f'No experiments dir under experiments/bst_x/shuttleset in {REPO_ROOT}')


def _common_module():
    return _first_importable('bst_x_common')


def _switchover_landed() -> bool:
    """True once Step 6b.1 has added 'BST_X' to the MODELS dict."""
    return 'BST_X' in _common_module().MODELS


EXPERIMENTS = _experiments_dir()


# ---------------------------------------------------------------------------
# T6: retained-weight integrity
# ---------------------------------------------------------------------------

def _retained_weights() -> list[Path]:
    """Git-tracked weights only: local machines hold extra untracked run
    weights, and 'retained' means the tracked set a CI clone receives."""
    out = subprocess.check_output(
        ['git', '-C', str(REPO_ROOT), 'ls-files',
         'experiments/bst_x/shuttleset/run_*/weights/*.pt'], text=True,
    ).splitlines()
    return sorted(REPO_ROOT / rel for rel in out)


def _run_dir_for_weight(weight_path: Path) -> Path:
    return weight_path.parent.parent


def test_t6_six_retained_weights_resolve():
    weights = _retained_weights()
    assert len(weights) == 6
    assert all(weight.is_file() for weight in weights)


def test_t6_retained_weights_match_run_manifests():
    """Every retained weight is recorded by its training-run manifest."""
    for weight_path in _retained_weights():
        manifest_path = _run_dir_for_weight(weight_path) / 'manifest.yaml'
        manifest = yaml.safe_load(manifest_path.read_text())
        weight_names = {
            Path(serial['weights_path']).name
            for serial in manifest['serials']
            if serial.get('weights_path')
        }
        assert weight_path.name in weight_names, weight_path


@pytest.mark.skipif(not _switchover_landed(), reason='Step 6b not landed')
def test_t6_post_switchover_weights_prefixed_bst_x():
    """Every retained destination uses the post-switchover BST-X prefix."""
    for weight_path in _retained_weights():
        assert weight_path.name.startswith('bst_x_'), weight_path


# ---------------------------------------------------------------------------
# T8: sidecar schema invariants
# ---------------------------------------------------------------------------

FE_FILES = {'clip_index.json.gz', 'test.json.gz', 'val.json.gz',
            'perclass_stats_test.json.gz', 'perclass_stats_val.json.gz'}

CLIPS_SPLIT_TOP_KEYS = {'run_id', 'serial_no', 'split', 'class_list', 'clips'}
CLIPS_ENTRY_KEYS = {'clip_stem', 'softmax', 'top_k_idx', 'top_k_prob', 'y_pred', 'y_true'}
PERCLASS_TOP_KEYS = {'class_list', 'n_clips', 'per_class', 'split'}
PERCLASS_ENTRY_KEYS = {'f1', 'precision', 'recall', 'support_pred', 'support_true',
                       'top5_when_pred', 'top5_when_true'}
CLIP_INDEX_TOP_KEYS = {'clips'}
CLIP_INDEX_ENTRY_KEYS = {'ball_round', 'match', 'player_side', 'rally',
                         'raw_type_en', 'set_id', 'split', 'video_path'}
NPZ_FIELDS = {'logits', 'y_true', 'y_pred_top1', 'topk_idx', 'clip_stems',
              'class_list', 'run_id', 'serial_no', 'taxonomy_name'}


def _load_gz_json(p: Path) -> dict:
    with gzip.open(p, 'rt') as fh:
        return json.load(fh)


def _check_no_model_name_key(obj, path='root'):
    """Recursive walk asserting 'model_name' never appears as a dict key."""
    if isinstance(obj, dict):
        assert 'model_name' not in obj, f'{path}: unexpected model_name key'
        for k, v in obj.items():
            _check_no_model_name_key(v, f'{path}.{k}')
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_no_model_name_key(v, f'{path}[{i}]')


def _retained_weight_fe_dirs() -> list[Path]:
    """Return sidecar directories for the six retained BST-X weights."""
    out: set[Path] = set()
    for weight_path in _retained_weights():
        fe_dir = _run_dir_for_weight(weight_path) / 'fe_jsons'
        if fe_dir.is_dir():
            out.add(fe_dir)
    return sorted(out)


@pytest.mark.parametrize(
    'fe_dir', _retained_weight_fe_dirs(), ids=lambda p: p.parent.name,
)
def test_t8_fe_jsons_exact_file_set(fe_dir):
    """Every fe_jsons/ dir holds exactly the five gzipped json sidecars."""
    assert {p.name for p in fe_dir.iterdir()} == FE_FILES


@pytest.mark.parametrize(
    'fe_dir', _retained_weight_fe_dirs(), ids=lambda p: p.parent.name,
)
def test_t8_clips_split_schema(fe_dir):
    run_id = fe_dir.parent.name
    for split in ('test', 'val'):
        data = _load_gz_json(fe_dir / f'{split}.json.gz')
        assert set(data) == CLIPS_SPLIT_TOP_KEYS
        assert data['run_id'] == run_id
        assert data['split'] == split
        assert isinstance(data['clips'], list) and data['clips']
        for clip in data['clips']:
            assert set(clip) == CLIPS_ENTRY_KEYS
        _check_no_model_name_key(data)


@pytest.mark.parametrize(
    'fe_dir', _retained_weight_fe_dirs(), ids=lambda p: p.parent.name,
)
def test_t8_perclass_stats_schema(fe_dir):
    for split in ('test', 'val'):
        data = _load_gz_json(fe_dir / f'perclass_stats_{split}.json.gz')
        assert set(data) == PERCLASS_TOP_KEYS
        assert data['split'] == split
        assert isinstance(data['per_class'], dict) and data['per_class']
        for stats in data['per_class'].values():
            assert set(stats) == PERCLASS_ENTRY_KEYS
        _check_no_model_name_key(data)


@pytest.mark.parametrize(
    'fe_dir', _retained_weight_fe_dirs(), ids=lambda p: p.parent.name,
)
def test_t8_clip_index_schema(fe_dir):
    data = _load_gz_json(fe_dir / 'clip_index.json.gz')
    assert set(data) == CLIP_INDEX_TOP_KEYS
    clips = data['clips']
    if isinstance(clips, dict):
        for meta in clips.values():
            assert set(meta) == CLIP_INDEX_ENTRY_KEYS
    else:
        for clip in clips:
            assert set(clip) == CLIP_INDEX_ENTRY_KEYS
    _check_no_model_name_key(data)


def test_t8_retained_weight_dirs_have_fe_jsons():
    """Every retained BST-X run carries the complete sidecar set."""
    for weight_path in _retained_weights():
        fe_dir = _run_dir_for_weight(weight_path) / 'fe_jsons'
        assert fe_dir.is_dir(), f'{weight_path}: fe_jsons/ missing'
        assert {p.name for p in fe_dir.iterdir()} == FE_FILES


# ---------------------------------------------------------------------------
# T10: Chang baseline run untouched
# ---------------------------------------------------------------------------

BASELINE_DIR = EXPERIMENTS / 'foundation_chang_baseline'
BASELINE_PREFIX_MIXED = 'bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_25'
BASELINE_PREFIX_LOWER = 'bst_cg_ap_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_25'
BASELINE_TRIPLE_MIXED = {
    f'{BASELINE_PREFIX_MIXED}.pt',
    f'{BASELINE_PREFIX_MIXED}_2.pt',
    f'{BASELINE_PREFIX_MIXED}_3.pt',
}
BASELINE_TRIPLE_LOWER = {
    f'{BASELINE_PREFIX_LOWER}.pt',
    f'{BASELINE_PREFIX_LOWER}_2.pt',
    f'{BASELINE_PREFIX_LOWER}_3.pt',
}


def _baseline_lowercased_gate() -> bool:
    """Different signal from the asserted property, per the self-gating rule:
    the run-dir rename lands in the same 6b.2 commit as the baseline
    lowercase. If ANY run_*/weights/bst_x_*.pt exists, 6b.2 has landed."""
    return any(EXPERIMENTS.glob('run_*/weights/bst_x_*.pt'))


def test_t10_baseline_dir_has_at_least_one_tracked_weight():
    """The design doc says three Chang baseline weights exist on disk; in git
    only the serial-2 .pt was ever tracked. CI's fresh clone shows just that
    one. Assert at least one is present (i.e. the baseline dir is alive in
    the repo), and that nothing in it carries the rebrand prefix."""
    weights = sorted(p.name for p in (BASELINE_DIR / 'weights').glob('*.pt'))
    assert len(weights) >= 1, weights


def test_t10_baseline_dir_never_carries_bst_x_prefix():
    """Decision 4: the baseline never becomes ``bst_x_*``; it lowercases in place."""
    bst_x = list((BASELINE_DIR / 'weights').glob('bst_x_*.pt'))
    assert bst_x == []


def test_t10_baseline_weight_files_match_expected_prefix():
    """Any baseline weight present must match the expected case for the current
    rebrand stage: mixed-case pre-6b.2, lowercase post-6b.2."""
    weights = {p.name for p in (BASELINE_DIR / 'weights').glob('*.pt')}
    expected = BASELINE_TRIPLE_LOWER if _baseline_lowercased_gate() else BASELINE_TRIPLE_MIXED
    # Subset: locally all three may be on disk; on CI only the tracked subset.
    assert weights.issubset(expected), (weights - expected)


def test_t10_baseline_manifest_declares_the_full_triple():
    """Catches a files-renamed-manifest-missed half-application. The manifest
    is git-tracked, so the full triple shows up on CI regardless of which
    weight files happen to be tracked alongside it. The baseline manifest uses
    ``experiments/``-relative paths; compare basenames so the test is
    convention-agnostic."""
    manifest = yaml.safe_load((BASELINE_DIR / 'manifest.yaml').read_text())
    serial_basenames = {Path(s['weights_path']).name for s in manifest['serials']}
    expected = BASELINE_TRIPLE_LOWER if _baseline_lowercased_gate() else BASELINE_TRIPLE_MIXED
    assert serial_basenames == expected


# ---------------------------------------------------------------------------
# T11: staged orphan-string scan
# ---------------------------------------------------------------------------

TEXT_EXTS = {
    '.py', '.md', '.yaml', '.yml', '.toml', '.sh', '.ipynb', '.txt', '.tsv',
    '.jsx', '.js',
}
# .gitignore must be matched by name: pathlib treats the leading dot as a
# hidden-file prefix, so Path('.gitignore').suffix is '' and a TEXT_EXTS
# entry never fires.
EXPLICIT_TEXT_NAMES = {'.gitignore', '.env.example'}

GLOBAL_EXCLUDE_PREFIXES = (
    'local_scratch/',
    # Agentic working docs and archived review evidence quote old names as
    # historical record; the stale-name guards police live surfaces only.
    'scratch/',
)


def _tracked_text_files() -> list[Path]:
    """git ls-files filtered to text extensions. Tracked-only keeps CI
    deterministic and skips venvs/caches by construction."""
    out = subprocess.check_output(
        ['git', '-C', str(REPO_ROOT), 'ls-files'], text=True,
    ).splitlines()
    paths: list[Path] = []
    for rel in out:
        if any(rel.startswith(p) for p in GLOBAL_EXCLUDE_PREFIXES):
            continue
        p = Path(rel)
        if p.suffix in TEXT_EXTS or p.name in EXPLICIT_TEXT_NAMES:
            paths.append(p)
    return paths


def _scan_pattern(pattern: re.Pattern[str], paths, allow_path):
    """Return list of (relpath, lineno, line) hits not covered by allow_path."""
    hits = []
    for rel in paths:
        if allow_path(rel):
            continue
        try:
            for i, line in enumerate((REPO_ROOT / rel).read_text(errors='ignore').splitlines(), 1):
                if pattern.search(line):
                    hits.append((str(rel), i, line.rstrip()))
        except (FileNotFoundError, UnicodeDecodeError):
            continue
    return hits


def _step8_landed() -> bool:
    """Step 8 lands when the old src/bst_refactor directory is gone and the
    new src/bst_x is in place."""
    return not (REPO_ROOT / 'src' / 'bst_refactor').exists() and \
        (REPO_ROOT / 'src' / 'bst_x').exists()


def _step6_common_landed() -> bool:
    try:
        importlib.import_module('bst_x_common')
        return True
    except ImportError:
        return False


def _step5_runtime_landed() -> bool:
    pyproject = (REPO_ROOT / 'pyproject.toml').read_text()
    return re.search(r'^\s*bst-x-runtime\s*=', pyproject, flags=re.MULTILINE) is not None


def _step9b_landed() -> bool:
    """Step 9b removed ENV_VAR_RENAMES from pipeline.data_access. On today's
    main the mapping is also absent (Step 4 hasn't landed yet), so detect Step
    4 separately via a new BST_X_* var in .env.example and require both."""
    try:
        from pipeline import data_access
        if hasattr(data_access, 'ENV_VAR_RENAMES'):
            return False
    except ImportError:
        return False
    env_example = REPO_ROOT / '.env.example'
    if not env_example.exists():
        return False
    return 'BST_X_CLIPS_DIR' in env_example.read_text()


def test_t11_stage1_bst_refactor():
    if not _step8_landed():
        pytest.skip('Step 8 not landed: src/bst_refactor still exists')
    # The test file and design doc both quote the pattern verbatim; the two
    # historical-archive narrative docs reference scratch/project_history/
    # bst_refactor_deprecated/ as the actual archived directory name.
    allowed = {
        'docs/architecture_notes/namespace_migration_test_design.md',
        'docs/architecture_notes/historical_bst.md',
        'docs/architecture_notes/pre_phase_2_tidy_plan.md',
        'tests/test_namespace_migration.py',
    }
    hits = _scan_pattern(
        re.compile(r'bst_refactor'), _tracked_text_files(),
        allow_path=lambda rel: str(rel) in allowed,
    )
    assert hits == [], '\n'.join(f'{r}:{n}: {line}' for r, n, line in hits)


def test_t11_stage2_module_paths():
    if not _step6_common_landed():
        pytest.skip('Step 6 not landed: bst_x_common not importable')
    pattern = re.compile(r'main_on_shuttleset\.bst_(train|infer|common)\b|\bbuild_bst_network\b')
    # The test file itself contains the pattern as a regex literal; exempt it
    # to avoid a self-flag. Pickup-doc + assessment historical narratives
    # outside the in-scope code aren't part of this gate either.
    allowed = {
        'tests/test_namespace_migration.py',
    }
    hits = _scan_pattern(
        pattern, _tracked_text_files(),
        allow_path=lambda rel: str(rel) in allowed,
    )
    assert hits == [], '\n'.join(f'{r}:{n}: {line}' for r, n, line in hits)


def test_t11_stage4_extras_group():
    if not _step5_runtime_landed():
        pytest.skip('Step 5 not landed: pyproject.toml has no bst-x-runtime group')
    hits = _scan_pattern(
        re.compile(r'\bbst-runtime\b'), _tracked_text_files(),
        allow_path=lambda rel: False,
    )
    assert hits == [], '\n'.join(f'{r}:{n}: {line}' for r, n, line in hits)


def test_t11_stage5_legacy_env_vars():
    if not _step9b_landed():
        pytest.skip('Step 9b not landed: ENV_VAR_RENAMES still present')
    pattern = re.compile(
        r'\bBST_(CLIPS_DIR|CLIPS_CSV|SHUTTLE_NPY_DIR|MMPOSE_NPY_DIR|INPUTS_DIR'
        r'|DATA_DIR|LOCAL_CLIPS_DIR|REPO_ROOT|REGISTRY_PATH|SHUTTLE_CSV_DIR)\b'
    )
    # Allowlist:
    # - The design doc and this test file quote the legacy names verbatim.
    # - Historical refactor logs / dated tidy-plan narratives reference the
    #   pre-rebrand names as they were at the time.
    allowed = {
        'docs/architecture_notes/namespace_migration_test_design.md',
        'docs/architecture_notes/pre_phase_2_tidy_plan.md',
        'docs/archive/completed_general_refactors/data_access_integration_plan.md',
        'docs/archive/completed_general_refactors/dir_flatten_refactor.md',
        'docs/architecture_notes/collation_taxon_pin_w_preds_refactor_log.md',
        'docs/architecture_notes/collation_taxon_pin_w_preds_refactor.md',
        'tests/test_namespace_migration.py',
    }
    hits = _scan_pattern(
        pattern, _tracked_text_files(),
        allow_path=lambda rel: str(rel) in allowed,
    )
    assert hits == [], '\n'.join(f'{r}:{n}: {line}' for r, n, line in hits)


def _stage6_in_scope(rel: str) -> bool:
    """Scoped stage-6 corpus: live shipped docs, run manifests, and the Chang
    baseline dir. Scratch history (now relocated
    under docs/architecture_notes/) and the ledger legitimately mention old
    filenames; not in scope here."""
    if rel.startswith('docs/architecture_notes/'):
        return False
    if rel.startswith('docs/'):
        return True
    if rel.startswith('experiments/bst_x/shuttleset/run_'):
        return True
    return rel.startswith('experiments/bst_x/shuttleset/foundation_chang_baseline')


def test_t11_stage6_bst_cg_ap_filename_prose():
    if not _switchover_landed():
        pytest.skip('Step 6b not landed: SWITCHOVER_LANDED is False')
    pattern = re.compile(r'bst_CG_AP')
    hits = []
    for rel in (str(p) for p in _tracked_text_files()):
        if not _stage6_in_scope(rel):
            continue
        path = REPO_ROOT / rel
        try:
            lines = path.read_text(errors='ignore').splitlines()
        except FileNotFoundError:
            continue
        for i, line in enumerate(lines, 1):
            if pattern.search(line):
                hits.append((rel, i, line.rstrip()))
    assert hits == [], '\n'.join(f'{r}:{n}: {line}' for r, n, line in hits)


def test_t11_stage7_legacy_venv_name():
    if os.environ.get('RENAME_SCAN_VENV') != '1':
        pytest.skip('Set RENAME_SCAN_VENV=1 to run the venv-name scan (out-of-repo ops step)')
    hits = _scan_pattern(
        re.compile(r'venv-bst(?!-x)'), _tracked_text_files(),
        allow_path=lambda rel: False,
    )
    assert hits == [], '\n'.join(f'{r}:{n}: {line}' for r, n, line in hits)
