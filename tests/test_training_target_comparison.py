"""Training-target bookkeeping preserves counts and resumes completed videos."""
import lzma
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from experiments.annotator.monotrack_benchmark.writeup_scripts import (
    compare_training_targets as comparison,
)


def make_group(training, candidate='raw_baseline', seed=0):
    counts = {'first': {'split': 'train'}, 'second': {'split': 'validation'}}
    roots = {}
    targets = ([-1, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1])
    for policy, labels in zip(comparison.POLICIES, targets, strict=True):
        root = training / policy / 'test-run/candidates' / candidate / f'seed-{seed}'
        roots[policy] = root
        (root / 'selector').mkdir(parents=True)
        comparison.write_json(root / 'selector/settings.json.gz', {'original_options': False})
        comparison.write_json(root / 'selector/pool-counts.json.gz', counts)
        (root / 'pools/videos').mkdir(parents=True)
        (root / 'selector/targets').mkdir()
        for video in counts:
            pool = SimpleNamespace(sequences=tuple(SimpleNamespace(span_id=span) for span in (4, 9, 4, 12)),
                                   matrix=np.array([[1.0], [np.nan], [3.0], [4.0]]))
            joblib.dump(pool, root / 'pools/videos' / f'{video}.joblib')
            with lzma.open(root / 'selector/targets' / f'{video}.npy.xz', 'wb') as stream:
                np.save(stream, np.array(labels))
    return roots


@pytest.mark.parametrize('rows', [0, 1, 1000])
def test_counts_match_direct_row_and_span_counts(rows):
    random = np.random.default_rng(42)
    left, right = random.integers(-1, 2, size=(2, rows))
    spans = random.integers(0, 23, size=rows)
    actual = comparison.compare_arrays(left, right, spans)
    assert actual['rows'] == rows
    assert actual['changed'] == sum(before != after for before, after in zip(left, right, strict=True))
    for before in (-1, 0, 1):
        for after in (-1, 0, 1):
            assert actual[f'{before}_to_{after}'] == sum(
                first == before and second == after for first, second in zip(left, right, strict=True))
    positive_left = {span for span in set(spans) if any(left[spans == span] == 1)}
    positive_right = {span for span in set(spans) if any(right[spans == span] == 1)}
    assert actual['spans'] == len(set(spans))
    assert actual['left_positive_spans'] == len(positive_left)
    assert actual['right_positive_spans'] == len(positive_right)
    assert actual['lost_positive_spans'] == len(positive_left - positive_right)
    assert actual['gained_positive_spans'] == len(positive_right - positive_left)


def test_resume_reads_only_unfinished_videos(tmp_path, monkeypatch):
    training, out = tmp_path / 'training', tmp_path / 'out'
    make_group(training)
    original = comparison.compare_video
    calls = []

    def interrupted(roots, settings, identity):
        calls.append(identity)
        if identity == 'second':
            raise RuntimeError('interrupted')
        return original(roots, settings, identity)

    monkeypatch.setattr(comparison, 'compare_video', interrupted)
    with pytest.raises(RuntimeError, match='interrupted'):
        comparison.compare_group(training, out, 'raw_baseline', 0)
    checkpoint = out / 'per_video/raw_baseline/seed-0/first.json.gz'
    saved = checkpoint.read_bytes()

    def resumed(roots, settings, identity):
        assert identity == 'second'
        return original(roots, settings, identity)

    monkeypatch.setattr(comparison, 'compare_video', resumed)
    records = comparison.compare_group(training, out, 'raw_baseline', 0)
    assert calls == ['first', 'second']
    assert len(records) == 6
    assert checkpoint.read_bytes() == saved
    assert records[0]['-1_to_0'] == 1
    assert records[0]['0_to_1'] == 1
    assert records[0]['gained_positive_spans'] == 1


def test_parallel_groups_match_serial_results(tmp_path):
    training, out = tmp_path / 'training', tmp_path / 'out'
    for seed in (0, 100):
        make_group(training, seed=seed)
    with comparison.ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(comparison.compare_group, training, out, 'raw_baseline', seed)
                   for seed in (0, 100)]
        parallel = [future.result() for future in futures]
    serial = [comparison.compare_group(training, tmp_path / 'serial', 'raw_baseline', seed) for seed in (0, 100)]
    assert parallel == serial


def test_misaligned_features_fail_without_checkpoint(tmp_path):
    training, out = tmp_path / 'training', tmp_path / 'out'
    roots = make_group(training)
    path = roots['5_5'] / 'pools/videos/first.joblib'
    pool = joblib.load(path)
    pool.matrix[0, 0] = 99
    joblib.dump(pool, path)
    with pytest.raises(ValueError, match='features differ'):
        comparison.compare_group(training, out, 'raw_baseline', 0)
    assert not list(out.rglob('*.json.gz'))
