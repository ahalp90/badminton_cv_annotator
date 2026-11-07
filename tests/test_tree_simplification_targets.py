"""Five-frame labels and span-local fallback for fixed labelled option pools."""

import lzma
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pytest

from annotator.rallies.build_contact_proposals import RallyContext
from annotator.training.rally import RallyFitConfig, prepare_training_pools
from annotator.training.rally_targets import option_targets_at_tolerances
from annotator.training.sequences import LabelledRally, SequenceTrainingVideo
from annotator.types import ContactEvent, ContactSequence, CourtHalf
from experiments.annotator import development as experiment
from experiments.annotator.tree_simplification import targets
from experiments.annotator.tree_simplification.options import (
    LabelledOptions,
)
from experiments.annotator.tree_simplification.targets import (
    TargetJob,
    _target_job,
    preferred_targets,
    prepare_targets,
    raw_five_targets,
)


def make_pool(
    sequences: tuple[ContactSequence, ...], targets: tuple[int, ...], *, fps: float = 30.0,
) -> LabelledOptions:
    initial = ContactSequence(
        0, 0, 200, (ContactEvent(20, 0.9, CourtHalf.TOP), ContactEvent(60, 0.9, CourtHalf.BOT)),
    )
    events = initial.events
    rally = LabelledRally('rally', (20, 60), (CourtHalf.TOP, CourtHalf.BOT))
    context = RallyContext(
        fps, (initial,), events, {}, {}, frame_count=210,
    )
    video = SequenceTrainingVideo('video', 'group', context, (rally,))
    return LabelledOptions(
        sequences,
        np.zeros((len(sequences), 1), dtype=np.float32),
        np.asarray(targets, dtype=np.int8),
        {0: np.arange(len(sequences), dtype=np.int64)},
        video,
    )


@pytest.mark.parametrize(('fps', 'shift'), [(25.0, 4), (30.0, 5), (60.0, 10)])
def test_two_tolerances_scale_labels(fps, shift):
    exact = ContactSequence(0, 0, 120, (ContactEvent(20, 0.9, CourtHalf.TOP), ContactEvent(60, 0.9, CourtHalf.BOT)))
    shifted = replace(exact, events=(ContactEvent(20 + shift, 0.9, CourtHalf.TOP),
                                     ContactEvent(60 + shift, 0.9, CourtHalf.BOT)))
    pool = make_pool((exact, shifted), (0, 0), fps=fps)
    wide, tight = option_targets_at_tolerances(pool.video, pool.runtime, (6, 3))
    np.testing.assert_array_equal(wide, [1, 1])
    np.testing.assert_array_equal(tight, [1, 0])


def test_configured_selector_tolerances_reach_real_target_labels(monkeypatch):
    exact = ContactSequence(0, 0, 200, (ContactEvent(20, 0.9, CourtHalf.TOP), ContactEvent(60, 0.9, CourtHalf.BOT)))
    shifted = replace(exact, events=(ContactEvent(25, 0.9, CourtHalf.TOP), ContactEvent(65, 0.9, CourtHalf.BOT)))
    pool = make_pool((exact, shifted), (0, 0))
    monkeypatch.setattr('annotator.training.rally.build_rally_options', lambda *_args: pool.runtime)
    strict = RallyFitConfig(preferred_tolerance_at_30_fps=3, fallback_tolerance_at_30_fps=6)
    relaxed = replace(strict, preferred_tolerance_at_30_fps=6)
    strict_pools = prepare_training_pools([pool.video], strict)
    relaxed_pools = prepare_training_pools([pool.video], relaxed)
    np.testing.assert_array_equal(strict_pools['video'].targets, [1, 0])
    np.testing.assert_array_equal(relaxed_pools['video'].targets, [1, 1])


def test_five_frame_positive_switches_known_rows_and_preserves_unknowns() -> None:
    sequences = tuple(
        ContactSequence(0, 0, 200, (ContactEvent(20 + row, 0.8, CourtHalf.TOP),))
        for row in range(3)
    )
    pool = make_pool(sequences, (1, 0, -1))

    actual = preferred_targets(pool, np.asarray([0, 1, 1], dtype=np.int8))

    np.testing.assert_array_equal(actual, (0, 1, -1))


def test_without_known_five_frame_positive_span_keeps_ten_frame_targets() -> None:
    sequences = tuple(
        ContactSequence(0, 0, 200, (ContactEvent(20 + row, 0.8, CourtHalf.TOP),))
        for row in range(2)
    )
    pool = make_pool(sequences, (1, 0))

    actual = preferred_targets(pool, np.asarray([0, 0], dtype=np.int8))

    np.testing.assert_array_equal(actual, pool.targets)


def test_unknown_ten_frame_row_cannot_supply_five_frame_positive() -> None:
    sequences = tuple(
        ContactSequence(0, 0, 200, (ContactEvent(20 + row, 0.8, CourtHalf.TOP),))
        for row in range(3)
    )
    pool = make_pool(sequences, (0, 0, -1))

    actual = preferred_targets(pool, np.asarray([0, 0, 1], dtype=np.int8))

    np.testing.assert_array_equal(actual, pool.targets)




@pytest.mark.parametrize(('fps', 'shift', 'expected'), [(30.0, 6, 0), (60.0, 10, 1)])
def test_raw_targets_use_scaled_five_frame_canonical_matching(fps: float, shift: int, expected: int) -> None:
    exact = ContactSequence(
        0, 0, 120, (ContactEvent(20, 0.9, CourtHalf.TOP), ContactEvent(60, 0.9, CourtHalf.BOT)),
    )
    shifted = ContactSequence(
        0, 0, 120,
        (ContactEvent(20 + shift, 0.9, CourtHalf.TOP), ContactEvent(60 + shift, 0.9, CourtHalf.BOT)),
    )
    pool = make_pool((exact, shifted), (0, 0), fps=fps)

    actual = raw_five_targets(pool)

    assert actual[0] == 1
    assert actual[1] == expected


def test_span_rally_pruning_matches_unpruned_targets_with_neighbour_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = ContactSequence(
        0, 0, 80, (ContactEvent(10, 0.9, CourtHalf.TOP), ContactEvent(30, 0.9, CourtHalf.BOT)),
    )
    second = ContactSequence(
        1, 100, 180, (ContactEvent(110, 0.9, CourtHalf.TOP), ContactEvent(130, 0.9, CourtHalf.BOT)),
    )
    sequences = (
        first,
        ContactSequence(0, 0, 80, (ContactEvent(16, 0.9, CourtHalf.TOP), ContactEvent(36, 0.9, CourtHalf.BOT))),
        second,
        ContactSequence(1, 100, 180, (ContactEvent(116, 0.9, CourtHalf.TOP), ContactEvent(136, 0.9, CourtHalf.BOT))),
    )
    events = (*first.events, *second.events)
    rallies = (
        LabelledRally('first', (10, 30), (CourtHalf.TOP, CourtHalf.BOT)),
        LabelledRally('boundary', (97,), (None,)),
        LabelledRally('second', (110, 130), (CourtHalf.TOP, CourtHalf.BOT)),
        LabelledRally('remote', (500,), (None,)),
    )
    context = RallyContext(
        30.0, (first, second), events, {}, {}, frame_count=220,
    )
    video = SequenceTrainingVideo('video', 'group', context, rallies)
    pool = LabelledOptions(
        sequences,
        np.zeros((4, 1), dtype=np.float32),
        np.zeros(4, dtype=np.int8),
        {0: np.asarray([0, 1]), 1: np.asarray([2, 3])},
        video,
    )
    original = targets.rally_targets.whole_rally_correctness
    observed_rallies = []

    def observe(sequence, labels, fps, *, tolerance_at_30_fps):
        observed_rallies.append(tuple(rally.rally_id for rally in labels))
        return original(
            sequence, labels, fps, tolerance_at_30_fps=tolerance_at_30_fps,
        )

    reference = np.empty(len(pool.sequences), dtype=np.int8)
    for row, sequence in enumerate(pool.sequences):
        reference[row] = original(sequence, video.rallies, video.context.fps, tolerance_at_30_fps=5)

    monkeypatch.setattr(targets.rally_targets, 'whole_rally_correctness', observe)
    actual = raw_five_targets(pool)

    np.testing.assert_array_equal(actual, reference)
    assert len(observed_rallies) == len(pool.sequences)
    assert all('remote' not in labels for labels in observed_rallies)
    assert all('boundary' not in labels for labels in observed_rallies[:2])
    assert all('boundary' in labels for labels in observed_rallies[2:])


def test_five_frame_row_mismatch_fails() -> None:
    sequence = ContactSequence(0, 0, 200, (ContactEvent(20, 0.8, CourtHalf.TOP),))
    pool = make_pool((sequence, sequence), (0, 0))

    with pytest.raises(ValueError, match='align'):
        preferred_targets(pool, np.asarray([1], dtype=np.int8))


def test_target_sidecar_is_compressed_and_matching_complete_output_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = make_pool(
        (
            ContactSequence(0, 0, 200, (ContactEvent(20, 0.9, CourtHalf.TOP), ContactEvent(60, 0.9, CourtHalf.BOT))),
        ),
        (0,),
    )
    pools_root = tmp_path / 'pools'
    pool_path = pools_root / 'videos' / 'video.joblib'
    pool_path.parent.mkdir(parents=True)
    joblib.dump(pool, pool_path)
    manifest = {'schema': targets.POOL_SCHEMA}
    job = TargetJob('video', 'group', 'train', pool_path, pools_root, manifest, tmp_path / 'targets')

    summary = _target_job(job)

    assert summary is not None
    target_path = job.output / 'videos' / 'video.npy.xz'
    metadata_path = job.output / 'videos' / 'video.json.gz'
    with lzma.open(target_path, 'rb') as stream:
        saved = np.load(stream, allow_pickle=False)
    assert saved.shape == (1,)
    assert experiment.read_json(metadata_path)['rows'] == 1

    def forbidden_recompute(_pool: LabelledOptions) -> np.ndarray:
        raise AssertionError('matching completed sidecars should be resumed')

    monkeypatch.setattr(targets, 'raw_five_targets', forbidden_recompute)
    assert _target_job(job) is None
    changed_manifest = replace(job, pools_manifest={'schema': 'pools/2'})
    with pytest.raises(ValueError, match='different source pool or manifest'):
        _target_job(changed_manifest)


def test_prepare_targets_smoke_only_uses_spawn_and_resumes(tmp_path: Path) -> None:
    pool = make_pool(
        (
            ContactSequence(0, 0, 200, (ContactEvent(20, 0.9, CourtHalf.TOP), ContactEvent(60, 0.9, CourtHalf.BOT))),
        ),
        (0,),
    )
    pools_root = tmp_path / 'pools'
    video_path = pools_root / 'videos' / 'video.joblib'
    video_path.parent.mkdir(parents=True)
    joblib.dump(pool, video_path)
    experiment.write_json(pools_root / 'pools.json.gz', {'schema': targets.POOL_SCHEMA})
    experiment.write_json(video_path.with_suffix('.json.gz'), {
        'video': 'video', 'group': 'group', 'split': 'train',
    })
    output = tmp_path / 'targets'

    summaries = prepare_targets(pools_root, output, workers=1, only='video')

    assert len(summaries) == 1
    assert summaries[0]['video'] == 'video'
    assert (output / 'videos' / 'video.npy.xz').is_file()
    assert prepare_targets(pools_root, output, workers=1, only='video') == []


def test_target_stage_rejects_pools_from_the_old_target_rule(tmp_path: Path) -> None:
    pools_root = tmp_path / 'pools'
    pools_root.mkdir()
    experiment.write_json(pools_root / 'pools.json.gz', {'schema': 'tree-simplification-option-pools/1'})
    with pytest.raises(ValueError, match='unsupported target rule'):
        prepare_targets(pools_root, tmp_path / 'targets', workers=1)


@pytest.mark.parametrize(('tolerance', 'next_label'), [(5, 109), (10, 114)])
def test_proposal_targets_do_not_use_padding_to_reach_another_rally(tolerance, next_label):
    sequence = ContactSequence(0, 0, 100, (ContactEvent(95, .9, CourtHalf.TOP),))
    pool = make_pool((sequence,), (0,))
    video = replace(pool.video, rallies=(LabelledRally('next', (next_label,), (CourtHalf.TOP,)),))
    (actual,) = option_targets_at_tolerances(video, pool.runtime, (tolerance,))
    np.testing.assert_array_equal(actual, [-1])
