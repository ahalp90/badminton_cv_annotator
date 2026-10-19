"""Check added context without changing the frozen court decisions or labels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dataset_builder.vision import load_court_vision, load_json_gz, persist_court_vision
from experiments.annotator.heuristic_only_tune import padding

INPUTS = (
    Path(__file__).resolve().parents[1] / 'experiments/annotator/monotrack_benchmark/benchmark_results/inputs'
)


def _item() -> dict[str, Any]:
    return {
        'fps': 10.0, 'frame_count': 600,
        'spans': [
            {'span_id': 0, 'start_frame': 100, 'end_frame': 201, 'label_frames': [110, 190],
             'label_sides': ['Top', 'Bot']},
            {'span_id': 1, 'start_frame': 400, 'end_frame': 501, 'label_frames': [410, 490],
             'label_sides': ['Bot', 'Top']},
        ],
    }


@pytest.mark.parametrize(
    ('before', 'after', 'expected'),
    [(3.0, 1.0, [(90, 201), (390, 501)]), (1.0, 3.0, [(100, 220), (400, 521)])],
)
def test_padding_stops_at_scene_cuts_outside_each_original_window(
    before: float, after: float, expected: list[tuple[int, int]],
) -> None:
    item = _item()
    cuts = ((0, 90), (90, 150), (150, 220), (220, 390), (390, 450), (450, 550), (550, 600))
    expanded, counts = padding.padded_spans(item, cuts, before, after)
    assert [(span['start_frame'], span['end_frame']) for span in expanded] == expected
    assert counts['windows'] == 2
    assert counts['changed_starts'] == (2 if before > 1 else 0)
    assert counts['changed_ends'] == (2 if after > 1 else 0)
    for original, padded in zip(item['spans'], expanded, strict=True):
        assert padded['label_frames'] == original['label_frames']
        assert padded['label_sides'] == original['label_sides']
        assert padded['start_frame'] <= original['start_frame']
        assert padded['end_frame'] >= original['end_frame']
    assert item['spans'][0]['start_frame'] == 100


def test_gap_midpoint_and_video_ends_limit_large_padding() -> None:
    expanded, counts = padding.padded_spans(_item(), ((0, 600),), 30.0, 30.0)
    assert [(span['start_frame'], span['end_frame']) for span in expanded] == [(0, 300), (300, 600)]
    assert counts['clipped_starts'] == counts['clipped_ends'] == 2
    assert counts['added_before_frames'] == 200
    assert counts['added_after_frames'] == 198


def test_cut_at_an_original_edge_prevents_crossing_it() -> None:
    item = _item()
    cuts = ((0, 100), (100, 201), (201, 400), (400, 501), (501, 600))
    expanded, counts = padding.padded_spans(item, cuts, 3.0, 3.0)
    assert expanded == item['spans']
    assert counts['changed_starts'] == counts['changed_ends'] == 0
    assert counts['clipped_starts'] == counts['clipped_ends'] == 2


@pytest.mark.parametrize(('before', 'after'), [(0.5, 1.0), (1.0, 0.5), (float('nan'), 1.0), (1.0, float('inf'))])
def test_invalid_padding_is_rejected_before_creating_outputs(tmp_path: Path, before: float, after: float) -> None:
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='at least one second'):
        padding.prepare(tmp_path / 'missing', output, before, after)
    assert not output.exists()


def _real_item() -> dict[str, Any]:
    return next(item for item in load_json_gz(INPUTS / 'manifest.json.gz')['items'] if item['item_id'] == '9')


@pytest.mark.parametrize('name', padding.COURT_FOLDERS)
def test_one_second_roundtrip_keeps_all_court_fields_and_masks(name: str, tmp_path: Path) -> None:
    item = _real_item()
    options = {'video_id': item['item_id'], 'frame_count': item['frame_count'],
               'resolution': tuple(map(float, item['resolution']))}
    original = load_court_vision(INPUTS / name / item['item_id'], **options)
    discovery = load_court_vision(INPUTS / 'discovery' / item['item_id'], **options)
    spans, counts = padding.padded_spans(item, discovery.raw_cuts, 1.0, 1.0)
    assert spans == item['spans']
    assert counts['changed_starts'] == counts['changed_ends'] == 0
    padded = padding.extend_courts(original, item['spans'], spans)
    persist_court_vision(tmp_path, court=padded, **options)
    restored = load_court_vision(tmp_path, **options)
    assert restored.raw_cuts == original.raw_cuts
    assert load_json_gz(tmp_path / 'court_evidence.json.gz') == load_json_gz(
        INPUTS / name / item['item_id'] / 'court_evidence.json.gz'
    )
    np.testing.assert_array_equal(restored.evidence.keep_vote, original.evidence.keep_vote)
    np.testing.assert_array_equal(restored.evidence.court_present, original.evidence.court_present)


@pytest.mark.parametrize('name', padding.COURT_FOLDERS)
@pytest.mark.parametrize(('before', 'after'), [(3.0, 1.0), (1.0, 3.0), (30.0, 30.0)])
def test_extended_courts_keep_geometry_votes_and_rejected_rallies(
    name: str, before: float, after: float, tmp_path: Path,
) -> None:
    item = _real_item()
    options = {'video_id': item['item_id'], 'frame_count': item['frame_count'],
               'resolution': tuple(map(float, item['resolution']))}
    original = load_court_vision(INPUTS / name / item['item_id'], **options)
    discovery = load_court_vision(INPUTS / 'discovery' / item['item_id'], **options)
    original_present = original.evidence.court_present.copy()
    spans, _counts = padding.padded_spans(item, discovery.raw_cuts, before, after)
    padded = padding.extend_courts(original, item['spans'], spans)
    persist_court_vision(tmp_path, court=padded, **options)
    restored = load_court_vision(tmp_path, **options)
    before_payload = load_json_gz(INPUTS / name / item['item_id'] / 'court_evidence.json.gz')
    after_payload = load_json_gz(tmp_path / 'court_evidence.json.gz')
    before_rows = before_payload['inputs'].pop('homography_rows')
    after_rows = after_payload['inputs'].pop('homography_rows')
    assert before_payload['inputs'] == after_payload['inputs']
    assert before_rows['columns'] == after_rows['columns']
    assert before_rows['index'] == after_rows['index']
    for old, new in zip(before_rows['data'], after_rows['data'], strict=True):
        assert old[0] == new[0]
        assert old[3:] == new[3:]
    old_records = {(row['start_frame'], row['end_frame']): row for row in before_payload['scene_records']}
    new_records = {(row['start_frame'], row['end_frame']): row for row in after_payload['scene_records']}
    moved = {'scene_index', 'start_frame', 'end_frame', 'exactly_two_fraction'}
    for old, new in zip(item['spans'], spans, strict=True):
        start, end = old['start_frame'], old['end_frame']
        new_start, new_end = new['start_frame'], new['end_frame']
        old_record, new_record = old_records[start, end], new_records[new_start, new_end]
        assert {key: value for key, value in old_record.items() if key not in moved} == {
            key: value for key, value in new_record.items() if key not in moved
        }
        np.testing.assert_array_equal(restored.evidence.court_present[start:end], original_present[start:end])
        assert np.all(restored.evidence.court_present[new_start:start] == old_record['scene_valid'])
        assert np.all(restored.evidence.court_present[end:new_end] == old_record['scene_valid'])
    np.testing.assert_array_equal(restored.evidence.keep_vote, original.evidence.keep_vote)
    np.testing.assert_array_equal(original.evidence.court_present, original_present)
