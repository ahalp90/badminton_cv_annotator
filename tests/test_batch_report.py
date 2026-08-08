"""Tests for the rally-segmentation batch outcome report."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from annotator.batch_report import (
    VideoOutcome,
    derive_batch_report_path,
    format_batch_report,
)
from annotator.run_video import AnnotatorResult
from annotator.types import ContactCandidate


def test_format_batch_report_mixed_outcomes_in_input_order():
    outcomes = [
        VideoOutcome('vid-x', 'processed', rallies=2, contacts=5),
        VideoOutcome('vid', 'excluded', reason='flagged doubles'),
        VideoOutcome('bravo', 'excluded', reason='no doubles row; not assuming singles'),
        VideoOutcome('charlie', 'processed', rallies=1, contacts=2),
        VideoOutcome('delta', 'skipped', reason='missing\tfps\nrow'),
        VideoOutcome('echo', 'skipped', reason='ValueError'),
    ]

    assert format_batch_report(outcomes) == (
        'batch completed: 2 of 6 videos processed\n\n'
        'Counts\n'
        '- videos processed: 2/6\n'
        '- videos excluded before processing: 2/6\n'
        '- videos skipped: 2/6\n'
        '- rallies written: 3\n'
        '- contacts written: 7\n\n'
        'Video outcomes\n'
        '- vid-x: processed; 2 rallies; 5 contacts\n'
        '- vid: excluded; flagged doubles\n'
        '- bravo: excluded; no doubles row; not assuming singles\n'
        '- charlie: processed; 1 rally; 2 contacts\n'
        '- delta: skipped; missing fps row\n'
        '- echo: skipped; ValueError\n\n'
        'Rally exclusions\n'
        '- not recorded by this CLI; rally-level reasons exist only in the run_video chain'
    )


def test_format_batch_report_empty_batch():
    assert format_batch_report([]) == (
        'batch completed: no input videos found\n\n'
        'Counts\n'
        '- videos processed: 0/0\n'
        '- videos excluded before processing: 0/0\n'
        '- videos skipped: 0/0\n'
        '- rallies written: 0\n'
        '- contacts written: 0\n\n'
        'Video outcomes\n'
        '- none\n\n'
        'Rally exclusions\n'
        '- not recorded by this CLI; rally-level reasons exist only in the run_video chain'
    )


def test_format_batch_report_none_processed_failure():
    outcomes = [VideoOutcome('vid', 'skipped', reason='ValueError')]

    assert format_batch_report(outcomes).startswith(
        'batch failed: 0 of 1 video processed\n'
    )


def test_publish_path_uses_custom_rally_spans_path(tmp_path):
    assert derive_batch_report_path(tmp_path / 'results' / 'rally_spans.csv') == (
        tmp_path / 'results' / 'rally_spans_batch_report.txt'
    )


def _run_cli(tmp_path, monkeypatch, *, extra_args, run_video):
    shuttle_dir = tmp_path / 'tracks'
    shuttle_dir.mkdir()
    np.save(shuttle_dir / 'vid.npy', np.zeros((4, 3)))
    spans_path = tmp_path / 'custom' / 'rally_spans.csv'
    contacts_path = tmp_path / 'custom' / 'contact_frames.csv'

    from annotator import rally_segmentation
    import annotator.run_video as run_video_module

    monkeypatch.setattr(run_video_module, 'run_video', run_video)
    monkeypatch.setattr(sys, 'argv', [
        'rally_segmentation',
        '--shuttle-dir', str(shuttle_dir),
        '--fps', '30',
        '--rally-spans-csv', str(spans_path),
        '--contact-frames-csv', str(contacts_path),
        *extra_args,
    ])
    rally_segmentation.main()
    return spans_path, contacts_path


def test_cli_publishes_report_with_custom_path_and_default_doubles_mode(
    tmp_path, monkeypatch, capsys,
):
    spans_path, contacts_path = _run_cli(
        tmp_path,
        monkeypatch,
        extra_args=[],
        run_video=lambda track, *args, **kwargs: AnnotatorResult(
            [(1, 3)], [ContactCandidate(0, 2, None, None, None)],
            [], {}, [], [], [], [], {}, {}, {}, {}, [],
        ),
    )

    report_path = tmp_path / 'custom' / 'rally_spans_batch_report.txt'
    report_text = report_path.read_text(encoding='utf-8')
    assert report_text.endswith('\n')
    assert report_text.count('\n') == report_text.rstrip('\n').count('\n') + 1
    assert capsys.readouterr().out == report_text
    assert report_text == (
        'batch completed: 1 of 1 video processed\n\n'
        'Counts\n'
        '- videos processed: 1/1\n'
        '- videos excluded before processing: 0/1\n'
        '- videos skipped: 0/1\n'
        '- rallies written: 1\n'
        '- contacts written: 1\n\n'
        'Video outcomes\n'
        '- vid: processed; 1 rally; 1 contact\n\n'
        'Rally exclusions\n'
        '- not recorded by this CLI; rally-level reasons exist only in the run_video chain\n'
    )
    assert spans_path.exists()
    assert contacts_path.exists()


def test_cli_uses_exception_name_when_processing_failure_has_no_message(
    tmp_path, monkeypatch,
):
    spans_path = tmp_path / 'custom' / 'rally_spans.csv'
    contacts_path = tmp_path / 'custom' / 'contact_frames.csv'
    spans_path.parent.mkdir()
    spans_path.write_text('stale rally output\n', encoding='utf-8')
    contacts_path.write_text('stale contact output\n', encoding='utf-8')

    def fail_without_message(*args, **kwargs):
        raise ValueError()

    with pytest.raises(RuntimeError, match='processed 0 of 1 video'):
        _run_cli(
            tmp_path,
            monkeypatch,
            extra_args=[],
            run_video=fail_without_message,
        )

    report_text = derive_batch_report_path(spans_path).read_text(encoding='utf-8')
    assert 'batch failed: 0 of 1 video processed\n' in report_text
    assert '- vid: skipped; ValueError\n' in report_text
    assert not spans_path.exists()
    assert not contacts_path.exists()


def test_all_excluded_publishes_before_raising_and_writes_no_output_csv(
    tmp_path, monkeypatch, capsys, write_doubles_flags,
):
    flags_path = tmp_path / 'doubles.csv'
    write_doubles_flags(flags_path, [('vid', '', 'True')])
    spans_path = tmp_path / 'spans.csv'
    contacts_path = tmp_path / 'contacts.csv'
    shuttle_dir = tmp_path / 'tracks'
    shuttle_dir.mkdir()
    np.save(shuttle_dir / 'vid.npy', np.zeros((4, 3)))

    from annotator import rally_segmentation
    import annotator.run_video as run_video_module

    def segment_must_not_run(*args, **kwargs):
        raise AssertionError('excluded videos must not reach segment_video')

    monkeypatch.setattr(run_video_module, 'run_video', segment_must_not_run)
    monkeypatch.setattr(sys, 'argv', [
        'rally_segmentation', '--shuttle-dir', str(shuttle_dir), '--fps', '30',
        '--doubles-csv', str(flags_path), '--rally-spans-csv', str(spans_path),
        '--contact-frames-csv', str(contacts_path),
    ])

    with pytest.raises(ValueError, match='excluded every video'):
        rally_segmentation.main()

    report_path = tmp_path / 'spans_batch_report.txt'
    report_text = report_path.read_text(encoding='utf-8')
    assert report_text.startswith('batch failed: all 1 video excluded\n')
    assert capsys.readouterr().out == report_text
    assert not spans_path.exists()
    assert not contacts_path.exists()


def test_all_excluded_report_failure_keeps_existing_value_error(
    tmp_path, monkeypatch, write_doubles_flags,
):
    flags_path = tmp_path / 'doubles.csv'
    write_doubles_flags(flags_path, [('vid', '', 'True')])
    spans_path = tmp_path / 'spans.csv'
    shuttle_dir = tmp_path / 'tracks'
    shuttle_dir.mkdir()
    np.save(shuttle_dir / 'vid.npy', np.zeros((4, 3)))

    from annotator import batch_report, rally_segmentation

    def publish_failure(*args, **kwargs):
        raise OSError('report destination unavailable')

    monkeypatch.setattr(batch_report, 'publish_batch_report', publish_failure)
    monkeypatch.setattr(sys, 'argv', [
        'rally_segmentation', '--shuttle-dir', str(shuttle_dir), '--fps', '30',
        '--doubles-csv', str(flags_path), '--rally-spans-csv', str(spans_path),
        '--contact-frames-csv', str(tmp_path / 'contacts.csv'),
    ])

    with pytest.raises(ValueError, match='excluded every video') as raised:
        rally_segmentation.main()

    assert isinstance(raised.value.__cause__, OSError)
    assert str(raised.value.__cause__) == 'report destination unavailable'
    assert not spans_path.exists()
