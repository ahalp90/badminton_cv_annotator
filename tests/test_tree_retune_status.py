"""Read-only status checks using a small saved run and a fixed process tree."""

import gzip
import json
from pathlib import Path

import pytest

from scripts import tree_retune_status as status


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'wt') as stream:
        json.dump(payload, stream)


def candidate(label: str, value: int) -> dict:
    return {'label': label, 'overrides_base30': {'smooth_window': value}, 'strategies': {}}


def result(row: dict, video_count: int) -> dict:
    counts = dict.fromkeys(status.COUNT_FIELDS, 0)
    scores = {'5': counts, '10': counts}
    return {
        'id': status.candidate_identity(row), 'label': row['label'], 'totals': scores,
        'per_video': {str(index): scores for index in range(video_count)},
    }


def process(pid: int, parent: int, command: str, state: str = 'S') -> dict:
    return {
        'pid': pid, 'ppid': parent, 'state': state, 'cpu_percent': 98.0,
        'rss_kib': 1024, 'elapsed_s': 600, 'command': command,
    }


def test_worker_tree_excludes_stopped_supervisor_and_unowned_workers() -> None:
    rows = [
        process(10, 1, 'bash batch1'), process(11, 10, 'timeout 3h python', 'T'),
        process(12, 11, 'python -m retune'), process(13, 12, 'python spawn_main(...)'),
        process(14, 12, 'python spawn_main(...)', 'T'), process(15, 12, 'python resource_tracker'),
        process(16, 1, 'python spawn_main(...)'),
    ]
    workers = status.owned_workers(rows, {10})
    assert [worker['pid'] for worker in workers] == [13, 14]
    assert [worker['status'] for worker in workers] == ['running', 'paused']


def test_saved_results_are_authoritative_and_progress_is_separate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [candidate('done', 1), candidate('working', 2), candidate('queued', 3)]
    output = tmp_path / 'grid48'
    write_json(output / 'snapshot.json.gz', {'candidates': {'candidates': rows}})
    write_json(output / 'training.json.gz', {'complete': 99})
    write_json(tmp_path / 'automation.json', {'phase': 'training', 'training_pid': 20, 'status': 'running'})
    for row in rows[:2]:
        trial = output / 'trials' / status.candidate_identity(row)
        if row['label'] == 'done':
            write_json(trial / 'result.json.gz', result(row, 32))
        else:
            write_json(trial / 'progress.json.gz', {'stage': 'downstream_fit'})
    monkeypatch.setattr(status, 'process_rows', lambda: [process(21, 20, 'python spawn_main(...)')])
    monkeypatch.setattr(status, 'available_memory_kib', lambda: 1024**2)
    report = status.collect_status(tmp_path)
    assert report['training'] == {'complete': 1, 'planned': 3}
    assert report['unfinished_trials'][0]['label'] == 'working'
    assert report['queued_labels'] == ['queued']
    assert report['workers'][0]['pid'] == 21
    assert report['validation'] == {'complete': 0, 'planned': 0}


def test_validation_completion_and_malformed_result_fail_loudly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    row = candidate('shipped', 3)
    identity = status.candidate_identity(row)
    output = tmp_path / 'grid48'
    write_json(output / 'snapshot.json.gz', {'candidates': {'candidates': [row]}})
    write_json(output / 'shortlist.json.gz', {'ids': [identity]})
    write_json(tmp_path / 'automation.json', {'phase': 'validation', 'initial_training_success': True})
    (tmp_path / 'batch1.exit').write_text('137\n')
    trial = output / 'trials' / identity
    write_json(trial / 'result.json.gz', result(row, 32))
    monkeypatch.setattr(status, 'process_rows', list)
    monkeypatch.setattr(status, 'available_memory_kib', lambda: 1024**2)
    report = status.collect_status(tmp_path)
    assert report['phase'] == 'validation'
    assert report['unfinished_trials'][0]['stage'] == 'validation'
    write_json(trial / 'validation.json.gz', result(row, 8))
    assert status.collect_status(tmp_path)['complete']
    write_json(trial / 'result.json.gz', {'id': identity, 'label': row['label']})
    with pytest.raises(ValueError, match='32 videos'):
        status.collect_status(tmp_path)


def test_missing_counts_are_not_a_completed_result(tmp_path: Path) -> None:
    row = candidate('shipped', 3)
    saved = result(row, 32)
    saved['totals']['10'] = {}
    path = tmp_path / 'result.json.gz'
    write_json(path, saved)
    with pytest.raises(ValueError, match='invalid labelled_rallies count'):
        status.result_exists(path, status.candidate_identity(row), row['label'], 32)


def test_memavailable_parser(tmp_path: Path) -> None:
    path = tmp_path / 'meminfo'
    path.write_text('MemTotal: 10000 kB\nMemAvailable: 7000 kB\n')
    assert status.available_memory_kib(path) == 7000


def test_missing_guard_is_visible_without_hiding_workers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    row = candidate('shipped', 3)
    write_json(tmp_path / 'grid48/snapshot.json.gz', {'candidates': {'candidates': [row]}})
    write_json(tmp_path / 'automation.json', {
        'phase': 'training', 'status': 'running', 'training_pid': 20, 'guard_pid': 30,
    })
    worker = process(21, 20, 'python spawn_main(...)')
    monkeypatch.setattr(status, 'available_memory_kib', lambda: 1024**2)
    monkeypatch.setattr(status, 'process_rows', lambda: [worker, process(30, 1, 'finish guard')])
    assert status.collect_status(tmp_path)['phase'] == 'training'
    monkeypatch.setattr(status, 'process_rows', lambda: [worker])
    report = status.collect_status(tmp_path)
    assert report['phase'] == 'failed'
    assert report['error'] == 'automation guard stopped'
    assert report['workers'][0]['pid'] == 21


def test_process_listing_requests_unlimited_command_width(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def check_output(command: list[str], *, text: bool) -> str:
        seen.append(command)
        return '21 20 S 99.0 1024 60 python spawn_main(...)\n'

    monkeypatch.setattr(status.subprocess, 'check_output', check_output)
    assert status.process_rows()[0]['pid'] == 21
    assert '-ww' in seen[0]


def test_repeat_phase_keeps_grid_counts_and_reports_seed_trials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    row = candidate('shipped', 3)
    identity = status.candidate_identity(row)
    write_json(tmp_path / 'grid48/snapshot.json.gz', {'candidates': {'candidates': [row]}})
    trial = tmp_path / 'grid48/trials' / identity
    write_json(trial / 'result.json.gz', result(row, 32))
    write_json(trial / 'validation.json.gz', result(row, 8))
    write_json(tmp_path / 'automation.json', {'phase': 'seed repeats', 'status': 'running'})
    write_json(tmp_path / 'repeats/plan.json.gz', {'candidates': [{'id': identity, **row}], 'seed_offsets': [0, 100, 200]})
    write_json(tmp_path / 'repeats/progress.json.gz', {
        'phase': 'repeat_training', 'fits_complete': 1, 'fits_planned': 3,
        'fresh_fits_complete': 0, 'fresh_fits_planned': 2, 'validation_complete': 1, 'validation_planned': 3,
    })
    write_json(tmp_path / 'repeats/seed-100/trials' / identity / 'progress.json.gz', {'stage': 'downstream_fit'})
    monkeypatch.setattr(status, 'process_rows', list)
    monkeypatch.setattr(status, 'available_memory_kib', lambda: 1024**2)
    report = status.collect_status(tmp_path)
    assert report['training'] == {'complete': 1, 'planned': 1}
    assert report['validation'] == {'complete': 1, 'planned': 1}
    assert not report['complete']
    assert report['phase'] == 'seed repeats'
    assert report['repeats']['fits_complete'] == 1
    assert report['repeats']['unfinished_trials'][0]['label'] == 'shipped seed+100'
    assert report['repeats']['queued_labels'] == ['shipped seed+200']


def test_final_test_progress_and_seed_selection(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    write_json(tmp_path / 'repeats/progress.json.gz', {'phase': 'complete'})
    write_json(tmp_path / 'repeats/selected.json.gz', {'config_id': 'one', 'label': 'chosen'})
    repeats = status.repeat_status(tmp_path, 100)
    assert repeats is not None and repeats['selected_label'] == 'chosen'
    write_json(tmp_path / 'final_test/progress.json.gz', {'phase': 'final_test', 'complete': 1, 'planned': 2})
    write_json(tmp_path / 'final_test/plan.json.gz', {
        'candidates': [{'id': 'one', 'label': 'chosen', 'bundles': [{'seed_offset': 0}, {'seed_offset': 100}]}],
    })
    finished = result(candidate('chosen', 3), 46)
    finished['id'] = 'one'
    write_json(tmp_path / 'final_test/trials/one/seed-0/result.json.gz', finished)
    write_json(tmp_path / 'final_test/trials/one/seed-100/progress.json.gz', {'stage': 'test', 'video': '8'})
    final_test = status.final_test_status(tmp_path, 100)
    assert final_test is not None and final_test['complete'] == 1
    assert final_test['unfinished_trials'][0]['label'] == 'final test chosen seed+100'
    status.print_status({
        'utc': 'now', 'phase': 'final test', 'elapsed_s': 10, 'batch_started_utc': 'before',
        'training': {'complete': 48, 'planned': 48}, 'validation': {'complete': 6, 'planned': 6},
        'repeats': repeats, 'final_test': final_test, 'workers': [], 'memory_available_kib': 1024**2,
        'error': None, 'failure': None, 'unfinished_trials': [], 'queued_labels': [], 'automation': {},
    })
    output = capsys.readouterr().out
    assert 'Selected configuration: chosen' in output
    assert 'Final test: final_test | 1/2' in output
    assert 'final test chosen seed+100: test 8' in output


def test_repeat_protocol_needs_guard_completion_and_finished_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    row = candidate('shipped', 3)
    identity = status.candidate_identity(row)
    write_json(tmp_path / 'grid48/snapshot.json.gz', {'candidates': {'candidates': [row]}})
    trial = tmp_path / 'grid48/trials' / identity
    write_json(trial / 'result.json.gz', result(row, 32))
    write_json(trial / 'validation.json.gz', result(row, 8))
    write_json(tmp_path / 'repeats/plan.json.gz', {'candidates': [{'id': identity, **row}], 'seed_offsets': [0]})
    repeat_progress = {
        'phase': 'complete', 'fits_complete': 1, 'fits_planned': 1,
        'fresh_fits_complete': 0, 'fresh_fits_planned': 0, 'validation_complete': 1, 'validation_planned': 1,
    }
    write_json(tmp_path / 'repeats/progress.json.gz', repeat_progress)
    write_json(tmp_path / 'repeats/selected.json.gz', {'config_id': identity, 'label': row['label']})
    write_json(tmp_path / 'final_test/progress.json.gz', {'phase': 'complete', 'complete': 1, 'planned': 1})
    monkeypatch.setattr(status, 'process_rows', list)
    monkeypatch.setattr(status, 'available_memory_kib', lambda: 1024**2)
    write_json(tmp_path / 'automation.json', {'phase': 'final test', 'status': 'running'})
    assert not status.collect_status(tmp_path)['complete']
    write_json(tmp_path / 'automation.json', {'phase': 'complete', 'status': 'complete'})
    assert status.collect_status(tmp_path)['complete']
    write_json(tmp_path / 'repeats/progress.json.gz', {**repeat_progress, 'validation_complete': 0})
    report = status.collect_status(tmp_path)
    assert not report['complete']
    assert report['phase'] == 'incomplete saved results'


def test_memory_pause_flags_show_alerts_until_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    write_json(tmp_path / 'grid48/snapshot.json.gz', {'candidates': {'candidates': [candidate('shipped', 3)]}})
    automation = {'phase': 'training', 'status': 'running', 'memory_emergency': True, 'memory_stalled': True}
    write_json(tmp_path / 'automation.json', automation)
    monkeypatch.setattr(status, 'process_rows', list)
    monkeypatch.setattr(status, 'available_memory_kib', lambda: 5 * 1024**2)
    report = status.collect_status(tmp_path)
    assert len(report['alerts']) == 2
    status.print_status(report)
    assert 'watcher action needed' in capsys.readouterr().out
    write_json(tmp_path / 'automation.json', {**automation, 'status': 'complete'})
    assert status.collect_status(tmp_path)['alerts'] == []
