import gzip
import json
from pathlib import Path

import pytest

from experiments.annotator.tree_simplification import status


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt') as stream:
        json.dump(value, stream)


def result(complete: bool, correct: int) -> dict[str, object]:
    scores = {
        'fully_correct_rallies': correct,
        'labelled_rallies': 2,
        'fully_correct_rallies_percent': 50.0 * correct,
        'attributed_f1': 0.5,
    }
    return {'complete': complete, 'layers': {'0': {'10': scores}}}


def test_status_marks_partial_results_and_reports_config_exit_and_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    failed = tmp_path / 'failed'
    completed = tmp_path / 'completed'
    write_json(failed / 'result.json.gz', result(False, 1))
    write_json(failed / 'status.json.gz', {'name': 'failed', 'exit_code': 1})
    write_json(failed / 'progress.json.gz', {'stage': 'fit_layers', 'totals': {}})
    write_json(completed / 'result.json.gz', result(True, 2))
    write_json(completed / 'status.json.gz', {'name': 'completed', 'exit_code': 0})
    write_json(tmp_path / 'status.json.gz', {
        'planned': 2,
        'completed': [
            {'name': 'failed', 'exit_code': 1},
            {'name': 'completed', 'exit_code': 0},
        ],
    })
    monkeypatch.setattr('sys.argv', ['status.py', str(tmp_path)])

    status.main()

    output = capsys.readouterr().out
    assert 'Validation, tolerance 10 frames at 30 fps' in output
    assert '| failed | 1 | 1/2 | 50.00% | 50.00% | partial |' in output
    assert '| completed | 1 | 2/2 | 100.00% | 50.00% | complete |' in output
    assert 'failed: failed (exit 1)' in output
    assert output.count('failed (exit 1)') == 1
    assert 'completed: complete (exit 0)' in output
    assert f'{tmp_path.name}: finished (2/2; 1 failed)' in output
    assert 'failed: fit_layers' in output


def test_nested_search_aggregate_stays_running_until_every_config_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    search = tmp_path / 'search-xgb'
    write_json(search / 'status.json.gz', {
        'planned': 5,
        'completed': [{'name': 'leaves-15', 'exit_code': 0}],
    })
    write_json(search / 'leaves-15' / 'result.json.gz', result(True, 2))
    write_json(search / 'leaves-15' / 'status.json.gz', {'name': 'leaves-15', 'exit_code': 0})
    write_json(search / 'leaves-7' / 'status.json.gz', {'state': 'running', 'command': ['python', '-m', 'train']})
    monkeypatch.setattr('sys.argv', ['status.py', str(tmp_path)])

    status.main()

    output = capsys.readouterr().out
    assert 'search-xgb: running (1/5 configs returned)' in output
    assert 'search-xgb: unknown' not in output
    assert 'search-xgb/leaves-15: complete (exit 0)' in output
    assert 'search-xgb/leaves-7: running' in output
