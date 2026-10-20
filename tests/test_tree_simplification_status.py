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
    running = tmp_path / 'running'
    write_json(failed / 'result.json.gz', result(False, 1))
    write_json(failed / 'status.json.gz', {'name': 'failed', 'exit_code': 1})
    write_json(failed / 'progress.json.gz', {'stage': 'fit_layers', 'totals': {}})
    write_json(completed / 'result.json.gz', result(True, 2))
    write_json(completed / 'status.json.gz', {'name': 'completed', 'exit_code': 0})
    write_json(running / 'status.json.gz', {'state': 'running', 'command': ['python', '-m', 'train']})
    write_json(tmp_path / 'status.json.gz', {
        'planned': 3,
        'completed': [
            {'name': 'failed', 'exit_code': 1},
            {'name': 'completed', 'exit_code': 0},
            {'name': 'running', 'exit_code': 9},
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
    assert 'running: running' in output
    assert 'failed: fit_layers' in output
