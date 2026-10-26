"""The exclusions use every original rally and keep independent reasons."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from scripts.build_rally_exclusions import build_exclusions, main


def write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    originals = tmp_path / 'set'
    originals.mkdir()
    pd.DataFrame([
        {'id': 1, 'video': 'first'},
        {'id': 2, 'video': 'second'},
    ]).to_csv(originals / 'match.csv', index=False)

    first = originals / 'first'
    first.mkdir()
    pd.DataFrame([
        # The file order differs from ball_round order. A hits twice in sequence.
        {'rally': 1, 'ball_round': 3, 'player': 'A', 'flaw': ''},
        {'rally': 1, 'ball_round': 1, 'player': 'B', 'flaw': '1'},
        {'rally': 1, 'ball_round': 2, 'player': 'A', 'flaw': ''},
        # An unknown player interrupts the pair of A contacts.
        {'rally': 2, 'ball_round': 1, 'player': 'A', 'flaw': ''},
        {'rally': 2, 'ball_round': 2, 'player': '', 'flaw': ''},
        {'rally': 2, 'ball_round': 3, 'player': 'A', 'flaw': ''},
        # This original rally is wholly absent from the master.
        {'rally': 3, 'ball_round': 1, 'player': 'A', 'flaw': ''},
        {'rally': 3, 'ball_round': 2, 'player': 'B', 'flaw': ''},
        # The lost contact has no flaw mark.
        {'rally': 4, 'ball_round': 1, 'player': 'A', 'flaw': ''},
        {'rally': 4, 'ball_round': 2, 'player': 'B', 'flaw': ''},
    ]).to_csv(first / 'set1.csv', index=False)

    second = originals / 'second'
    second.mkdir()
    pd.DataFrame([
        {'rally': 1, 'ball_round': 1, 'player': 'B', 'flaw': ''},
        {'rally': 1, 'ball_round': 2, 'player': 'B', 'flaw': ''},
    ]).to_csv(second / 'set2.csv', index=False)

    master = tmp_path / 'shots_master.csv'
    pd.DataFrame([
        {'vid': 1, 'set_id': 'set1', 'rally': 1},
        {'vid': 1, 'set_id': 'set1', 'rally': 1},
        {'vid': 1, 'set_id': 'set1', 'rally': 2},
        {'vid': 1, 'set_id': 'set1', 'rally': 2},
        {'vid': 1, 'set_id': 'set1', 'rally': 2},
        {'vid': 1, 'set_id': 'set1', 'rally': 4},
    ]).to_csv(master, index=False)
    return originals, master


def test_build_exclusions_preserves_all_original_rallies_and_reasons(tmp_path: Path) -> None:
    originals, master = write_fixture(tmp_path)

    actual = set(build_exclusions(originals, master).itertuples(index=False, name=None))

    assert actual == {
        (1, 1, 1, 'flaw'),
        (1, 1, 1, 'missing_master_rows'),
        (1, 1, 1, 'consecutive_same_player'),
        (1, 1, 3, 'missing_master_rows'),
        (1, 1, 4, 'missing_master_rows'),
        (2, 2, 1, 'missing_master_rows'),
        (2, 2, 1, 'consecutive_same_player'),
    }


def test_cli_writes_gzip_csv_with_requested_paths(tmp_path: Path, monkeypatch) -> None:
    originals, master = write_fixture(tmp_path)
    output = tmp_path / 'nested' / 'exclusions.csv.gz'
    monkeypatch.setattr(sys, 'argv', [
        'build_rally_exclusions', '--originals', str(originals),
        '--master', str(master), '--output', str(output),
    ])

    main()

    assert output.read_bytes().startswith(b'\x1f\x8b')
    written = pd.read_csv(output)
    assert list(written.columns) == ['vid', 'set', 'rally', 'reason']
    assert len(written) == 7
