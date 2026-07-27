"""Focused contracts for the manifest calibration CLI."""

import os
from contextlib import contextmanager

import pytest

from annotator.calibration import run_cli
from annotator.calibration.fixtures import PILOT


@contextmanager
def fixture_root(path):
    prior = os.environ.get("ANNOTATOR_FIXTURES_ROOT")
    os.environ["ANNOTATOR_FIXTURES_ROOT"] = str(path)
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("ANNOTATOR_FIXTURES_ROOT", None)
        else:
            os.environ["ANNOTATOR_FIXTURES_ROOT"] = prior


def test_parser_has_only_manifest_and_output_controls():
    args = run_cli.build_parser().parse_args([])
    assert args.fixtures is None
    assert args.out is None
    assert args.no_replay_mask is False
    with pytest.raises(SystemExit) as excinfo:
        run_cli.build_parser().parse_args(["--fps", "25"])
    assert excinfo.value.code == 2


def test_selected_fixture_writes_flat_metrics_row(tmp_path, capsys):
    scoring = object()
    with fixture_root(tmp_path):
        assert run_cli.run_manifest(
            ["pilot"], tmp_path / "out", registry=(PILOT,), runner=lambda _: scoring,
            flattener=lambda _: {"covered": 2, "f1": 0.5},
            renderer=lambda scores: f"TABLE {sorted(scores)}",
        ) == 0
    assert (tmp_path / "out" / "pilot_metrics.csv").read_text() == "covered,f1\n2,0.5\n"
    assert "TABLE ['pilot']" in capsys.readouterr().out


def test_maskless_switch_reaches_fixture_runner(tmp_path):
    received = []

    def runner(fixture, *, no_replay_mask=False):
        received.append((fixture.name, no_replay_mask))
        return object()

    with fixture_root(tmp_path):
        assert run_cli.run_manifest(
            ['pilot'], registry=(PILOT,), runner=runner,
            flattener=lambda _: {'covered': 1}, renderer=lambda scores: 'TABLE',
            no_replay_mask=True,
        ) == 0

    assert received == [('pilot', True)]


def test_fixture_failure_is_skipped_and_next_fixture_runs(tmp_path, capsys):
    second = PILOT.__class__(**{**PILOT.__dict__, "name": "second"})
    calls = []

    def runner(fixture):
        calls.append(fixture.name)
        if fixture.name == "pilot":
            raise ValueError("bad digest")
        return object()

    with fixture_root(tmp_path):
        assert run_cli.run_manifest(
            registry=(PILOT, second), runner=runner,
            flattener=lambda _: {"covered": 1}, renderer=lambda scores: "TABLE",
        ) == 3
    assert calls == ["pilot", "second"]
    assert "pilot: SKIP" in capsys.readouterr().err


def test_one_fixture_failure_in_three_returns_usable_output(tmp_path):
    second = PILOT.__class__(**{**PILOT.__dict__, "name": "second"})
    third = PILOT.__class__(**{**PILOT.__dict__, "name": "third"})

    def runner(fixture):
        if fixture.name == "pilot":
            raise ValueError("bad digest")
        return object()

    with fixture_root(tmp_path):
        assert run_cli.run_manifest(
            registry=(PILOT, second, third), runner=runner,
            flattener=lambda _: {"covered": 1}, renderer=lambda scores: "TABLE",
        ) == 0


def test_every_fixture_failure_returns_three(tmp_path):
    second = PILOT.__class__(**{**PILOT.__dict__, "name": "second"})

    def runner(_):
        raise ValueError("bad fixture")

    with fixture_root(tmp_path):
        assert run_cli.run_manifest(
            registry=(PILOT, second), runner=runner,
            flattener=lambda _: {"covered": 1}, renderer=lambda scores: "TABLE",
        ) == 3


def test_environment_and_registry_fail_before_runner(tmp_path):
    prior = os.environ.pop("ANNOTATOR_FIXTURES_ROOT", None)
    try:
        with pytest.raises(RuntimeError, match="ANNOTATOR_FIXTURES_ROOT"):
            run_cli.run_manifest(registry=(PILOT,), runner=lambda _: pytest.fail("not called"))

        with fixture_root(tmp_path):
            with pytest.raises(ValueError, match="malformed"):
                run_cli.run_manifest(registry=(object(),), runner=lambda _: pytest.fail("not called"))
    finally:
        if prior is not None:
            os.environ["ANNOTATOR_FIXTURES_ROOT"] = prior
