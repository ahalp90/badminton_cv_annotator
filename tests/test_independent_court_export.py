"""Source selection at the optional neural inference boundary."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from experiments.annotator.independent_court import export_lines


def test_requested_model_source_takes_priority_when_already_on_path(tmp_path: Path, monkeypatch) -> None:
    competing = tmp_path / 'competing'
    requested = tmp_path / 'requested'
    for directory in (competing, requested):
        directory.mkdir()
        (directory / 'court_export_source_probe.py').write_text('SOURCE = __file__\n')
    monkeypatch.setattr(sys, 'path', [str(competing), str(requested), *sys.path])
    export_lines._add_source_path(requested)
    try:
        module = importlib.import_module('court_export_source_probe')
        assert Path(module.SOURCE).parent == requested
    finally:
        sys.modules.pop('court_export_source_probe', None)
