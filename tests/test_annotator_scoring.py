"""Regression floors for the full annotator GT scoring harness."""
import os

import pytest

from annotator.calibration.fixtures import FIXTURES
from annotator.calibration.gt_scoring import assert_floors, flatten_metrics, render_table, run_fixture


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda fixture: fixture.name)
def test_annotator_gt_floors(fixture):
    if not os.environ.get("ANNOTATOR_FIXTURES_ROOT"):
        pytest.skip("ANNOTATOR_FIXTURES_ROOT is unset; external fixtures are unavailable")
    metrics = flatten_metrics(run_fixture(fixture))
    print(render_table({fixture.name: metrics}))
    assert_floors(fixture, metrics)
