"""Contract tests for the annotator migration fixture manifest."""

import os

import pytest

from annotator.calibration.fixtures import (
    FIXTURES,
    REPO_ROOT,
    SHARED_FILES,
    verify_file,
    verify_fixture,
)


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda fixture: fixture.name)
def test_fixture_files_are_present_and_pinned(fixture):
    if not os.environ.get("ANNOTATOR_FIXTURES_ROOT"):
        pytest.skip("ANNOTATOR_FIXTURES_ROOT is unset; external fixtures are unavailable")
    verify_fixture(fixture)


@pytest.mark.parametrize("pin", SHARED_FILES, ids=lambda pin: str(pin.path))
def test_shared_files_are_present_and_pinned(pin):
    if pin.root == "fixtures" and not os.environ.get("ANNOTATOR_FIXTURES_ROOT"):
        pytest.skip("ANNOTATOR_FIXTURES_ROOT is unset; external fixtures are unavailable")
    verify_file(pin)


def test_gt_repo_paths_exist():
    for fixture in FIXTURES:
        assert (REPO_ROOT / fixture.gt_set_dir).is_dir(), fixture.gt_set_dir
    for pin in SHARED_FILES:
        if pin.root == "repo":
            assert (REPO_ROOT / pin.path).is_file(), pin.path
