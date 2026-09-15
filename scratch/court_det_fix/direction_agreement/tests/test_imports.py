"""Every stage script must import cleanly; the synthetic tests do not exercise their entry points."""

import importlib

import pytest


@pytest.mark.parametrize('module', ['common', 'membership', 'selection', 'run_arms', 'run_fits', 'run_matcher',
                                    'diagnose_matrix', 'manifest'])
def test_module_imports(module: str) -> None:
    assert importlib.import_module(module).__doc__
