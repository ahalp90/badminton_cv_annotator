"""Reuse sticky player evidence inside one immutable search output directory.

The runner verifies source and input fingerprints before using this directory.
Each video's raw track, pose, tracker segments and scene geometry then stay fixed.
Only the resolved body-height window changes the sticky helper's result across
candidates. Cache files belong to this run and are never imported from elsewhere.
"""

from __future__ import annotations

import fcntl
import gzip
import inspect
import pickle
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import numpy as np

from annotator.rally import evidence as rally_evidence
from annotator.types import StickyResult


class StickyCacheError(RuntimeError):
    """A cache failure that must abort the search rather than count as bad inference."""


def _readonly(result: Any, path: Path) -> StickyResult:
    if not isinstance(result, StickyResult) or any(not isinstance(values, np.ndarray) for values in result):
        raise StickyCacheError(f'{path}: cache does not contain StickyResult arrays')
    for values in result:
        values.flags.writeable = False
    return result


def _read_result(path: Path) -> StickyResult:
    try:
        with gzip.open(path, 'rb') as handle:
            result = pickle.load(handle)
    except (OSError, EOFError, pickle.PickleError, ValueError, AttributeError, ImportError, zlib.error) as error:
        raise StickyCacheError(f'Cannot read sticky cache {path}: {error}') from error
    return _readonly(result, path)


def _write_result(path: Path, result: StickyResult) -> None:
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        try:
            with gzip.open(temporary, 'wb', compresslevel=1) as handle:
                pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    except (OSError, pickle.PickleError) as error:
        raise StickyCacheError(f'Cannot save sticky cache {path}: {error}') from error


@contextmanager
def _cache_lock(path: Path) -> Iterator[None]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.with_suffix('.lock').open('a+b')
    except OSError as error:
        raise StickyCacheError(f'Cannot open sticky cache lock for {path}: {error}') from error
    # Closing the descriptor releases the process lock, including on builder failure.
    with handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
        except OSError as error:
            raise StickyCacheError(f'Cannot lock sticky cache {path}: {error}') from error
        yield


@contextmanager
def cache_sticky_result(cache_dir: Path, video_id: str) -> Iterator[None]:
    """Cache the sticky producer used by annotation and direct preparation.

    Each worker runs one candidate at a time. The patch is process-local and
    always restored. Candidate-dependent masks are applied after sticky evidence
    is built from the original track, so they do not belong in this cache key.

    :param cache_dir: Cache directory beneath the validated, immutable run output.
    :param video_id: The fixed video's identity, as passed to the sticky helper.
    """
    original = rally_evidence.build_sticky_result
    signature = inspect.signature(original)

    @wraps(original)
    def cached(*args: Any, **kwargs: Any) -> StickyResult:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if bound.arguments['gate_video_id'] != video_id:
            raise StickyCacheError('sticky cache video differs from the helper input')
        half_window = bound.arguments['half_window']
        path = Path(cache_dir) / video_id / f'window-{half_window}.pkl.gz'
        with _cache_lock(path):
            if path.exists():
                return _read_result(path)
            result = original(*args, **kwargs)
            result = _readonly(result, path)
            _write_result(path, result)
            return result

    with patch.object(rally_evidence, 'build_sticky_result', cached):
        yield
