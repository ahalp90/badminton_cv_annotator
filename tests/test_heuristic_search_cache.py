"""Check durable sticky evidence reuse and restoration of the production helper."""

from __future__ import annotations

import gzip
import multiprocessing
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from annotator.rally import evidence as rally_evidence
from annotator.types import StickyResult
from experiments.annotator.heuristic_only_tune.cache import (
    StickyCacheError,
    cache_sticky_result,
)


def _result(value: float) -> StickyResult:
    return StickyResult(
        np.full(3, value), np.zeros((3, 2), dtype=int), np.full(3, 2, dtype=int),
        np.full((3, 2, 2), value), np.full((3, 2), 100.0), np.full((3, 2), value),
        np.full((3, 2), value), np.ones(3, dtype=bool),
    )


def _build(gate_video_id: str, half_window: int = 1) -> StickyResult:
    return _result(float(half_window))


def _process_job(cache_dir: Path, barrier: Any, results: Any) -> None:
    barrier.wait(timeout=10)
    with cache_sticky_result(cache_dir, 'video'):
        result = rally_evidence.build_sticky_result(gate_video_id='video', half_window=2)
        results.put(float(result.distances[0]))


def test_cache_reuses_saved_arrays_after_the_context_ends(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def build(gate_video_id: str, half_window: int = 1) -> StickyResult:
        calls.append((gate_video_id, half_window))
        return _result(7.0)

    monkeypatch.setattr(rally_evidence, 'build_sticky_result', build)
    with cache_sticky_result(tmp_path, 'video'):
        first = rally_evidence.build_sticky_result('video')
    assert rally_evidence.build_sticky_result is build
    with cache_sticky_result(tmp_path, 'video'):
        second = rally_evidence.build_sticky_result(gate_video_id='video', half_window=1)
    assert calls == [('video', 1)]
    assert second is not first
    for computed, loaded in zip(first, second, strict=True):
        np.testing.assert_array_equal(computed, loaded)
        assert not computed.flags.writeable
        assert not loaded.flags.writeable
    with pytest.raises(ValueError, match='read-only'):
        second.distances[0] = 100.0


def test_video_and_resolved_window_each_get_separate_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def build(gate_video_id: str, half_window: int = 1) -> StickyResult:
        calls.append((gate_video_id, half_window))
        return _result(float(half_window))

    monkeypatch.setattr(rally_evidence, 'build_sticky_result', build)
    for video in ('first', 'second'):
        with cache_sticky_result(tmp_path, video):
            for window in (1, 2, 1):
                result = rally_evidence.build_sticky_result(video, window)
                np.testing.assert_array_equal(result.distances, np.full(3, window))
    assert calls == [('first', 1), ('first', 2), ('second', 1), ('second', 2)]
    assert len(list(tmp_path.rglob('*.pkl.gz'))) == 4


def test_builder_exception_restores_helper_and_releases_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(gate_video_id: str, half_window: int = 1) -> StickyResult:
        raise RuntimeError('tracking failed')

    monkeypatch.setattr(rally_evidence, 'build_sticky_result', fail)
    with pytest.raises(RuntimeError, match='tracking failed'), cache_sticky_result(tmp_path, 'video'):
        rally_evidence.build_sticky_result('video')
    assert rally_evidence.build_sticky_result is fail
    assert not list(tmp_path.rglob('*.pkl.gz'))
    monkeypatch.setattr(rally_evidence, 'build_sticky_result', _build)
    with cache_sticky_result(tmp_path, 'video'):
        rally_evidence.build_sticky_result('video')


def test_exception_in_annotation_also_restores_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rally_evidence, 'build_sticky_result', _build)
    with pytest.raises(RuntimeError, match='annotation failed'), cache_sticky_result(tmp_path, 'video'):
        rally_evidence.build_sticky_result('video')
        raise RuntimeError('annotation failed')
    assert rally_evidence.build_sticky_result is _build


def test_cache_rejects_a_different_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rally_evidence, 'build_sticky_result', _build)
    with cache_sticky_result(tmp_path, 'first'), pytest.raises(StickyCacheError, match='video differs'):
        rally_evidence.build_sticky_result('second')


@pytest.mark.parametrize('invalid', ['wrong_type', 'truncated', 'deflate'])
def test_corrupt_or_wrong_cache_payload_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str,
) -> None:
    path = tmp_path / 'video' / 'window-1.pkl.gz'
    path.parent.mkdir()
    if invalid == 'wrong_type':
        with gzip.open(path, 'wb') as handle:
            pickle.dump({'wrong': 'result'}, handle)
    elif invalid == 'deflate':
        compressed = bytearray(gzip.compress(pickle.dumps(_result(1.0))))
        compressed[10:14] = b'\xff\xff\xff\xff'
        path.write_bytes(compressed)
    else:
        path.write_bytes(b'incomplete gzip')
    monkeypatch.setattr(rally_evidence, 'build_sticky_result', _build)
    with pytest.raises(StickyCacheError), cache_sticky_result(tmp_path, 'video'):
        rally_evidence.build_sticky_result('video')
    assert rally_evidence.build_sticky_result is _build


def test_failed_write_cleans_up_and_raises_cache_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError('disk full')

    monkeypatch.setattr(pickle, 'dump', fail)
    monkeypatch.setattr(rally_evidence, 'build_sticky_result', _build)
    with pytest.raises(StickyCacheError, match='disk full'), cache_sticky_result(tmp_path, 'video'):
        rally_evidence.build_sticky_result('video')
    assert rally_evidence.build_sticky_result is _build
    assert not list(tmp_path.rglob('*.pkl.gz'))
    assert not list(tmp_path.rglob('*.tmp'))


def test_unwritable_cache_path_raises_cache_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_path = tmp_path / 'file'
    cache_path.write_text('a file cannot contain cache directories')
    monkeypatch.setattr(rally_evidence, 'build_sticky_result', _build)
    with (
        pytest.raises(StickyCacheError, match='Cannot open sticky cache lock'),
        cache_sticky_result(cache_path, 'video'),
    ):
        rally_evidence.build_sticky_result('video')


def test_two_processes_build_one_shared_cache_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    counter = tmp_path / 'builds.txt'

    def build(gate_video_id: str, half_window: int = 1) -> StickyResult:
        with counter.open('a') as handle:
            handle.write('built\n')
        time.sleep(0.1)
        return _result(float(half_window))

    monkeypatch.setattr(rally_evidence, 'build_sticky_result', build)
    context = multiprocessing.get_context('fork')
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [context.Process(target=_process_job, args=(tmp_path, barrier, results)) for _index in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail('sticky cache process did not finish')
        assert process.exitcode == 0
    assert [results.get(timeout=2), results.get(timeout=2)] == [2.0, 2.0]
    results.close()
    results.join_thread()
    assert counter.read_text() == 'built\n'
    assert not list(tmp_path.rglob('*.tmp'))
