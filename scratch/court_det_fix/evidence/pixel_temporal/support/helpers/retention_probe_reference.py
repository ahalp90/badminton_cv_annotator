"""One deliberately simple, fixed-budget admission hypothesis; NOT a detector fix.

Keep the first `core` score-ordered distinct assignments, then distribute the
remaining slots across quantile cells of rectified court centre and log span.
Signed scale branches remain separate. There are no reference/control inputs.
The caller must supply the ORIGINAL eligible, score-ordered distinct indexes.
Run `python retention_probe_reference.py --self-test` for synthetic checks.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import numpy as np


def select_score_plus_strata(
    parameters: np.ndarray,
    scores: np.ndarray,
    distinct: np.ndarray,
    coordinates: np.ndarray,
    *, keep: int = 512, core: int = 256, bins: int = 16,
) -> np.ndarray:
    """Return original enumeration indexes, in original score order.

    Only admission after original duplicate removal changes. Original support,
    player pruning, duplicate representative choice and directions stay intact.
    Quantile bins are a development hypothesis in the matcher's fixed chart;
    they are not a physical uncertainty model or an acceptance threshold.
    """
    p = np.asarray(parameters, dtype=float)
    s = np.asarray(scores, dtype=float)
    raw_d = np.asarray(distinct)
    q = np.asarray(coordinates, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or s.shape != (len(p),):
        raise ValueError('parameters must be (N,2), scores (N,)')
    if raw_d.ndim != 1 or raw_d.dtype.kind not in 'iu':
        raise ValueError('distinct must be a 1D integer array')
    d = raw_d.astype(np.int64, copy=False)
    if not (0 <= core <= keep) or keep < 1 or bins < 1:
        raise ValueError('require keep>=1, 0<=core<=keep, bins>=1')
    if q.ndim != 1 or len(q) < 2 or not np.isfinite(q).all() or np.ptp(q) <= 0:
        raise ValueError('coordinates must have finite positive extent')
    if len(d) != len(np.unique(d)) or (len(d) and (d.min() < 0 or d.max() >= len(p))):
        raise ValueError('distinct indexes must be unique and in bounds')
    if not np.isfinite(p[d]).all() or not np.isfinite(s[d]).all() or np.any(p[d, 0] == 0):
        raise ValueError('eligible assignments must have finite values and nonzero scales')
    if np.any(np.diff(s[d]) > 0):
        raise ValueError('distinct must already be ordered by descending original score')
    if len(d) <= keep:
        return d.copy()

    centre = p[d, 1] + p[d, 0] * ((q.min() + q.max()) / 2.0)
    # log(abs(scale)) + log(extent) avoids multiplication overflow.
    log_span = np.log(np.abs(p[d, 0])) + np.log(np.ptp(q))
    if not np.isfinite(centre).all() or not np.isfinite(log_span).all():
        raise ValueError('nonfinite descriptor; do not silently drop assignments')
    sign = np.sign(p[d, 0]).astype(int)
    centre_bin = np.zeros(len(d), dtype=int)
    span_bin = np.zeros(len(d), dtype=int)
    fractions = np.arange(1, bins, dtype=float) / bins
    for branch in (-1, 1):
        ix = np.flatnonzero(sign == branch)
        if not len(ix):
            continue
        for values, labels in ((centre, centre_bin), (log_span, span_bin)):
            edges = np.quantile(values[ix], fractions, method='linear')
            labels[ix] = np.searchsorted(edges, values[ix], side='right')

    # Buckets use baseline score positions, not renumbered enumeration IDs.
    buckets: dict[tuple[int, int, int], deque[int]] = defaultdict(deque)
    for pos in range(core, len(d)):
        buckets[(int(sign[pos]), int(centre_bin[pos]), int(span_bin[pos]))].append(pos)
    # Visit strongest currently unrepresented cell first; then fixed round-robin.
    order = sorted(buckets, key=lambda key: buckets[key][0])
    selected = list(range(core))
    active = order
    while len(selected) < keep:
        next_active = []
        for key in active:
            selected.append(buckets[key].popleft())
            if buckets[key]:
                next_active.append(key)
            if len(selected) == keep:
                break
        if len(selected) < keep and not next_active:
            raise RuntimeError('admission exhausted despite enough distinct assignments')
        active = next_active
    return d[np.sort(np.asarray(selected, dtype=int))]


def self_test() -> None:
    rng = np.random.default_rng(20260916)
    for n in (0, 2, 511, 512, 513, 1000, 10000):
        p = np.column_stack((rng.choice([-1., 1.], n) * np.exp(rng.normal(size=n)), rng.normal(size=n)))
        scores = rng.random(n)
        d = np.argsort(-scores, kind='stable')
        before = (p.copy(), scores.copy(), d.copy())
        out = select_score_plus_strata(p, scores, d, np.array([0., .46, 3.05, 5.64, 6.10]))
        assert len(out) == min(n, 512)
        assert len(out) == len(set(out.tolist()))
        assert set(d[:min(n, 256)]) <= set(out)
        positions = {int(index): rank for rank, index in enumerate(d)}
        assert [positions[int(i)] for i in out] == sorted(positions[int(i)] for i in out)
        assert np.array_equal(out, select_score_plus_strata(p, scores, d, np.array([0., .46, 3.05, 5.64, 6.10])))
        for a, b in zip((p, scores, d), before):
            assert np.array_equal(a, b)
        if n <= 512:
            assert np.array_equal(out, d)
    # Collapsed quantile boundaries and score ties must not manufacture rows.
    p = np.tile([1., 0.], (600, 1))
    out = select_score_plus_strata(p, np.ones(600), np.arange(600), np.array([0., 6.1]))
    assert np.array_equal(out, np.arange(512))
    print('PASS: synthetic budget, baseline-core, determinism, index, ordering, tie and immutability checks.')
    print('No real-court experiment or effectiveness check has been run by this helper.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        parser.print_help()
