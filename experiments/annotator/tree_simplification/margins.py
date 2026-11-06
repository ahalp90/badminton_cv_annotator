"""Replay a fixed final-choice margin without refitting or changing layer history."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))

from experiments.annotator.heuristic_only_tune.run import read_json, write_json

# isort: split

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from annotator.contacts.score_candidates import positive_probabilities
from annotator.rallies.refine_rallies import choose_rows, finish_rallies
from experiments.annotator.contact_records import stream_payload
from experiments.annotator.development import pooled_scores
from experiments.annotator.heuristic_only_tune.data import (
    load_original_videos,
    score_stream,
)
from experiments.annotator.tree_simplification.options import LabelledOptions

MARGINS = (0.0, 0.05)


def gated_rows(pool: LabelledOptions, scores: np.ndarray, margin: float) -> dict[int, int]:
    """Compare the best option and initial keep using the same final model."""
    chosen = choose_rows(pool.runtime, scores)
    for span_id, rows in pool.span_rows.items():
        keep, best = int(rows[0]), chosen[span_id]
        if scores[best] - scores[keep] < margin:
            chosen[span_id] = keep
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.threads <= 28:
        parser.error('--threads must be between 1 and 28')
    settings = read_json(args.run / 'settings.json.gz')
    source_result = read_json(args.run / 'result.json.gz')
    if not source_result['complete']:
        parser.error('--run must name a completed configuration')
    pools_path = Path(settings['pools'])
    preparation = read_json(Path(settings['pool_settings']['prepared']) / 'preparation.json.gz')
    videos = [video for video in load_original_videos(Path(preparation['inputs']['paths']['shots_master']))
              if video.split == 'validation']
    if len(videos) != 8:
        raise ValueError('Replay requires the original eight validation videos')
    with threadpool_limits(limits=args.threads):
        models = joblib.load(args.run / 'selector.joblib')
        per_video: dict[float, dict[str, Any]] = {margin: {} for margin in MARGINS}
        for video in videos:
            pool = joblib.load(pools_path / 'videos' / f'{video.identity}.joblib')
            scores = positive_probabilities(models, pool.matrix)
            for margin in MARGINS:
                rallies, events = finish_rallies(pool.runtime, gated_rows(pool, scores, margin))
                sequences = tuple(rally.sequence for rally in rallies)
                stream = stream_payload(sequences, events)
                counts = score_stream(video, stream)
                per_video[margin][video.identity] = counts
                destination = args.output / f'margin-{margin:g}'
                (destination / 'streams').mkdir(parents=True, exist_ok=True)
                write_json(destination / 'streams' / f'{video.identity}.json.gz', stream)
        for margin, rows in per_video.items():
            destination = args.output / f'margin-{margin:g}'
            totals = pooled_scores(rows)
            write_json(destination / 'validation.json.gz', {
                'per_video': rows, 'totals': totals,
            })
            write_json(destination / 'result.json.gz', {
                'source_run': str(args.run.resolve()), 'source_settings': settings,
                'margin': margin, 'comparison': 'final model against initial keep',
                'validation': totals,
                'complete': True,
            })
        print({'stage': 'scored', 'margins': MARGINS}, flush=True)


if __name__ == '__main__':
    main()
