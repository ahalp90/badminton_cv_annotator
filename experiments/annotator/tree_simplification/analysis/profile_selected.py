"""Profile the chosen model on one validation video and check that its predictions do not change.

This made ``selected-profile.json.gz``. It runs ``predict_contact_features`` once under
``cProfile`` on two CPU threads, then compares the predicted stream with the saved one, so the
profile describes the same predictions as the evidence. Profiling adds its own overhead, so the
times are only good for comparing parts of one run.

The prepared arrays are not in the repository. ``prepare`` rebuilds them from the video fixtures
(see the reproduction guide). Paths under this checkout and its Python environment are written as
``CHECKOUT/`` and ``VENV/``.

Run from the repository root with ``PYTHONPATH=src:.``.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
from pathlib import Path

import joblib
from threadpoolctl import threadpool_limits

from annotator.hybrid import predict_contact_features
from annotator.models import load_models
from experiments.annotator.heuristic_only_tune.run import source_record
from experiments.annotator.old_court_regression.retrain import stream_payload
from experiments.annotator.tree_simplification.analysis.common import (
    REPOSITORY,
    RESULTS,
    read_json,
    write_json_gz,
)

THREADS = 2
SAVED_STREAMS = RESULTS / 'streams/selection/31-leaves-400-rounds-1-selector/seed-0'


def placeholder_path(filename: str) -> str:
    """Replace the private part of a profiled file's path with the README's placeholders."""
    for root, placeholder in ((REPOSITORY, 'CHECKOUT'), (Path(sys.prefix), 'VENV')):
        if filename.startswith(f'{root}/'):
            return placeholder + filename.removeprefix(str(root))
    return filename


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bundle', type=Path, default=RESULTS / 'models/selected', help='model bundle folder')
    parser.add_argument('--prepared', type=Path, required=True, help="the video's prepared .joblib file")
    parser.add_argument('--saved-stream', type=Path, help='saved prediction file to compare with; '
                        'defaults to the chosen model\'s seed-0 validation file for the same video')
    parser.add_argument('--output', type=Path, required=True, help='gzip JSON file to write')
    args = parser.parse_args()

    models = load_models(args.bundle)
    prepared = joblib.load(args.prepared)
    saved_stream = args.saved_stream or SAVED_STREAMS / f'{prepared.video.identity}.json.gz'
    profiler = cProfile.Profile()
    with threadpool_limits(limits=THREADS):
        prediction = profiler.runcall(predict_contact_features, prepared.evidence, prepared.features, models)
    stream = stream_payload(prediction.refined.sequences, prediction.refined.events)
    if stream != read_json(saved_stream):
        raise ValueError(f'The profiled run predicted a different stream from {saved_stream}')

    rows = [
        {'file': placeholder_path(filename), 'line': line, 'name': name, 'primitive_calls': primitive,
         'calls': calls, 'own_s': own, 'cumulative_s': cumulative}
        for (filename, line, name), (primitive, calls, own, cumulative, _callers)
        in pstats.Stats(profiler).stats.items()
    ]
    rows.sort(key=lambda row: row['cumulative_s'], reverse=True)
    write_json_gz(args.output, {
        'complete': True, 'source': source_record(), 'stream_matches': True, 'video': prepared.video.identity,
        'frames': len(prepared.evidence.track), 'fps': prepared.video.fps, 'threads': THREADS, 'profile': rows,
    })
    print(f'Profile written to {args.output}; the predictions match {saved_stream.name}', flush=True)


if __name__ == '__main__':
    main()
