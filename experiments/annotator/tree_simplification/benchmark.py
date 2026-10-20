"""Time both bundles through production inference on the same prepared validation inputs.

Excludes vision, contact-feature construction, file reads and metric scoring.
Includes contact prediction, fresh rally proposals, selection and review scoring.
Two repetitions reverse model order; every stream must match its saved reference.
"""

import argparse
import time
from pathlib import Path

import joblib
from threadpoolctl import threadpool_limits

from annotator.hybrid import predict_contact_features
from annotator.models import fitted_trees, load_models
from experiments.annotator.heuristic_only_tune.run import (
    read_json,
    source_record,
    write_json,
)
from experiments.annotator.old_court_regression.retrain import stream_payload
from experiments.annotator.tree_simplification.pilot import progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True, type=Path)
    parser.add_argument('--old-bundle', required=True, type=Path)
    parser.add_argument('--new-bundle', required=True, type=Path)
    parser.add_argument('--old-streams', required=True, type=Path)
    parser.add_argument('--new-streams', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bundles = {'old': load_models(args.old_bundle), 'new': load_models(args.new_bundle)}
    streams = {'old': args.old_streams, 'new': args.new_streams}
    for bundle in bundles.values():
        for tree in fitted_trees(bundle):
            if type(tree).__module__.startswith('xgboost'):
                tree.set_params(device='cpu')
    rows = []
    videos = []
    with threadpool_limits(limits=2):
        for path in sorted((args.prepared / 'videos').glob('*.joblib')):
            if read_json(path.with_suffix('.json.gz'))['split'] != 'validation':
                continue
            prepared = joblib.load(path)
            identity = prepared.video.identity
            videos.append(identity)
            expected = {name: read_json(folder / f'{identity}.json.gz') for name, folder in streams.items()}
            for repetition, order in enumerate((('old', 'new'), ('new', 'old'))):
                for name in order:
                    started = time.perf_counter()
                    prediction = predict_contact_features(prepared.evidence, prepared.features, bundles[name])
                    elapsed = time.perf_counter() - started
                    actual = stream_payload(prediction.refined.sequences, prediction.refined.events)
                    if actual != expected[name]:
                        raise ValueError(f'{identity}/{name}: timing replay changed the saved annotation stream')
                    rows.append({'video': identity, 'model': name, 'repetition': repetition, 'wall_s': elapsed})
                    del prediction
            progress(args.output, 'timed-video', video=identity, complete=len(videos), planned=8)
            del prepared
    if len(videos) != 8:
        raise ValueError('Timing requires all eight validation videos')
    totals = {}
    for name in bundles:
        totals[name] = [
            sum(row['wall_s'] for row in rows if row['model'] == name and row['repetition'] == repetition)
            for repetition in (0, 1)
        ]
    write_json(args.output / 'result.json.gz', {
        'complete': True, 'source': source_record(), 'threads': 2, 'rows': rows, 'total_wall_s': totals,
        'inputs': {name: str(value.resolve()) for name, value in vars(args).items()},
        'scope': __doc__, 'streams_match': True,
    })
    progress(args.output, 'complete', total_wall_s=totals)


if __name__ == '__main__':
    main()
