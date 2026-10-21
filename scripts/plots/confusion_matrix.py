#!/usr/bin/env python3
"""Confusion matrix render for the supervisor presentation.

Reads a per-split predictions npz (``<split>_serial_<n>.npz`` produced by
``bst_x_train`` at end-of-serial, or post-hoc by ``bst_x_infer --fe``) and renders a
dual-panel confusion matrix: precision-normalised (columns sum to 1) and
recall-normalised (rows sum to 1). Classes are ordered ascending by
per-class F1.

The 'Blues' colourmap is a single-hue sequential, universally readable
(protanopia-safe by virtue of being one-hue).

Usage::

    python scripts/plots/confusion_matrix.py \\
        --predictions experiments/bst_x/shuttleset/run_<id>/predictions/test_serial_5.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from classifier_shared.eval_plots import plot_confusion_matrix  # noqa: E402

DEFAULT_OUT_PATH = REPO_ROOT / 'local_scratch/presentation_prep/confusion_matrix.png'

# Mapping from raw run_id to the in-doc / in-chat common name. Used to title the figure
# so the reader sees the ablation by its working name first, run_id second. Extend as
# new runs get rendered; missing entries fall back to the run_id alone.
RUN_LABELS: dict[str, str] = {
    'run_20260505_154907': 'aug v1 + p_jit=0.3',
    'run_20260503_172922': 'shuttle_zero_fix [wipe_drop]',
    'run_20260430_170325': 'first nosides (Phase 2 LS=0.1)',
    'run_20260530_225714_593038': 'bst_24 baseline (drop unknown)',
    'run_20260530_210600_435552': 'bst_25 baseline (keep unknown)',
}


def _parse_figsize(value: str) -> tuple[float, float]:
    try:
        width, height = (float(part) for part in value.split(','))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('figsize must be W,H') from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError('figsize values must be positive')
    return width, height


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--predictions', type=Path, required=True,
        help='Path to a predictions/<split>_serial_<n>.npz dumped by '
             'bst_x_train (end-of-serial) or bst_x_infer --fe',
    )
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument(
        '--figsize', type=_parse_figsize, default=(20.0, 9.0),
        help='W,H in inches; bump for taxonomies with many classes '
             '(e.g. 28,13 for ~24)',
    )
    parser.add_argument(
        '--font-size', type=int, default=9,
        help='tick and cell-annotation font size',
    )
    args = parser.parse_args()

    payload = np.load(args.predictions, allow_pickle=True)
    run_id = str(payload['run_id'])
    serial_no = int(payload['serial_no'])
    common_name = RUN_LABELS.get(run_id)
    run_label = f'{common_name} ({run_id})' if common_name else run_id

    plot_confusion_matrix(
        y_true=payload['y_true'],
        y_pred=payload['y_pred_top1'],  # argmax preds (top-1 of the top-k dump)
        class_names=list(payload['class_list']),
        model_name=f'{run_label} S{serial_no}',
        font_size=args.font_size,
        output_path=args.out,
        figsize=args.figsize,
    )
    print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()
