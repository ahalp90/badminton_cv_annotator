"""Re-score BST-X's saved 25-class test predictions on the ShuttleSet authors' 10 broader stroke types.

The project README quotes two numbers for the best bst_25 run: macro-F1 on the 25 trained classes,
and macro-F1 after re-scoring the same predictions on the 10 stroke types the ShuttleSet authors use
(ShuttleNet, arXiv 2112.01044; ShuttleSet, arXiv 2306.04948). Nothing is retrained.

The 10-type re-score:
- ignores court side, so Top_X and Bottom_X both become X;
- folds cross-court net shots into net shots and rushes into pushes, as the authors' grouping does;
- leaves out every stroke labelled or predicted `unknown`. Nearly all `unknown` labels are ShuttleSet
  flaw rows, so the class is a catch-all rather than a stroke.

BST's 25 classes had already folded defensive return lobs and drives into lob and drive (about 1.9%
of strokes), which the authors' list files under defensive shots. The saved predictions cannot
separate them back out, so this grouping is a close copy of the authors' list, not an exact one.

Usage: python experiments/bst_x/rescore_broader_strokes.py [test_serial_N.npz ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

README_PREDICTIONS = (Path(__file__).parent
                      / 'shuttleset/run_20260530_210600_435552/predictions/test_serial_1.npz')
SIDE_PREFIXES = ('Top_', 'Bottom_')
BROADER_TYPES = {'cross_court_net_shot': 'net_shot', 'rush': 'push'}
UNKNOWN = 'unknown'


def broader_type(class_name: str) -> str:
    stroke = class_name.split('_', 1)[1] if class_name.startswith(SIDE_PREFIXES) else class_name
    return BROADER_TYPES.get(stroke, stroke)


def rescore(path: Path) -> None:
    # The class list is saved as an object array, so loading needs pickle; these are our own run files.
    predictions = np.load(path, allow_pickle=True)
    class_names = [str(name) for name in predictions['class_list']]
    true_index, predicted_index = predictions['y_true'], predictions['y_pred_top1']
    trained_score = f1_score(true_index, predicted_index, labels=range(len(class_names)), average='macro')

    true_types = np.array([broader_type(class_names[index]) for index in true_index])
    predicted_types = np.array([broader_type(class_names[index]) for index in predicted_index])
    labelled_stroke = true_types != UNKNOWN
    predicted_stroke = predicted_types != UNKNOWN
    scored = labelled_stroke & predicted_stroke
    stroke_types = sorted(set(true_types[scored]))
    broader_score = f1_score(true_types[scored], predicted_types[scored], labels=stroke_types, average='macro')

    print(path)
    print(f'  {len(class_names)} trained classes, {len(true_index)} strokes: macro-F1 {trained_score:.4f}')
    print(f'  {len(stroke_types)} broader types, {int(scored.sum())} strokes: macro-F1 {broader_score:.4f}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('predictions', type=Path, nargs='*', default=[README_PREDICTIONS],
                        help='bst_25 test_serial_N.npz files; defaults to the run the README quotes')
    for path in parser.parse_args().predictions:
        rescore(path)


if __name__ == '__main__':
    main()
