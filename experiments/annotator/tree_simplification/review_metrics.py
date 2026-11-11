"""Measure how well confidence scores bring incorrect rallies to review first."""

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    judged = labels >= 0
    correct, confidence = labels[judged], scores[judged]
    wrong = 1 - correct
    result: dict[str, Any] = {'judged': len(correct), 'unjudgeable': int((~judged).sum()), 'wrong': int(wrong.sum())}
    result['correct_auc'] = float(roc_auc_score(correct, confidence)) if len(np.unique(correct)) == 2 else None
    result['wrong_average_precision'] = float(average_precision_score(wrong, 1 - confidence)) if wrong.any() else None
    ordered = np.argsort(confidence, kind='stable')
    for fraction in (.1, .25, .5):
        budget = int(np.ceil(len(ordered) * fraction))
        result[f'wrong_in_first_{fraction:g}'] = int(wrong[ordered[:budget]].sum())
    return result
