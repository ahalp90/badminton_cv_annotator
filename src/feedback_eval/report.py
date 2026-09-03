"""Serialise and display run scores.

The JSON payload is the durable artefact: it carries the scorer config, so a run
kept from August is still checkably comparable against one produced in September.
"""
from __future__ import annotations

from dataclasses import asdict

from .contracts import RunScore, assert_comparable


def as_dict(run: RunScore) -> dict:
    """Full run payload -- config, aggregates, and every per-clip score."""
    return {
        "model_version": run.model_version,
        "scorer": asdict(run.scorer),
        "aggregates": {
            "n_clips": run.n_clips,
            "n_players": run.n_players,
            "n_empty_predictions": run.n_empty,
            "mean_precision": run.mean_precision,
            "mean_recall": run.mean_recall,
            "mean_f1": run.mean_f1,
            "stdev_f1": run.stdev_f1,
        },
        "mean_f1_by_player": run.mean_f1_by_player(),
        "clips": [asdict(clip) for clip in run.clips],
    }


def render_summary(run: RunScore) -> str:
    """One-screen summary for the terminal."""
    lines = [
        f"model version : {run.model_version}",
        f"scorer        : model_type={run.scorer.model_type} lang={run.scorer.lang} "
        f"rescaled={run.scorer.rescale_with_baseline}",
        f"clips         : {run.n_clips} over {run.n_players} player(s), "
        f"{run.n_empty} empty prediction(s)",
        f"mean F1       : {run.mean_f1:.4f} (sd {run.stdev_f1:.4f})",
        f"mean P / R    : {run.mean_precision:.4f} / {run.mean_recall:.4f}",
        "",
        "per player:",
    ]
    for player, mean_f1 in run.mean_f1_by_player().items():
        lines.append(f"  {player:<24} {mean_f1:.4f}")
    return "\n".join(lines)


def render_comparison(baseline: RunScore, candidate: RunScore) -> str:
    """Render candidate-minus-baseline, refusing incomparable runs.

    The delta is the project's actual result -- "does cross-sport critique
    transfer to badminton" is answered by B minus A, not by either number alone.
    """
    assert_comparable(baseline, candidate)
    delta = candidate.mean_f1 - baseline.mean_f1
    return "\n".join(
        [
            f"{baseline.model_version} mean F1 : {baseline.mean_f1:.4f}",
            f"{candidate.model_version} mean F1 : {candidate.mean_f1:.4f}",
            f"delta          : {delta:+.4f}",
            "",
            "Both runs share a scorer config and clip set, so the delta is the "
            "comparable quantity. The absolute values are not a skill score.",
        ]
    )
