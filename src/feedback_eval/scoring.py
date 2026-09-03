"""BERTScore scoring for generated coaching feedback.

The scorer is injected rather than constructed inline. That keeps the whole
harness testable on CPU with no model download -- `bert-score` lives in the
optional `scraper` extra and pulls the transformers stack, so requiring it to
run the tests would make this code untestable in CI.

Why BERTScore at all: the proposal scores *meaning*, not wording. Two pieces of
coaching feedback can share almost no tokens and say the same thing, which is
exactly the case exact-match and n-gram metrics get wrong.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from .contracts import (
    PredictionRecord,
    ReferenceRecord,
    RunScore,
    ScoredClip,
    ScorerConfig,
)

logger = logging.getLogger(__name__)

# A scorer takes candidates and their per-candidate reference lists, and returns
# (precision, recall, f1) with one float per candidate. bert_score's BERTScorer
# already has this shape; a fake with the same signature drives the tests.
Scorer = Callable[
    [Sequence[str], Sequence[Sequence[str]]],
    tuple[Sequence[float], Sequence[Sequence[float] | float], Sequence[float]],
]


def _bert_score_device() -> str:
    """Choose CUDA only when the installed torch build supports the GPU arch.

    Same check as src/scraper/commentary_cleaning.py: a torch build without the
    host's compute capability compiled in fails at the first kernel launch rather
    than at device selection, which reads as a model bug instead of an install
    one.
    """
    import torch

    capability = "unavailable"
    device = "cpu"
    if torch.cuda.is_available():
        capability_tuple = torch.cuda.get_device_capability()
        capability = capability_tuple
        arch = f"sm_{capability_tuple[0]}{capability_tuple[1]}"
        if arch in torch.cuda.get_arch_list():
            device = "cuda"
    logger.warning("BERTScore device=%s capability=%s", device, capability)
    if device == "cpu" and capability != "unavailable":
        logger.warning("BERTScore CUDA architecture is unsupported by this torch build; using CPU")
    return device


def build_bert_scorer(config: ScorerConfig) -> Scorer:
    """Build the real BERTScore-backed scorer.

    Imported lazily so that importing this module -- and running the tests --
    never needs the transformers stack.
    """
    from bert_score import BERTScorer

    scorer = BERTScorer(
        model_type=config.model_type,
        lang=config.lang,
        rescale_with_baseline=config.rescale_with_baseline,
        device=_bert_score_device(),
        batch_size=config.batch_size,
    )

    def score(
        candidates: Sequence[str],
        references: Sequence[Sequence[str]],
    ) -> tuple[Sequence[float], Sequence[Sequence[float] | float], Sequence[float]]:
        # BERTScorer accepts a list of reference lists and keeps the best-matching
        # reference per candidate, which is the multi-phrasing behaviour we want.
        precision, recall, f1 = scorer.score(list(candidates), [list(r) for r in references])
        return precision.tolist(), recall.tolist(), f1.tolist()

    return score


def score_run(
    pairs: Sequence[tuple[ReferenceRecord, PredictionRecord]],
    *,
    model_version: str,
    scorer: Scorer,
    config: ScorerConfig,
) -> RunScore:
    """Score one model version's feedback against the reference set.

    Empty predictions are scored as a hard zero and never sent to the scorer.
    BERTScore on an empty candidate is not meaningful -- it either errors or
    returns a similarity against nothing -- and dropping those clips instead
    would quietly reward a model for staying silent.
    """
    if not pairs:
        raise ValueError("cannot score an empty run")

    scored_indices = [index for index, (_, prediction) in enumerate(pairs) if prediction.feedback]
    candidates = [pairs[index][1].feedback for index in scored_indices]
    references = [list(pairs[index][0].references) for index in scored_indices]

    results: dict[int, tuple[float, float, float]] = {}
    if candidates:
        precision, recall, f1 = scorer(candidates, references)
        if not (len(precision) == len(recall) == len(f1) == len(candidates)):
            raise ValueError(
                f"scorer returned {len(precision)}/{len(recall)}/{len(f1)} scores "
                f"for {len(candidates)} candidates"
            )
        for position, index in enumerate(scored_indices):
            results[index] = (
                float(precision[position]),
                float(recall[position]),
                float(f1[position]),
            )

    clips = []
    for index, (reference, prediction) in enumerate(pairs):
        clip_precision, clip_recall, clip_f1 = results.get(index, (0.0, 0.0, 0.0))
        clips.append(
            ScoredClip(
                clip_id=reference.clip_id,
                player_ids=reference.player_ids,
                precision=clip_precision,
                recall=clip_recall,
                f1=clip_f1,
                empty_prediction=not prediction.feedback,
            )
        )

    empty = sum(1 for clip in clips if clip.empty_prediction)
    if empty:
        logger.warning(
            "%s: %d/%d clips had empty feedback and were scored 0.0",
            model_version,
            empty,
            len(clips),
        )
    return RunScore(model_version=model_version, scorer=config, clips=tuple(clips))
