"""Typed records for the COSC320 feedback-evaluation harness.

The harness answers one question: how close is a model version's written
coaching feedback to an informed reference? Every type here exists to keep the
A / B / C comparison honest, so the invariants are worth stating up front.

Scores are only comparable across model versions when they were produced by the
same scorer on the same clips. BERTScore is not an absolute quantity: change the
embedding model, the language, or the baseline rescaling and every number moves.
`ScorerConfig` is therefore stamped into each `RunScore` and checked by
`assert_comparable` before any A-vs-B claim is made.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass


# Where a reference came from. The proposal treats all three as valid references
# but the distinction has to survive into the results, because the strength of
# the claim differs:
#   expert     -- an informed assessment of this clip; the strongest reference.
#   template   -- good feedback compiled from coaching literature; says what a
#                 coach would say about this fault, not about this clip.
#   commentary -- broadcast commentary paired to the rally by the COSC595
#                 scraper. Descriptive, not corrective: it reports what happened
#                 rather than what to change. Matching it is a weaker and
#                 different claim than matching a coach, and it must never be
#                 recorded as `expert`. See commentary_references.py.
REFERENCE_SOURCES = ("expert", "template", "commentary")


def _require_player_ids(clip_id: str, player_ids: tuple[str, ...]) -> None:
    """Reject an empty or string-shaped player list.

    A bare string is the mistake worth catching: it is iterable, so it would
    pass silently and give the clip one "player" per letter, quietly inflating
    `n_players` and scattering the per-player means.
    """
    if isinstance(player_ids, str):
        raise TypeError(
            f"{clip_id}: player_ids must be a tuple of player ids, not the string "
            f"{player_ids!r}"
        )
    if not player_ids:
        raise ValueError(f"{clip_id}: a clip must name at least one player")


@dataclass(frozen=True)
class ReferenceRecord:
    """One clip's gold feedback, with every accepted phrasing of it.

    Multiple references are not optional polish. Coaching feedback has many
    correct wordings ("get your racket up earlier" / "prepare the racket sooner"),
    and BERTScore takes the best-matching reference, so a single reference
    systematically under-scores correct feedback that happens to be phrased
    differently.

    `player_ids` is a set, not one player, because a singles rally has two of
    them. Recording only one would force a false choice: naming a "subject"
    player needs a per-rally judgement the commentary rarely supports, and a
    combined pair key is not player-disjoint at all -- pair AB in train and pair
    AC in test put A on both sides. Carrying both lets `splits` do the only
    construction that is actually disjoint. See splits.build_split.
    """

    clip_id: str
    player_ids: tuple[str, ...]
    references: tuple[str, ...]
    source: str

    def __post_init__(self) -> None:
        _require_player_ids(self.clip_id, self.player_ids)


@dataclass(frozen=True)
class PredictionRecord:
    """One clip's generated feedback from a single model version."""

    clip_id: str
    feedback: str


@dataclass(frozen=True)
class ScorerConfig:
    """The scorer settings that make two runs comparable.

    `model_type` of None means bert-score picks its own default for `lang`. That
    default can change between bert-score releases, so a run pinned for the
    report should set it explicitly.
    """

    model_type: str | None = None
    lang: str = "en"
    rescale_with_baseline: bool = False
    batch_size: int = 16


@dataclass(frozen=True)
class ScoredClip:
    """BERTScore precision/recall/F1 for one clip."""

    clip_id: str
    player_ids: tuple[str, ...]
    precision: float
    recall: float
    f1: float
    # True when the model returned nothing for this clip. Empty output is scored
    # as a hard zero rather than dropped -- see scoring.score_run.
    empty_prediction: bool = False

    def __post_init__(self) -> None:
        _require_player_ids(self.clip_id, self.player_ids)


@dataclass(frozen=True)
class RunScore:
    """Every clip score for one model version, plus the config that produced it."""

    model_version: str
    scorer: ScorerConfig
    clips: tuple[ScoredClip, ...]

    @property
    def n_clips(self) -> int:
        return len(self.clips)

    @property
    def n_players(self) -> int:
        return len({player for clip in self.clips for player in clip.player_ids})

    @property
    def n_empty(self) -> int:
        return sum(1 for clip in self.clips if clip.empty_prediction)

    @property
    def mean_f1(self) -> float:
        return self._mean("f1")

    @property
    def mean_precision(self) -> float:
        return self._mean("precision")

    @property
    def mean_recall(self) -> float:
        return self._mean("recall")

    @property
    def stdev_f1(self) -> float:
        """Population stdev of F1. Zero for a single clip rather than undefined."""
        if len(self.clips) < 2:
            return 0.0
        return statistics.stdev([clip.f1 for clip in self.clips])

    def mean_f1_by_player(self) -> dict[str, float]:
        """Per-player mean F1.

        A single player carrying the mean is the failure this surfaces: with a
        player-disjoint split the test players are few, so one atypical player
        can move the headline number more than the model version does.

        A clip counts towards every player it involves, so these means overlap
        and do not partition the clips -- a rally between two test players is in
        both of their figures. They diagnose an outlying player; they do not sum
        back to the headline mean.
        """
        by_player: dict[str, list[float]] = {}
        for clip in self.clips:
            for player in clip.player_ids:
                by_player.setdefault(player, []).append(clip.f1)
        return {player: statistics.fmean(scores) for player, scores in sorted(by_player.items())}

    def _mean(self, field: str) -> float:
        if not self.clips:
            raise ValueError("cannot average an empty run")
        return statistics.fmean([getattr(clip, field) for clip in self.clips])


class ComparabilityError(RuntimeError):
    """Raised when two runs cannot be compared as they stand."""


def assert_comparable(left: RunScore, right: RunScore) -> None:
    """Refuse to compare runs that differ in anything except the model version.

    Called before reporting "B beats A". Without it the harness will happily
    subtract two numbers produced by different embedding models on different
    clip sets and report the difference as a finding.
    """
    if left.scorer != right.scorer:
        raise ComparabilityError(
            f"scorer config differs: {left.model_version}={left.scorer!r} "
            f"vs {right.model_version}={right.scorer!r}"
        )
    left_clips = {clip.clip_id for clip in left.clips}
    right_clips = {clip.clip_id for clip in right.clips}
    if left_clips != right_clips:
        only_left = sorted(left_clips - right_clips)
        only_right = sorted(right_clips - left_clips)
        raise ComparabilityError(
            f"clip sets differ: only in {left.model_version}={only_left}, "
            f"only in {right.model_version}={only_right}"
        )
    if left.model_version == right.model_version:
        raise ComparabilityError(f"both runs are model version {left.model_version!r}")
