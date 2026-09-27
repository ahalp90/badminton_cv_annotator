# Draft update to Christian Arthur (client)

*Attachments referenced: the three PNGs in
`experiments/feedback_eval/shuttleset_v1/figures/`.*

---

Hi Arthur,

Thanks — and sorry for the confusion over the version labels. Let me set those
out properly, then give you the results you asked for.

## What Version A and Version B are

They are three builds of the same multimodal model, differing only in what
they were trained on. The point of the staging is that the comparison between
them is itself the research question.

| Version | What it is | Trained on | Status |
|---|---|---|---|
| **A — Baseline** | The pretrained open-source model, no fine-tuning at all. Asked to watch a rally clip and write coaching feedback. | nothing | committed |
| **B — Cross-sport** | Version A fine-tuned on expert critique from Ego-Exo4D (other sports), then tested on badminton "cold". | Ego-Exo4D | committed |
| **C — Full** | Version B plus a badminton fine-tuning round. | badminton clip–commentary pairs | stretch |

A is the "does an off-the-shelf model already do this?" control. B answers the
actual question — **does expert critique learned on other sports transfer to
badminton?** — and A is what we measure that transfer against. C closes the
loop to badminton but is not needed for a defensible result.

## Where we are

Honestly: the two streams are at different stages.

**The evaluation side is built and has produced real results.** That is the
subject of the rest of this email.

**The model side is behind.** InternVideo3-8B is running in a container on the
GPU node and produces a good description of a rally clip, but it is not yet
generating feedback over a clip set, so there is no Version A score yet. That
is now the single remaining gap.

## The reference set — 2,247 clips, no expert required

The risk register flagged that we have no expert assessments to score against.
We have largely solved this without needing one.

ShuttleSet annotates, for every rally, the shot that ended it, who played it,
and how the rally was lost. That is already an expert's per-rally record of
what went wrong. We join it to a library of coaching corrections, giving a
reference set of **2,247 clips over 27 players** built from 3,508 annotated
rallies.

*Figure 1: `reference_set_construction.png`*

We are deliberately conservative about what this supports. ShuttleSet records
the *outcome* — the shuttle went out, or into the net — not the *technical
cause*. Inferring "no trunk rotation" from "smash hit out" would look rigorous
while being a guess, so our templates are keyed to the observed outcome
instead: eight stroke families × two error modes. The fault statement comes
from the annotation; only the wording of the correction is ours. We also drop
the 1,189 rallies that ended in a clean winner, because the loser made no
recorded error and any feedback would have to be invented.

*Figure 2: `fault_distribution.png`*

## The result you asked for — what a BERTScore number is worth

This is the finding I would most like your view on.

Because there is no model output yet, we scored four **model-free baselines**
through the identical pipeline a real version will use, on the 543 held-out
test clips:

| Baseline | Raw F1 | Rescaled F1 |
|---|---|---|
| no output at all | 0.000 | 0.000 |
| the same generic coaching sentence on every clip | 0.879 | 0.282 |
| real coaching text about the **wrong** fault | 0.897 | 0.387 |
| the reference itself (ceiling) | 1.000 | 1.000 |

*Figure 3: `bertscore_band.png`*

A worked example from the test set — An Se Young vs Pornpawee Chochuwong,
rally 8. The rally ended with a net shot into the net, so the reference is:

> "The net shot was played into the net. Take the shuttle earlier and higher,
> keep the racket head above the hand, and let the racket face carry it over
> the tape instead of hitting down on it."

A fixed generic sentence that ignores the clip entirely — "Move to the shuttle
earlier… recover to your base position… keep the racket head up" — scores
**0.887** against that.

Two consequences:

1. **The floor is ~0.89, not 0.** Fluent, in-domain, badminton-flavoured text
   that is about the wrong thing already scores 0.897. Across all 543 clips no
   such text scored below 0.866. So a Version A at, say, 0.91 would be close to
   meaningless, and we will always report the distance above the wrong-fault
   floor rather than the raw score.
2. **We will report the rescaled figures.** Rescaling widens the gap between
   "generic" and "wrong fault" from 0.018 to 0.105 — about sixfold. That gap is
   the resolution we have to detect any real difference between Version A and
   Version B, and on the raw scale almost all of it is spent on the text being
   English.

## One methodological caveat we want to flag

Enforcing a strictly player-disjoint split turned out to be harder than
expected, and the finding is worth recording. Our first attempt was rejected
outright: the players drawn for the test set had played every one of their
matches against players in the training set, so the test side came out empty.

The corpus splits cleanly into four groups of players who only ever played each
other. Cutting along those boundaries gives 1,704 training and 543 test clips
with **zero** clips discarded. The catch is that those groups are the men's and
women's draws — so our split is sex-disjoint as well as player-disjoint, and
test performance confounds "unseen player" with "different game". For an A-vs-B
comparison, where both versions face the same shift, we think that is
acceptable and clearly the best available option. We would not use it to make
any absolute claim, and we will say so in the report.

## Freeze date

Thanks for checking that against the study schedule — noted and agreed.

Happy to walk through any of this on a call if that is easier.

Best,
Prarthu
