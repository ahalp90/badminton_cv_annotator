# Email to Christian Arthur — progress update + extension request

**To:** carthur7@une.edu.au
**Cc:** Raymond Chiong (rchiong@une.edu.au) — *if the extension needs his sign-off*
**Subject:** COSC320 badminton feedback — progress update, and a request on the submission date

---

Hi Arthur,

Thanks for the notes on the last update, and apologies for the confusion over
the version labels. Let me clear that up, give you where we have got to, and
then ask you about one thing.

## Version A and Version B

They are three builds of the same multimodal model, differing only in what they
were trained on:

- **Version A — baseline.** The pretrained open-source model, no fine-tuning.
  Given a rally clip, asked to write coaching feedback. This is the "does an
  off-the-shelf model already do this?" control.
- **Version B — cross-sport.** Version A fine-tuned on expert critique from
  Ego-Exo4D (other sports), then tested on badminton cold. This is the one that
  answers our actual research question: **does expert critique learned on other
  sports transfer to badminton?**
- **Version C — full.** Version B plus a badminton fine-tuning round. Stretch
  only.

A is what we measure B against, so the comparison between them is the result,
not either number on its own.

## Where we have got to

**The evaluation side has moved a long way, and has produced real results.**

*We no longer need expert assessments.* This was the highest risk on our
register. ShuttleSet already annotates, for every rally, which shot ended it,
who played it and how the rally was lost — that is an expert's per-rally record
of the fault, published with the dataset. Joining it to a library of coaching
corrections gives us a reference set of **2,247 clips over 27 players**, built
from 3,508 annotated rallies.

We have been deliberately conservative about what that supports. ShuttleSet
records the *outcome* (the shuttle went out, or into the net), not the
*technical cause*. Inferring "no trunk rotation" from "smash hit out" would look
rigorous while being a guess, so our templates are keyed to the observed outcome
instead — eight stroke families × two error modes. The fault statement is
annotated; only the wording of the correction is ours. We also exclude the 1,189
rallies that ended in a clean winner, because the player who lost them made no
recorded error and any feedback would have to be invented.

*We have measured what a score is actually worth.* You asked for results that
make the evaluation easier to assess, and this is the one I would most like your
view on. Because there is no model output yet, we scored four **model-free
baselines** through the identical pipeline a real version will use, on 543
held-out clips:

| Baseline | Raw F1 | Rescaled F1 |
|---|---|---|
| no output at all | 0.000 | 0.000 |
| the same generic coaching sentence on every clip | 0.879 | 0.282 |
| real coaching text about the **wrong** fault | 0.897 | 0.387 |
| the reference itself (ceiling) | 1.000 | 1.000 |

Two consequences:

1. **The floor is roughly 0.89, not 0.** Fluent, in-domain, badminton-flavoured
   text that is about the wrong thing already scores 0.897, and across all 543
   clips nothing scored below 0.866. So a Version A reporting, say, 0.91 would
   have shown almost nothing. We will always report the distance above the
   wrong-fault floor rather than the raw score.
2. **We will report rescaled figures.** Rescaling widens the gap between
   "generic" and "wrong fault" from 0.018 to 0.105 — about sixfold. That gap is
   the resolution we have to detect any real difference between Version A and
   Version B at all.

*One methodological finding worth flagging.* Enforcing a strictly
player-disjoint split was harder than expected. Our first attempt was rejected
outright: the players drawn for the test set had played every one of their
matches against players in the training set, so the test side came out empty.
The corpus turns out to split into four groups of players who only ever played
each other, and cutting along those boundaries gives 1,704 training and 543 test
clips with **zero** clips discarded. The catch is that those groups are the
men's and women's draws, so our split is sex-disjoint as well as
player-disjoint, and test performance confounds "unseen player" with "different
game". We think that is acceptable for an A-vs-B comparison, where both versions
face the same shift, and we will say plainly in the report that it is not a
basis for any absolute claim.

**The model side is behind.** InternVideo3-8B is running in a container on the
GPU node and produces a good description of a rally clip, but it is not yet
generating feedback over a clip set, so there is no Version A score. That is now
the single remaining gap, and everything it needs to be scored against is built
and waiting.

I have figures for all of the above, plus an interactive page where you can read
any of the 543 held-out rallies against what each baseline said about it. Happy
to send either, or walk you through it on a call if that is easier.

## The request

Could we have until **Thursday 2 October** to hand in the recorded presentation
and the individual project diaries?

The technical work is where it is, and the extra days would go entirely into
producing a presentation and diaries that properly reflect it rather than
rushing both. If you would prefer we submit what we have on time and treat the
2nd as a hard cap, we are happy to do that too — just let us know which you
would rather.

Thanks very much,

Prarthu Sapkota
COSC320 — prathusapkota3@gmail.com
