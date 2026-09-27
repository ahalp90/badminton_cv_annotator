# Email to Christian Arthur — short version

**To:** carthur7@une.edu.au
**Subject:** COSC320 badminton feedback — progress, and a request on the submission date

---

Hi Arthur,

Thanks for the notes, and sorry for the confusion over the version labels.

**Versions A, B and C** are three builds of the same multimodal model, differing
only in what they were trained on:

- **A — baseline.** The pretrained model, no fine-tuning, asked to write
  coaching feedback on a rally clip. The "does an off-the-shelf model already do
  this?" control.
- **B — cross-sport.** A, fine-tuned on expert critique from Ego-Exo4D (other
  sports), then tested on badminton cold. This answers our actual research
  question: does critique learned on other sports transfer to badminton?
- **C — full.** B plus a badminton fine-tuning round. Stretch only.

**Where we have got to.** The evaluation side has produced real results, and we
no longer need expert assessments — our highest-rated risk. ShuttleSet already
annotates which shot ended each rally and how it was lost, which is an expert's
record of the fault. Joining that to a library of coaching corrections gives us
a reference set of **2,247 clips over 27 players**, on a player-disjoint split
of 1,704 training / 543 held-out clips.

With no model output yet, we scored four **model-free baselines** through the
pipeline a real version will use, on the 543 held-out clips:

| Baseline | Raw F1 | Rescaled |
|---|---|---|
| no output at all | 0.000 | 0.000 |
| the same generic sentence on every clip | 0.879 | 0.282 |
| real coaching text about the **wrong** fault | 0.897 | 0.387 |
| the reference itself (ceiling) | 1.000 | 1.000 |

The finding worth your view: **the floor is roughly 0.89, not 0.** Fluent
badminton text about the wrong thing already scores 0.897, and nothing across
543 clips scored below 0.866. So a Version A of 0.91 would show almost nothing —
we will report the distance above the wrong-fault floor, not the raw score.

**The model side is behind.** InternVideo3-8B runs on the GPU node and describes
a clip well, but it is not yet generating feedback over a clip set, so there is
no Version A score. Everything it needs to be scored against is built and
waiting.

**What we can finish by 2 October.** A Version A score. What remains is plumbing
— fetch the 11 matches our test clips come from, cut the rallies using timings
ShuttleSet already provides, and run the model over them; scoring itself takes
about ten seconds. We will use a fixed subset of ~100 clips rather than all 543,
which is still ample to separate the model from the floor. The real unknown is
the prompt: asked to describe a clip the model writes commentary, where our
references are corrective, and closing that gap is the interesting part.

**We will not have Version B.** Ego-Exo4D needs an access application and a
fine-tuning setup none of us has built before — weeks, not days. Better to say
so now than promise it and miss.

Could we have until **Thursday 2 October** for the recorded presentation and the
individual project diaries? The extra days would go into producing both
properly rather than rushing them.

I have figures for all of the above, plus an interactive page where you can read
any held-out rally against what each baseline said — happy to send either, or
walk you through it on a call.

Thanks very much,

Prarthu Sapkota
COSC320
