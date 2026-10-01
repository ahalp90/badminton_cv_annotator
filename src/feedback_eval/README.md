# Feedback-evaluation harness (COSC320)

Scores generated coaching feedback against a reference set with BERTScore, on a
player-disjoint split. This is proposal tasks **3.1**, **3.2** and **3.3**, and
it covers the functional requirements that every model version be scored by a
documented, repeatable evaluation and that the split reflect generalisation
rather than memorisation.

It is deliberately independent of the model. The harness takes text in and gives
scores out, so it was built and tested before Version A existed rather than
waiting on it — Appendix A lists 3.3 as depending on 2.3, but BERTScore only
needs two strings, so the dependency is on the *predictions file*, not on the
model.

> **Every command below needs `PYTHONPATH=src`.** The project is not installed
> into the venv as a package; `conftest.py` puts `src` on the path for pytest,
> and nothing does it for a bare `python -m`.

## The commands

```bash
# 3.1a derive per-clip faults from ShuttleSet's expert annotations (the built route)
python -m feedback_eval.shuttleset_faults --out-faults ... --out-players ...

# 3.1b turn coaching templates + those faults into a reference set
python -m feedback_eval.template_references --templates ... --clip-faults ... --players ... --out ...

# 3.1c or from the COSC595 commentary lane, once that pipeline has been run
python -m feedback_eval.commentary_references --pairs ... --chunks-dir ... --players ... --out ...

# 3.2  partition players once, and keep the file
python -m feedback_eval.split_cli --references ... --seed 20260903 --out ...

# 3.3  score one model version on one side of that split
python -m feedback_eval.score_cli --references ... --predictions ... --split ... --model-version A

# Version A inputs: a fixed subset of the test side, and the clips it names
python -m feedback_eval.subset_split --split ... --references ... --size 100 --seed ... --out ...
python -m feedback_eval.shuttleset_clips --split ... --fetch
```

## Inputs

Two JSONL files, joined on `clip_id`.

`references.jsonl` — the gold set, written once and reused for every model version:

```json
{"clip_id": "abc123_r7", "player_ids": ["Viktor Axelsen", "Kento Momota"], "references": ["...", "..."], "source": "commentary"}
```

- `references` — every accepted phrasing. BERTScore keeps the best-matching one.
  A single reference systematically under-scores correct feedback that happens
  to be worded differently, so more phrasings is better.
- `player_ids` — **every** player in the rally, not one of them. See
  *The split* below for why this is a list.
- `source` — `expert`, `template` or `commentary`. Kept per clip because the
  three support different claims and the report has to be able to say which.
  Only a genuine per-clip assessment may be recorded as `expert`.

`predictions.jsonl` — one file per model version:

```json
{"clip_id": "abc123_r7", "feedback": "The backhand backswing starts too late..."}
```

Blank `feedback` is legal and scores a hard zero. A model that returns nothing
is a real result; dropping those clips would quietly reward it for staying silent.

## The split (task 3.2)

The unit of assignment is the **player**, never the clip. A model that has learnt
one player's habits scores well on that player's held-out clips for the wrong
reason.

A singles rally has two players, which makes the construction less obvious than
it looks. Two tempting shortcuts are both wrong:

- **Naming one "subject" player per rally** needs a per-rally judgement about
  whose error it was. Commentary rarely says, so the call would come from a
  winner heuristic and its error would land straight in the split.
- **Keying on the pair** is not disjoint at all. Pair AB in train and pair AC in
  test both contain A, so A trains *and* tests.

So `build_split` partitions the players, then keeps a clip only when **every**
player in it falls on the same side. Clips spanning the boundary are discarded
and named in the split file. That costs clips — more at an even partition, fewer
at 70/30 — and paying it is the only version where "player-disjoint" is true as
written. `--test-fraction` is therefore a target, not a promise: read the
realised counts off the output.

Pin the test players by name for anything reported:

```bash
python -m feedback_eval.split_cli --references data/feedback_eval/references.jsonl \
    --test-player "Viktor Axelsen" --test-player "An Se Young" \
    --out data/feedback_eval/split.json
```

A named split survives a change of seed, of Python version, and of `build_split`
itself. Commit the JSON and pass it to every run: A, B and C have to be scored on
the *same* test clips or their means are not comparable. `split_cli` refuses to
overwrite an existing split without `--force`, because re-splitting after seeing
a score is how a player-disjoint split stops meaning anything.

`assert_player_disjoint` runs on the records actually handed to the scorer, not
on the split file — the file can be right while the selection is not.

## The reference set (task 3.1)

`commentary_references.py` builds `references.jsonl` from the COSC595 commentary
lane. The two lanes already agree in shape:

| From | File | Gives |
|---|---|---|
| `scraper.commentary_pairing` | `rally_commentary_pairs.csv` | `video_id, rally_id -> chunk_id` |
| `scraper.commentary_cleaning` | `chunks/<video_id>.json` | `text_clean`, `alt_phrasings`, `clean_pass` |
| Curtis's v1 schema | your `players.csv` | `video_id, rally_id -> player_id` (one row per player) |

`alt_phrasings` is the useful coincidence: the cleaning stage already produces
`ALT_PHRASINGS_K` meaning-preserving paraphrases per chunk, which is exactly the
multi-phrasing reference list BERTScore wants, for the same reason. Chunks that
failed the cleaning stage's own BERTScore gate (`clean_pass`) are dropped unless
`--keep-failed-clean` is passed.

Every drop is counted and printed — unpaired rally, missing chunk, failed gate,
blank text, no player — because a reference set silently missing a third of its
rallies still scores, and the mean it produces looks fine.

**What this reference set is, and is not.** Broadcast commentary is *descriptive
and reactive*: it says what happened and how good it looked, not what the player
should change. Scoring against it measures whether generated text matches how an
informed observer described the rally — **not** whether the model gives good
coaching feedback. That is a weaker and different claim than the proposal's
expert-assessment or coaching-template references, and the report must say so
plainly. Records are written `source="commentary"` and must never be recorded as
`expert`. This is the route Rai & Kovashka (2026) take deliberately, pairing
competition commentary with coaching literature, so it is defensible — it is not
interchangeable.

`clip_id` is `{video_id}_r{rally_id}`, optionally prefixed by `--run-id`. The
rally dataset contract is explicit that `rally_id` is a list position and is
**not** stable across extraction runs, so pass `--run-id` whenever references and
predictions could come from different runs.

## Coaching templates (task 3.1, the unblocked route)

`templates.py` holds feedback keyed by **fault** rather than by clip: what a
coach says when a player meets the shuttle behind the body, whoever that player
is. This is the proposal's primary reference plan, chosen so evaluation is
"never blocked on securing an expert" — it needs no GPU, no API budget, and no
other team's pipeline output.

The shipped library is `data/feedback_eval/templates/badminton_singles_v1.jsonl`
— 16 templates across serve, net play, defence, overhead, movement, recovery,
grip, positioning and tactics, each with at least three phrasings of the
correction.

```bash
python -m feedback_eval.template_references \
    --templates data/feedback_eval/templates/badminton_singles_v1.jsonl \
    --clip-faults data/feedback_eval/clip_faults.csv \
    --players data/feedback_eval/clip_players.csv \
    --out data/feedback_eval/references.jsonl
```

`clip_faults.csv` is `clip_id,template_id`, one row per fault the clip shows; a
clip with two faults takes the union of both templates' corrections, because
feedback naming either real fault is not wrong. Deciding which faults a clip
shows is a human judgement and `template_references.py` does not make it —
something has to watch the clip. For the cause-keyed library above, nobody has;
`shuttleset_faults.py` below is the route that got around it.

## The ShuttleSet reference set (task 3.1, built)

The clip-labelling cost above was the last thing between the harness and a real
reference set: somebody had to watch several thousand rallies and say what went
wrong in each. **They already did.** ShuttleSet annotates, for every rally, the
shot that ended it, who played it, and how the rally was lost. That is an
expert's per-rally record of the fault, published with the dataset.

```bash
python -m feedback_eval.shuttleset_faults \
    --out-faults data/feedback_eval/shuttleset_clip_faults.csv \
    --out-players data/feedback_eval/shuttleset_clip_players.csv
```

**What this derivation claims.** ShuttleSet records the *outcome* of the
terminal shot — the shuttle went out, or into the net — not its *technical
cause*. Mapping "smash hit out" back to "no trunk rotation" would be an
inference the data cannot support, and the reference set would look rigorous
while laundering a guess. So this route targets a second, **outcome-keyed**
library, `badminton_singles_shuttleset_v1.jsonl`: 16 templates over eight stroke
families × two error modes. The fault statement is annotated; only the
correction text is drafted.

Three classes of rally are excluded, and each is counted:

| Excluded | Rallies | Why |
|---|---|---|
| ended in a clean winner | 1189 | the loser made no recorded error; a fault would have to be invented |
| ambiguous `lose_reason` | 55 | the misjudgement label does not consistently identify who erred |
| point credited to the erring player | 14 | annotation noise: you cannot lose the point you erred on and win it |
| stroke type `未知球種` | 3 | no family to key a template to |

That leaves **2247 clips over 27 players**, from 3508 annotated rallies.

### The split this corpus actually admits

`build_split` refused the first seeded 30% split outright: the three players it
drew shared every one of their clips with a train player, so the test side was
empty. That is not a bug, it is what a strict player-disjoint rule does to a
densely-connected tournament corpus, and it is worth reporting.

The co-occurrence graph has **four connected components** (16, 7, 2, 2 players),
so a cut along component boundaries discards nothing. Putting the three smaller
components in test gives **1704 train / 543 test clips, 0 discarded**:

```bash
python -m feedback_eval.split_cli --references data/feedback_eval/references_shuttleset_v1.jsonl \
    --test-player "An Se Young" --test-player "Carolina MARIN" ... \
    --out data/feedback_eval/split_shuttleset_v1.json
```

**Caveat that has to reach the report:** those components are the men's and
women's draws, so this split is sex-disjoint as well as player-disjoint. Test
performance therefore confounds "unseen player" with "different game". It is
still the right split for an A-vs-B comparison, where both versions face the
same shift, and it is the wrong basis for any absolute claim.

## What the baselines say (the band to read scores against)

`baselines.py` generates model-free predictions over the reference set — no GPU,
no checkpoint — and scores them through exactly the same path a real model
version will use. They exist because a single BERTScore number from Version A
answers nothing on its own.

| Baseline | Raw F1 | Rescaled F1 |
|---|---|---|
| `empty` — no output | 0.000 | 0.000 |
| `constant` — same generic advice on every clip | 0.879 | 0.282 |
| `random_template` — real coaching text, wrong fault | 0.897 | 0.387 |
| `oracle` — copies the reference | 1.000 | 1.000 |

543 held-out clips, `--model-type roberta-large`.

`random_template` draws from 48 corrections over 16 templates, so 6.8% of draws
(37 clips) land on the clip's own template and score 1.000. Those lucky hits
lift its mean from 0.889 to 0.897; both numbers are floors worth quoting, and
the lower one is the fairer "wrong fault, right register" figure. Across all
543 clips no draw scored below **0.866** — the entire observed range of fluent
coaching English against this reference set is 0.866 to 1.000.

Two findings the report should carry:

1. **The floor is 0.897, not 0.** Text that is fluent, in-domain and about the
   *wrong fault* already scores 0.897 raw. A Version A below that has
   demonstrated nothing, and a Version A at 0.91 has demonstrated very little.
   The interesting quantity is the distance above `random_template`, not the
   score.
2. **Pin `--rescale-with-baseline` for the A/B comparison.** Rescaling widens
   the `constant`→`random_template` gap from 0.018 to 0.105, roughly sixfold.
   That gap is the resolution the harness has to detect a real difference
   between model versions, and the raw scale spends almost all of its range on
   text being English.

Figures: `experiments/feedback_eval/shuttleset_v1/figures/`, regenerated by
`scripts/feedback_eval/plot_shuttleset_v1.py`.

### Provenance is mandatory

Every template records where it came from, because three sources support three
different claims and the weakest is the one that quietly poses as the others:

| kind | meaning | strength |
|---|---|---|
| `publication` | quoted from a book, manual or paper; `detail` is the citation | strongest |
| `transcript` | from a coaching video; `detail` names channel, video and time | good |
| `drafted` | written by the team from general coaching knowledge, traced to no source | weakest |

`library_summary()` counts the mix so the report can state it. **The shipped
library is entirely `drafted`** — usable, honest, and the thing to upgrade
first. A test asserts this, so if the mix ever changes the claim is re-examined
deliberately rather than by accident.

### Growing the library from coaching videos

Match commentary is *descriptive* ("lovely drop shot"). Coaching videos are
*corrective* ("you're taking it too late, start your racket earlier"), which is
the language a reference set actually needs — and the gap the commentary route
cannot close.

The repo already has most of the machinery: `scraper.transcript_acquisition`
pulls speech to text with Whisper, and `scraper.commentary_cleaning` cleans a
chunk and generates alternate phrasings of it — the same multi-phrasing shape
`corrections` wants, for the same reason. Point that at coaching videos rather
than match broadcasts, then write each fault/correction pair into the library
with `kind: "transcript"` and the channel, video id and timestamp in `detail`.

Unlike the commentary route, this produces a **fault → correction** library
rather than per-clip references, which is exactly what a template set is.

## Reading the numbers

**BERTScore is not an absolute quantity.** Raw F1 sits in a compressed high band
even for unrelated text, and changing the embedding model, the language, or
baseline rescaling moves every number. Only differences between runs mean
anything, and only when the runs share a scorer.

That is enforced, not just documented: `ScorerConfig` is stamped into every
saved run, and `assert_comparable` refuses to report a delta across runs with
different scorer settings or different clip sets. `--model-type` should be
pinned explicitly for anything that goes in the report, because bert-score's
per-language default can move between releases.

`mean_f1_by_player` is printed alongside the headline mean. With a
player-disjoint split the test players are few, so one atypical player can move
the mean more than the model version does. Note these per-player means
**overlap**: a rally between two test players counts towards both, so they
diagnose an outlier rather than summing back to the headline figure.

Running without `--split` scores every clip in the reference file and warns.
That is a wiring check, not a result.

## Running it

Wiring check with the bundled stub set — no bert-score install, no model download:

```bash
python -m feedback_eval.score_cli \
    --references experiments/feedback_eval/stub/references.jsonl \
    --predictions experiments/feedback_eval/stub/predictions_version_a_stub.jsonl \
    --model-version A-stub --dry-run
```

`--dry-run` swaps in a token-overlap fake scorer. It cannot see meaning, which is
the whole reason the real harness uses BERTScore, so its numbers are never
reportable — output is stamped `"dry_run": true`.

A real run needs the optional extra (`uv sync --extra scraper`, which is where
`bert-score` already lives) and drops `--dry-run`:

```bash
python -m feedback_eval.score_cli \
    --references data/feedback_eval/references.jsonl \
    --predictions runs/version_a/predictions.jsonl \
    --split data/feedback_eval/split.json \
    --model-version A --model-type roberta-large \
    --out runs/version_a/scores.json
```

## Getting to a Version A score

Scoring all 543 held-out clips means cutting and running the model on all of
them. A seeded 100-clip subset is enough to separate a model from the floor:

```bash
export PYTHONPATH=src

# 1. the subset, written as a split file (committed: score everything against it)
python -m feedback_eval.subset_split \
    --split data/feedback_eval/split_shuttleset_v1.json \
    --references data/feedback_eval/references_shuttleset_v1.jsonl \
    --size 100 --seed 20261002 \
    --out data/feedback_eval/split_shuttleset_v1_test100.json

# 2. download just the 100 clips' spans at 720p (~18 min of video, ~80 MB, ~20 min;
#    resumable) -> data/feedback_eval/clips/ + clips.csv. Needs yt-dlp and ffmpeg.
python -m feedback_eval.shuttleset_clips \
    --split data/feedback_eval/split_shuttleset_v1_test100.json --fetch

# (alternative: whole matches through the existing ShuttleSet downloader, then cut
#  locally. 11 full broadcasts at 1080p need 10 GB or more of disk.)
#   python -m feedback_eval.shuttleset_clips --split ... --write-match-csv data/feedback_eval/test100_matches.csv
#   PYTHONPATH=src:src/bst_x python src/bst_x/pipeline/download_adapter.py \
#       --match-csv data/feedback_eval/test100_matches.csv --output-dir data/shuttleset/raw_video
#   python -m feedback_eval.shuttleset_clips --split ...
```

**Why a split file and not a smaller reference file.** `splits.select` requires
every clip a split names to be present, so a 100-clip reference file scored
against the full split is refused. The full reference file scored against the
subset split selects exactly the 100, and the player-disjoint guard still runs
because the train side is unchanged.

**Clip bounds** are ShuttleSet's own: first stroke minus 1 s to last stroke plus
2 s, converted with the fps ShuttleSet annotated at. Checked by eye on
`..._TOYOTA_THAILAND_OPEN_2021_QuarterFinals_s1_r9` against the current YouTube
upload: the broadcast is on a close-up for the serve (which is why ShuttleSet
types that stroke `未知球種`), cuts to the court about 0.3 s after the clip's
first stroke, and returns to a close-up about 0.5 s after the clip ends. A
re-uploaded or re-edited source would shift every clip silently, so spot-check
a few before running the model on all 100.

**3. Predictions** (GPU): `src/mllm/batch_feedback.py`, run through
`src/mllm/run_batch.sh` on `bourbaki` (A100 40 GB) or `carmack` (L40 48 GB) —
not `engelbart`, whose 16 GB V100 cannot hold the 8B model in bf16 (see
`docs/gpu-access.md`). It loops InternVideo3 over `clips.csv`, asks for a
correction rather than a description, writes each prediction as it goes, and
resumes after an interruption. Settings are recorded in `<out>.meta.json`.

```bash
# on the node, from src/mllm/, after copying data/feedback_eval/clips/ to /scratch
apptainer build --fakeroot container.sif container.def       # once
HF_CACHE=/scratch/comp320a/hf-cache ./run_batch.sh /scratch/comp320a/clips \
    /scratch/comp320a/runs/version_a/predictions.jsonl --limit 3   # smoke test
HF_CACHE=/scratch/comp320a/hf-cache ./run_batch.sh /scratch/comp320a/clips \
    /scratch/comp320a/runs/version_a/predictions.jsonl            # all 100
```

Read the three smoke-test outputs before the full run: if they describe the
venue instead of correcting a shot, fix the prompt (`--prompt-file`) first, and
use a new output path, because the run refuses to mix settings in one file.

**4. Score A and re-score the floor on the same 100.** The baseline figures
above are over all 543 clips and are not comparable with a 100-clip mean:

```bash
SUB=data/feedback_eval/split_shuttleset_v1_test100.json
REFS=data/feedback_eval/references_shuttleset_v1.jsonl
OUT=experiments/feedback_eval/shuttleset_v1_test100
for b in constant random_template; do
  python -m feedback_eval.baselines --references $REFS --baseline $b --seed 0 \
      --templates data/feedback_eval/templates/badminton_singles_shuttleset_v1.jsonl \
      --out $OUT/predictions_$b.jsonl
  python -m feedback_eval.score_cli --references $REFS --predictions $OUT/predictions_$b.jsonl \
      --split $SUB --model-version $b --model-type roberta-large --rescale-with-baseline \
      --out $OUT/scores_${b}_rescaled.json
done
python -m feedback_eval.score_cli --references $REFS --predictions runs/version_a/predictions.jsonl \
    --split $SUB --model-version A --model-type roberta-large --rescale-with-baseline \
    --out $OUT/scores_version_a_rescaled.json
```

Report Version A as its distance above `random_template` on the rescaled scale.
The floor on this subset is already scored (`experiments/feedback_eval/shuttleset_v1_test100/`):

| Baseline | Raw F1 | Rescaled F1 |
|---|---|---|
| `constant` | 0.879 | 0.280 |
| `random_template` | 0.894 | 0.370 |

100 clips, 11 players, `--model-type roberta-large`, baseline seed 0. Close to
the 543-clip figures (0.879 / 0.282 and 0.897 / 0.387), so the subset is not an
unusually easy or hard draw.

### Version A result (2 October 2026)

| On the 100-clip subset | Raw F1 | Rescaled F1 | Beats it on |
|---|---|---|---|
| `random_template` | 0.894 | 0.370 | — |
| `constant` | 0.879 | 0.280 | — |
| **Version A** — InternVideo3-8B, no fine-tuning | **0.853** | **0.178** | `constant` 8/99, `random_template` 3/99 clips |

Run on `bourbaki` (A100 40 GB), ~57 s per clip, settings in
`experiments/feedback_eval/shuttleset_v1_test100/version_a_predictions.meta.json`.
Excluding the one blank clip gives 0.862 / 0.180, so the blank does not drive
the result.

**Version A is below both model-free baselines.** It writes in a coaching
register but does not diagnose the rally: 54 of 99 answers open with the same
sentence ("The final shot was a high, soft backhand lift…"), there are only 25
distinct opening sentences, and a keyword check finds the right stroke family
in the opening sentence on 24 of 99 clips — no better than always answering
"lift or drop" (22 of these clips). Its invented specifics ("the opponent then
won with a drop shot") cost precision, which is why it falls below generic
advice. That is the gap Version B's fine-tuning is meant to close.

**One clip is blank.** `..._TOYOTA_THAILAND_OPEN_2021_QuarterFinals_s2_r25`,
the longest clip at 39.5 s, fails in torchvision's decoder every time
(`swscaler: Failed initializing scaling graph`), at 720p and when shrunk to
360p, and was written with `--blank-failures`. The container has no
`torchcodec`; rebuilding it with one is the likely fix.

## Not done yet

- **Version B.** Version A is scored (above). The next model version needs
  Ego-Exo4D access and a LoRA fine-tuning setup; score it against the same
  subset file so the comparison holds.
- **`torchcodec` in the container.** One Version A clip could not be decoded by
  the torchvision fallback; see *Version A result*.
- **Excluded matches in the reference set.** `video_metadata.csv` marks four
  matches as excluded (ids 9, 10, 12: labelling or frame numbers incorrect;
  27: video removed). `shuttleset_faults` does not read that file, so 192 of
  the 2247 reference clips come from them. All four are men's matches and sit
  on the train side, so no test score is affected, but the 128 clips from 9, 10
  and 12 should be dropped before any training run uses the train side.
- **Attaching the cause-keyed templates to clips.** `badminton_singles_v1.jsonl`
  (16 cause-keyed templates: late preparation, no trunk rotation, …) still has
  no `clip_faults.csv`, and deriving one from ShuttleSet is exactly the
  inference the outcome-keyed route was built to avoid. It needs someone to
  watch clips.
- **Upgrading template provenance.** All 32 shipped templates across both
  libraries are `drafted`. Coaching-video transcripts and published coaching
  manuals both raise that.
- **A commentary reference set.** The adapter is written and tested against the
  scraper's file contracts, but **nobody has run the pipeline that produces
  them**: there is no `data/scrape_output/`, no `rally_commentary_pairs.csv` and
  no `chunks/` anywhere in the repo. `src/scraper/commentary_pairing.py` and its
  config are on this branch and runnable; the artifacts simply do not exist yet.
  Until they do, `experiments/feedback_eval/stub/` is eight hand-written clips
  for testing the wiring, and no result should come from them.
- **`players.csv`.** Its source is `configs/players.csv` plus
  `rallies.top_player_id` / `bottom_player_id` on `origin/issue-18-schema-freeze`
  (43 named players, court-side resolution 98.9%), which is **unmerged**.

## Tests

```bash
python -m pytest tests/test_feedback_eval_*.py
```

All 175 run on CPU in well under a second with no transformers install: the
scorer is injected, so the tests drive a fake with the same signature as
`bert_score.BERTScorer.score`.

## The readout page

`scripts/feedback_eval/build_readout_payload.py` rebuilds the interactive
readout from the harness's own outputs — the score runs, the pinned split, the
derived faults and the template library — and inlines it into
`readout_template.html`:

```bash
PYTHONPATH=src python scripts/feedback_eval/build_readout_payload.py \
    --out /tmp/payload.json --html-out /tmp/harness_readout.html
```

Nothing on the page is typed in. The exclusion counts in particular are
produced by re-running `derive_faults` rather than copied out of its console
output, because a transcribed count goes on looking plausible long after the
thing it counted has changed.

The page is a **snapshot, not a live view**: re-running the scorer does not
update a published readout. Rebuild and republish.
