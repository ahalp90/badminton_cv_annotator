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

## The three commands

```bash
# 3.1  build a reference set from the COSC595 commentary lane
python -m feedback_eval.commentary_references --pairs ... --chunks-dir ... --players ... --out ...

# 3.2  partition players once, and keep the file
python -m feedback_eval.split_cli --references ... --seed 20260903 --out ...

# 3.3  score one model version on one side of that split
python -m feedback_eval.score_cli --references ... --predictions ... --split ... --model-version A
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

## Not done yet

- **A real reference set.** The adapter is written and tested against the
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
python -m pytest tests/test_feedback_eval_records.py \
                 tests/test_feedback_eval_scoring.py \
                 tests/test_feedback_eval_splits.py \
                 tests/test_feedback_eval_commentary_references.py \
                 tests/test_feedback_eval_cli.py
```

All 94 run on CPU with no transformers install: the scorer is injected, so the
tests drive a fake with the same signature as `bert_score.BERTScorer.score`.
