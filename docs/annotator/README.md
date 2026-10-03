# Auto-annotator

The auto-annotator turns saved shuttle, pose and court data into rally spans, contact frames, court-half assignments and several outcome estimates. The dataset builder runs the same code before it produces rally records.

The documents are split by task:

| Task | Document |
| --- | --- |
| Run one already-extracted video | [Quickstart](quickstart.md) |
| Understand the full annotation chain | [Overview](how_it_works.md) |
| Understand the fixed rules before any tree runs | [Fixed heuristics](heuristics.md) |
| Understand what each fitted tree sees and passes onward | [Tree model stack](tree_stack.md) |
| Check the exact files, array shapes and result fields | [Inputs and outputs](inputs_outputs.md) |
| Fit and evaluate a new model set | [Refit guide](retuning.md) |
| Work out whether a code change affects fitted models | [Maintainer guide](maintaining.md) |
| Find the module responsible for a piece of the pipeline | [Code map](code_map.md) |

## The short version

The annotator has two main parts.

1. **Rules narrow the search.** They reject unreliable frames, flag unreliable shuttle positions, form rough rally spans, track one player on each court half, and mark the parts of the timeline where a contact looks plausible. [Fixed heuristics](heuristics.md) explains those rules and the main thresholds.
2. **Tree models choose contacts and repair sequences.** A contact model scores plausible hit frames. Sequence models can repair an early serve, remove a bad hit or add one likely missed later hit. A final tree gives each completed rally a review score. [Tree model stack](tree_stack.md) explains what each model sees and how the scores pass between them.

After the contact sequence is settled, rule-based code adjusts rally bounds and assigns alternating `Top`/`Bot` court halves. It also estimates server, winner, landing and hit height when enough evidence is available.

The standalone command for one video is:

```bash
PYTHONPATH=src uv run python -m annotator \
  --run-dir data/dataset-run \
  --video-id match-name \
  --models models/annotator \
  --output-dir data/annotations/match-name
```

This command expects metadata, shuttle, pose and court extraction to be finished already. Raw footage goes through the dataset builder first; the annotator is one stage inside that larger pipeline. The [quickstart](quickstart.md) covers the required files, the outputs and the common failures.

`python -m annotator.rally_segmentation` is an older command that only runs the rule-based rally segmentation. Full annotation is `python -m annotator`.

## Current status

Annotation needs a model directory fitted in the current environment. Old experiment binaries are comparison material. Copying them into `models/annotator` does not make them supported models.

A refit and validation queue using the released new court detections was launched on 3 October 2026. Results and model selection are pending, so these documents make no claim about annotation quality on those inputs. The [refit guide](retuning.md#planned-new-court-refit) describes the comparison.

For background, [PR 149](https://github.com/ahalp90/badminton_cv_annotator/pull/149) describes the retained mixed rule-and-tree model and [PR 150](https://github.com/ahalp90/badminton_cv_annotator/pull/150) its court-failure analysis. Both describe the old court inputs.

## Run settings and model settings

Each model directory contains the fitted trees plus the settings used with them. These include the contact score cutoff, an optional rule for unreliable shuttle frames, the rough-rally and masking settings, and the choice of video-level or scene-level court geometry for player sides.

Two kinds of setting matter:

- **Run selection** — which saved extraction run, video, model directory and output directory to use.
- **Model settings** — settings that change contact features, candidate sequences or values seen by the trees. A change here usually means fitting and evaluating a new model directory.

The [maintainer guide](maintaining.md) lists common changes and whether they affect fitted models.

## Main source files

These files cover most of the maintained annotation path:

- [`cli.py`](../../src/annotator/cli.py) — standalone command for saved inputs
- [`run_video.py`](../../src/annotator/run_video.py) — full one-video annotation chain
- [`hybrid.py`](../../src/annotator/hybrid.py) — contact scoring, sequence refinement and rally review score
- [`sequence/refine.py`](../../src/annotator/sequence/refine.py) — the three sequence chooser stages
- [`models.py`](../../src/annotator/models.py) — model directory contents and load-time checks
- [`training/workflow.py`](../../src/annotator/training/workflow.py) — fitting and held-out evaluation
- [`dataset_builder/vision.py`](../../src/dataset_builder/vision.py) — saved-input loaders and persisted annotation output

The [code map](code_map.md) covers the rest of the package. Research experiments live under [`experiments/annotator/`](../../experiments/annotator/README.md) and are separate from this reading path.
