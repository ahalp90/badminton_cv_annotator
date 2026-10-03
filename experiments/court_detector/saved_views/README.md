# Fixed-frame detector checks

These tools inspect detector behaviour on saved images, line observations and
player boxes. They are separate from the released whole-video court dataset.

| File | Purpose |
| --- | --- |
| `frozen_cases.py` | Loads saved images and line observations for detector tests and the independent-court experiment. The three input files are in `data/packs/`. |
| `run_views.py` | Runs selected saved views through the detector; can compare detailed outputs with a supplied baseline. Requires player-evidence records and their source videos. |
| `extract_window_people.py`, `build_feet_variants.py` | Research helpers whose sampling rules are compared with the production feet code by a focused test. |

Checked-in inputs live under `data/`: the three input files and their image-source record, source frames for all 27 registered cases and all 28 runner views, plus the
frame lists and checks used to prepare them. Nine control frames include views
where the detector should reject a court. Test-only baseline snapshots live under
`tests/fixtures/court_detector/`.

The runner lists its arguments with:

```bash
PYTHONPATH=.:src python -m experiments.court_detector.saved_views.run_views --help
```

The detector [usage guide](../../../docs/court_detector/usage.md) describes
the production pipeline. The [independent-court experiment](../../annotator/independent_court/README.md)
uses the same fixture loader.
