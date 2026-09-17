# W2 paired-pixel return review

Start with **REVIEW.md**. This handover closes the two-pair diagnostic; it does not request another remote job.

- `evidence/`: the 19 original files from the user's return archive, unchanged. Open `evidence/atlas.html` to inspect both pairs.
- `verify_return.py`: standalone numerical replay from the returned working PNGs and four detailed interval traces.
- `test_verify_return.py`: four constructed mathematical checks and two real-packet checks; all six passed in this review.
- `results/verification.json`: independent geometry, interpolation, mask, crop and shared-source-pixel calculations. It also carries the producer manifest, explicitly labelled as producer-reported.
- `results/all_offset_checks.csv`: all 480 offset tests, including the one local-runtime threshold difference.
- `results/visual_readout.json`: bounded human-readable image interpretations, kept separate from calculations.
- `inputs/real_interval_probe.csv`: the prior uploaded 48-row table, unchanged.

No new scorer, candidate, label-based selection, acceptance threshold or model run is included. No repository or network access is needed for reproduction; use an existing NumPy/OpenCV environment.

```bash
python verify_return.py
python -m unittest -v test_verify_return
```

The manifest in the returned evidence identifies an edited atlas-producing helper; preserve the actual executed version when committing the local work. It is not included in the returned archive. Writing outputs inside the repository is not itself an evidence problem.
