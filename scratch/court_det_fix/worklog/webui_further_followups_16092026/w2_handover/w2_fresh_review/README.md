# W2 fresh-data handover

The two-pair pixel export is complete. Start with `../pickup.md` for the current
status. `REVIEW.md` is the pre-export analysis that motivated the follow-up.

The substantive result is that the simple connected-run
replacement scores false Am2 `184:4123` above approved `30:33`, while scene19 still
separates strongly on its common five markings.

## Reproduce the calculations already performed

```bash
python analyze_fresh.py
python -m unittest discover -s tests -v
```

The analysis uses Python's standard library. Image tests and the bounded follow-up
use NumPy and OpenCV from the existing environment. The test log and actual results
are in `results/`. The fresh-data analysis did not view source pixels. The completed
follow-up replayed four fixed intervals from the pinned native frames and wrote the
unpacked evidence to `w2_pair_atlas/`.

## The completed follow-up

`LUNA_EXECUTOR.md` records the bounded execution contract. The export used exactly two
Am2 candidates, four target intervals and three hash-pinned input files. It did not
change scoring, extract new fragments, refit a court or replay full pools. The ordinary
files in `w2_pair_atlas/` are the committed paired pixel witnesses. The runner's ZIP is
transient and is omitted from the commit.

## Files

`analyze_fresh.py` validates and analyses the uploaded CSV. `build_pair_atlas.py`
executes the pixel export. `w2_probe.py` is the unchanged earlier helper, used for
replay consistency. `inputs/real_interval_probe.csv` is the current upload;
`inputs/l2_comparison.csv` and `inputs/prior_source_index.json` are explicitly prior
provenance, not fresh pixel evidence. `source_manifest.json` and
`results/execution_status.json` separate supplied, executed and unexecuted scopes.
