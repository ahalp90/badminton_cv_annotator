# C2 WebUI seed

This is a small, uploadable evidence seed for the two C2 traces. It records
the raw-record verdicts without bundling the multi-million-court arrays, native
frames or the 530 MB courts-before-the-gate side files.

## Start here

- `findings.md`: bounded verdict, four Check A rows, and the corrected Check B survivor chain.
- `witnesses.json`: exact source paths, MD5s, control corners, scales, candidate identities and replay gates.
- `ranking_panel.json`: 17 distinct automatic winners, observed-bank controls 1864 and 5144, and candidate 89 as a separately labelled comparator. Original visual-ruling text is retained.
- `check_traces.py`: the single-process local check that produced the witnesses. It expects the repository-relative raw inputs named in the witnesses.

## Contents

- `source/`: the frozen matcher and directly relevant producer, observation,
  detector and replay sources. `source/SOURCE_MAP.md` gives original paths and
  line ranges.
- `tables/`: compact saved axis-replay and filter-replay tables.
- `reports/`: the corrected line-identity reports and the small WebUI/follow-up assessments.

Distances in the findings use working pixels at 960 by 540. Native dimensions
are 1920 by 1080. The GX0 control is the visually approved candidate-89
reference; the Amateur-3 control is the manual reference recorded by the
direction experiment.

## Reproduce from the repository root

```text
~/.venvs/badminton-cicd/bin/python \
  scratch/court_det_fix/next_steps_20260916/C2_traces/check_traces.py \
  --output scratch/court_det_fix/next_steps_20260916/C2_traces
```

The check reads the saved records and inputs listed in `witnesses.json`. It
does not launch a remote job or regenerate the full matcher population.
