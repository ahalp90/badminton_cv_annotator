# Net evidence and court homography: frontier handover

This pack captures the branch history, the bounded Am1/GX audit, and the most
promising open directions around using a badminton net to help situate a court.
It is intentionally descriptive rather than prescriptive. The local agent has
room to reinterpret the evidence, replace the proposed measurements, and pursue
stronger formulations.

## Provenance

- Current `fix/court-det` tip observed during this handover:
  `1353541e9e0f08633a12c746f9690785af543aa5`
- Revision used by the frozen Am1/GX packet:
  `13f02fdbf8954ead15dbc7241a696a314473f9c5`
- Commit introducing the historical projected-net experiment:
  `be94d28dbc78289aa21386bbefcf059d416f407e`
- Production edits from this WebUI work: none

## Core finding

Net evidence has existed in the branch, although not as an independent semantic
net detector. The historical arm projected regulation net geometry from each
court candidate, measured generic line support near that projection, and mixed
it into floor scoring. It produced one very large improvement, one regression,
and one neutral result. The modern W5 path retains the camera plausibility part
but not the image-level net support term.

The strongest conceptual opportunity is therefore not simply “add the old net
score back.” A more independent net observation could add non-coplanar evidence,
help associate a candidate with the played court, and prevent net pixels from
being reused as floor-marking evidence. The main risks are circular scoring,
multiple visible courts and nets, cropping, weak post visibility, wall rails,
and double-counting the same fragments.

## Local Am1/GX evidence

The supplied Am1 image confirms that the false far baseline follows the lower
white net band. GX remains an important partial-view control: a valid court
corner lies outside the frame, and no retained fragment is assigned to its far
baseline.

Two image tests were carried through:

- A projected attached-mesh falsifier failed to distinguish Am1 from GX.
- A local same-paint chroma comparison strongly rejected Am1's claimed baseline
  while abstaining on GX because GX had no supported far-baseline fragments.

The chroma result is exploratory. It is useful as evidence that semantic
material consistency can separate the cases; it is not a calibrated general
net detector.

## Pack map

- `docs/01_EXISTING_NET_WORK.md` — what the branch already tried and where it
  lives.
- `docs/02_AM1_GX_FINDINGS.md` — verified local findings, negative results, and
  the chroma lead.
- `docs/03_FRONTIER_DIRECTIONS.md` — open research avenues and their likely
  information value.
- `docs/04_REPO_MAP.md` — branch paths, replay bundles, tests, and likely data
  sources.
- `docs/05_ARTIFACT_NOTES.md` — included scripts, inputs, outputs, and replay
  semantics.
- `scripts/` — the two bounded WebUI analyses.
- `data/` — frozen case packet and the two supplied source images.
- `results/` — machine-readable and text outputs.
- `overlays/` — fragment and boundary visualisations.
- `SHA256SUMS.txt` — integrity manifest for the pack contents.

## Status vocabulary

- **Verified**: reproduced from frozen inputs or directly measured from the two
  images.
- **Historical**: recorded branch experiment, with the limitations stated in
  its own replay material.
- **Exploratory**: a locally useful result without broad calibration.
- **Negative evidence**: a tested formulation that did not distinguish these
  cases; not a claim that the wider idea is exhausted.
- **Open**: a direction whose value remains unresolved.
