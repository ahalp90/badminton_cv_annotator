# Included artifact notes

## `scripts/net_boundary_bounded_audit.py`

Scope:

- frozen Am1/GX packet inspection;
- fixed-fit status and gate reporting;
- projective depth ambiguity construction;
- retained-fragment support for the projected physical net;
- optional outward attached-mesh image test.

The script contains no candidate search and changes no assignments. The
attached-mesh test is retained because its contrary result is informative.

A representative invocation from the pack root is:

```text
python scripts/net_boundary_bounded_audit.py \
  data/cases.json.gz \
  --am1-image data/images/am1_frame54.png \
  --gx-image data/images/gx_frame5.png \
  --mesh-test
```

## `scripts/boundary_paint_followup.py`

Scope:

- rerun of the attached-mesh test;
- per-fragment local Lab ridge signatures;
- same-paint chroma comparison;
- result JSON and overlays.

A representative invocation from the pack root is:

```text
python scripts/boundary_paint_followup.py \
  data/cases.json.gz \
  --am1-image data/images/am1_frame54.png \
  --gx-image data/images/gx_frame5.png \
  --output /tmp/boundary_paint_results.json \
  --overlay-dir /tmp/boundary_paint_overlays
```

## Inputs

```text
data/cases.json.gz
data/images/am1_frame54.png
data/images/gx_frame5.png
```

The two PNGs are the WebUI-supplied images used by the local calculations. They
have the requested 1920 × 1080 dimensions and visually match the target frames.
They may be re-encoded relative to the exact GitHub blobs.

## Results

```text
results/net_boundary_bounded_audit_output.txt
results/boundary_paint_followup_results.json
```

The JSON contains full per-fragment measurements, not only the summary table.
It preserves the fragment IDs, interval identities, strong sample counts,
contrast statistics, local Lab increments, mesh-test output, and final local
status for both cases.

## Overlays

```text
overlays/am1_window_00_frame_54_paint_overlay.png
overlays/gxBQ_window_00_frame_5_paint_overlay.png
```

Overlay semantics:

- red: fitted far boundary;
- yellow labels and segments: usable far-baseline fragments;
- cyan labels and segments: usable transverse court-paint references.

## Dependencies

The scripts use Python, NumPy, and OpenCV. They are analysis utilities rather
than production modules.

## Interpretation boundary

The included analyses answer a two-case semantic ambiguity question. They do
not contain a full candidate population, an independent net detector, or a
calibrated acceptance threshold. Their main value lies in preserving exact
negative results, exposing the semantic gap in the current evidence, and
providing concrete image measurements that a broader approach can supersede.
