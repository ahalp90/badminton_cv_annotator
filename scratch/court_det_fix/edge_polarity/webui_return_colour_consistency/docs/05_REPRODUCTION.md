# Reproduction

## Environment

The supplied outputs were reproduced with:

```text
Python 3.13.5
NumPy 2.3.5
OpenCV 4.13.0
```

The scripts should work on Python 3.10+ with NumPy and OpenCV 4.x.

Install into a clean environment:

```bash
python -m pip install -r requirements.txt
```

## Inputs

- `data/cases.json.gz` — frozen two-case packet.
- `data/images/am1_frame54_user_supplied.png`
- `data/images/gx_frame5_user_supplied.png`

The images are the 1920×1080 files supplied in the WebUI conversation. They
were sufficient to reproduce the reported image measurements. They are
re-encoded RGBA files and were not byte-verified against the pinned GitHub PNG
blobs because the connector exposed binary metadata rather than downloadable
bodies. A local agent that already has the repository may replace them with the
raw pinned files:

```text
scratch/court_det_fix/frozen_views/frames/amateur/am1/frame_00000054.png
scratch/court_det_fix/frozen_views/frames/gx/images/gxBQ_window_00_frame_00000005.png
```

## Run everything

```bash
bash run_all.sh
```

Generated files are written under `results/generated/` and
`overlays/generated/`.

## Run the geometry and negative-control audit

```bash
python scripts/net_boundary_bounded_audit.py \
  data/cases.json.gz \
  --am1-image data/images/am1_frame54_user_supplied.png \
  --gx-image data/images/gx_frame5_user_supplied.png \
  --mesh-test \
  > results/generated/net_boundary_bounded_audit_with_images.txt
```

This reports:

- packet hashes and revision;
- frozen assignment counts;
- recorded strong-centre fit and gate values;
- projective-depth counterexample;
- projected physical-net support negative control;
- actual attached-mesh result.

## Run the image follow-up

```bash
python scripts/boundary_paint_followup.py \
  data/cases.json.gz \
  --am1-image data/images/am1_frame54_user_supplied.png \
  --gx-image data/images/gx_frame5_user_supplied.png \
  --output results/generated/boundary_paint_followup_results.json \
  --overlay-dir overlays/generated
```

The canonical machine-readable output is
`results/boundary_paint_followup_results.json`.

## Overlay convention

- red: fitted far boundary;
- yellow: usable fragments assigned to the far baseline;
- cyan: usable transverse paint-reference fragments;
- numeric labels: frozen raw fragment IDs.

## Determinism check

The image follow-up was rerun in the container. The regenerated JSON was
byte-identical to the canonical JSON, and both regenerated overlay hashes were
identical to the canonical overlay hashes.

## Packet hashes

```text
gzip SHA-256: dd5689b8882bdc6c6c90e3e26935420c0e055a5be15937ef7250907aff7bdd03
JSON  SHA-256: 8155a82b1c7597bafd9d8d0991b9ac603f4cef61c18978f433738ec1fff11389
```
