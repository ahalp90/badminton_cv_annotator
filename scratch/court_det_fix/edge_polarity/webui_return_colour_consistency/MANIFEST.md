# Pack manifest

Created for the bounded Am1/GX court-boundary versus net-tape handover.

- Repository revision: `13f02fdbf8954ead15dbc7241a696a314473f9c5`
- Frozen packet schema: `net-versus-court-bounded-audit/1`
- Pack creation date: 2026-09-23
- Production edits: none

## Contents

| Path | Purpose |
|---|---|
| `README.md` | Entry point, verdict and reading order. |
| `docs/01_SCOPE_AND_EXECUTIVE_SUMMARY.md` | Bounded question, conclusion and result table. |
| `docs/02_TRIED_AND_DISCARDED.md` | Failed, unsafe or merely diagnostic ideas. |
| `docs/03_ACCEPTANCE_PATH_AND_IDENTIFIABILITY.md` | Actual caller path, gate semantics and counterexample. |
| `docs/04_MEANINGFUL_FINDING_SAME_PAINT_CHROMA.md` | Full discussion of the useful exploratory veto. |
| `docs/05_REPRODUCTION.md` | Environment, commands, inputs and deterministic rerun notes. |
| `docs/06_NEXT_STEPS.md` | Recommended local validation and integration sequence. |
| `docs/07_SOURCE_ANCHORS.md` | Pinned repository paths and function anchors. |
| `scripts/net_boundary_bounded_audit.py` | Frozen geometry audit, depth construction and negative controls. |
| `scripts/boundary_paint_followup.py` | Image tests, chroma veto and overlay rendering. |
| `run_all.sh` | One-command rerun and canonical JSON comparison. |
| `requirements.txt` | Minimal Python dependencies. |
| `data/cases.json.gz` | Exact frozen two-case packet used by both scripts. |
| `data/images/*.png` | User-supplied 1920×1080 image files used for the image measurements. |
| `results/net_boundary_bounded_audit_with_images.txt` | Complete geometry/negative-control output, including the contrary mesh result. |
| `results/boundary_paint_followup_results.json` | Full machine-readable per-fragment image measurements. |
| `overlays/*.png` | Visual location of fitted boundary and selected fragment groups. |
| `CHECKSUMS.sha256` | SHA-256 for every file in the unpacked handover. |

## Input image provenance

The WebUI-supplied images were used directly. They visually and dimensionally
match the requested cases, but their binary SHA values do not match the GitHub
blob identifiers because the uploads are re-encoded RGBA files. Replace them
with the pinned raw repository images when exact source-byte identity is needed.

- `data/images/am1_frame54_user_supplied.png`: 1920×1080, mode `RGBA`, SHA-256 `50acf4f054bb952855a399ef6f64c7ea553c8a0be4cb6ffe25858fad4b5fb9d2`
- `data/images/gx_frame5_user_supplied.png`: 1920×1080, mode `RGBA`, SHA-256 `53ff1a9dfb63593a24515ad3124429eddf1b014e433c5cd05d9b3e1ae241a38a`


## Integrity

Run from the pack root:

```bash
sha256sum -c CHECKSUMS.sha256
```

Then run:

```bash
bash run_all.sh
```

The second command verifies that the regenerated boundary-paint JSON is
byte-identical to the canonical result in this pack.
