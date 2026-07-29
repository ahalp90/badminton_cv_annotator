# Annotator cleanup: per-commit worklog

Authentic per-commit record of the cleanup chain, drawn from `git log` on
`feature/commentary-scraper`. Commit subjects are quoted verbatim; each row
adds one sentence of intent.

| SHA | Subject | Intent |
|---|---|---|
| `6ebf9cc` | Derive fixture geometry from tracked calibration data | Load geometry and resolution for the three fixtures through one shared derivation path instead of copied literals; approved homography-derived bounds replace the unsourced pilot occupancy values. |
| `d355019` | Remove the frozen sweep, scorer and wrist tools | Remove ~4,268 lines of retired Stage 8 sweep, scorer, and wrist-analysis scripts and their tests as one dependency-safe unit. |
| `c19c355` | Remove the old rally shim and archive frozen harnesses | Delete `stage8_rally_segmentation.py` and move four historical scripts into `scripts/archive/`; the live S28 anchor-picks harness stays put and unchanged. |
| `d2f2fbb` | Remove the remaining Stage 2 compatibility shims | Delete the five modules and the re-export block that preserved retired Stage 2 import paths; live imports point at their canonical locations. |
| `0f2c090` | Remove the retired serve and wideshot APIs | Remove the old serve-start and wideshot helpers, options, constants, and single-purpose tests; keep the court-scaling helpers and update their tests to the surviving contract. |
| `f4e69e1` | Remove dead calibration settings and pilot geometry residue | Remove the unused direction-change threshold and every surface that carried it; delete `pilot_geometry.py` and require the minimum-descend setting to be passed explicitly. |
| `3f6e34d` | Route batch processing through run_video | Replace the batch command's separate path with a thin wrapper around `run_video`; retire pose-gated batch mode, `--thresholds`, and the rally-span sticky fallback. |
| `1e372c2` | Replace the old player geometry with CourtGeo | Remove the retired player-height filter and duplicate body-distance helpers; rename the surviving three-part geometry and fixture field to `CourtGeo` and `court_geo`. |
| `7a7337a` | Move annotator dependencies to their proper homes | Move four path constants into annotator-owned configuration and repoint point-winner imports at their surviving public locations, removing the related circular dependencies. |
| `eee3e29` | Enable gap classification and invisible-frame smoothing | Ship `IGNORE_INVISIBLE / base-30 75 / TWO_SIDED / 0.05` together in `BaseAnnotatorConfig`; one live reference moved and received an immediate re-pin. Measurement record: `w2_9_delta.diff`. |
| `85b8751` | Re-pin: live three-fixture calibration capture | Refresh the scratch capture and `REFERENCE_SCORES` to the post-flip numbers. Contact F1 improved on all three fixtures; the expected span, server, and winner movements are captured in the diff. |
| `806ef2c` | Close the Wave 2 documentation residue | Mark the retired S28 pin harness as historical evidence and remove stale references to the deleted sweep patcher; no runtime path changes. |
| `93477bd` | Use canonical names for fixtures and local data files | Rename the three local fixture identities to `sset_01`, `sset_15`, and `sset_21`; derive local paths from `Fixture.name`. Every external file retained its pre-move MD5. |
| `9f8b59f` | Archive the frozen autoseg trial records | Move the retired S28–S29 scripts and their four frozen CSVs to `scripts/archive/autoseg_trials/`; bytes preserved, no runtime behaviour changed. |

## Non-commit checkpoints

- **W3.2 external-record refresh.** After W3.1 the reviewed live capture
  was normalised to use the canonical stems (`pilot → sset_01`,
  `vid15 → sset_15`, `sset21 → sset_21`). The normalised old record and the
  reviewed canonical capture were byte-equal, so no empty Git commit was
  created.
- **Final handoff.** The runbook allowed one closing commit if it carried
  tracked changes. The handoff files sat under gitignored `local_scratch/`,
  so the commit was omitted per the runbook's rule and the handoff was
  recorded outside Git.

## Final gate at tip `9f8b59f`

Ruff 0, Pyrefly 0 (7 pre-existing suppressions), pytest 1,267 passed / 20
skipped / 33 warnings, calibration merge smoke exit 0, live capture
matched the then-current scratch reference. The maintained capture now
lives under `tests/data/annotator_calibration/reference/`.
