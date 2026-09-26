# Court detector: start here

Read **[pickup.md](pickup.md)** to resume work. Each document below owns one
kind of information.

| Need | Document | What belongs there |
| --- | --- | --- |
| Current state and next work | [pickup.md](pickup.md) | The only live handover; update it when parking a session |
| Why a choice was made | [DETECTOR_DECISIONS.md](DETECTOR_DECISIONS.md) | Dated decisions, rejected ideas and links to evidence |
| Code, data or a rerun entry point | [FP_INDEX.md](FP_INDEX.md) | Paths and data requirements |
| Use the detector | [Detector guide](court_detector/README.md) | Inputs, outputs, algorithm, settings and commands |
| Build the speed-ups | [Speed-up design](court_detector/PERFORMANCE.md) | Proposed changes, agreed constraints and checks; no running status log |
| Recover older work | [26 September archive map](archive/20260926/README.md) | Former paths, intact worklogs and experiment scripts |
| Recover the former detector source | [27 September code archive](archive/20260927_code/README.md) | Old implementations and the map to their maintained replacements |
| Recover earlier cleanups | [Older archive map](archive/README.md) | Earlier moves and sealed recovery files |

## Keep the roles separate

- Put current work and unresolved blockers in pickup; rewrite it at handover
- Add a dated decision when a trial changes what should be kept or tried
- Record a run once, beside its outputs; link to it instead of copying its tables
- Keep API facts in the detector guide and future designs in the speed-up design
- Freeze completed worklogs; their old plans do not become new tasks
- Archive superseded handovers after extracting useful decisions and open work
- Use ordinary descriptions first. G0 means searching all line fragments;
  G1 means searching only paint-like fragments. Use code labels where they help
  locate code or a saved court, not as a substitute for explaining the work
- Keep this index a map. It does not need progress reports

<a id="recovery-not-another-reading-path"></a>

## Recovery

Large data remains at its existing paths. Several dated folders still supply
inputs to the detector; [the file map](FP_INDEX.md#code-and-input-paths-to-keep-stable)
explains why they stay. Archived scripts may contain outdated paths; read the
[archive notice](archive/20260926/README.md#rerunning-an-archived-experiment)
before attempting a rerun.

The [26 September record](archive/20260926/README.md#recovery-and-checks)
identifies the verified pre-tidy snapshot. Older recovery archives remain
listed in [archive/README.md](archive/README.md#earlier-sealed-recovery).
