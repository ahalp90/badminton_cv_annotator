# Court detector: start here

Use **[pickup.md](pickup.md)** for a quick account of where the detector stands
and what to do next. Use **[DETECTOR_DECISIONS.md](DETECTOR_DECISIONS.md)** for the
detailed experiment history, evidence and rulings. They serve different readers.

## Choose a document

| Need | Read | Purpose |
| --- | --- | --- |
| Skim current state or resume work | [pickup.md](pickup.md) | Current result, blockers, next action and working state |
| Understand what was tried and decided | [DETECTOR_DECISIONS.md](DETECTOR_DECISIONS.md) | Detailed, concise-per-item findings with evidence and contrary cases |
| Find the implementation, inputs or a rerun command | [FP_INDEX.md](FP_INDEX.md) | Core code/data map; fresh-generation and historical-replay entry points |
| Start the next optimisation session | [Shared contract](handover_20260923/00_SHARED_CONTRACT.md), then [launch prompt](handover_20260923/01_LAUNCH_OPTIMISATION.md) | Permissions, bounded intake, first work and verification gates |
| Check net-weight statistics | [Paired comparison](net_recovery/statistics/paired_reference_report.md) | Measures, source sensitivity, per-frame results and limits |
| Follow the net experiment in depth | [Net worklog](net_recovery/WORKLOG.md), [assessment](net_recovery/ASSESSMENT.md) | Run history, user rulings, mechanisms and independent audit |
| Recover earlier material | [Archive map](archive/README.md) | Archived records and sealed recovery routes |

The detailed decision history owns the experiment narrative. This index maps
documents; the filepath index maps implementations and artefacts. Neither needs
a second copy of the chronology.

<a id="recovery-not-another-reading-path"></a>

## Recovery

The [archive map](archive/README.md) covers previous retirements and
[sealed recovery](archive/README.md#earlier-sealed-recovery). The
[23 September top-level refresh](archive/20260923_top_level/README.md) preserves
pre-refresh prose, the disposition record and the inventory of retained data.
Original experiments and their worklogs remain at their existing locations.

## Keep the roles distinct

- INDEX stays a thin entry map; current state belongs only in pickup
- Rewrite pickup at handover; archive its previous account when restructuring
- Put durable findings and rulings in DETECTOR_DECISIONS, linked to source evidence
- Put core paths, commands and local-data requirements in FP_INDEX
- Keep full results and worklogs beside their experiment; preserve worklogs intact
- Launch packets describe a bounded phase; they do not become another live status log
- Track small reproducible results and required inputs deliberately; document exact
  local-only data paths and producers rather than broadly ignoring new evidence
- Numerical results lead evaluation; visual review answers a few material questions
- Preserve IDs, units, image identity, contrary cases and annotation limitations
- Link to stable decision/report sections, not numbered items in the changing pickup
