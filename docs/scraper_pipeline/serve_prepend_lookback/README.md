# Serve-prepend lookback

This directory is the current measurement package for the deferred serve-prepend feature.

- [issue_28_serve_lookback_decision_20260809.md](issue_28_serve_lookback_decision_20260809.md)
  gives the closeout recommendation, verified evidence, remaining-work value, caveats,
  and source index
- [serve_prepend_lookback_20260808_measurement.md](serve_prepend_lookback_20260808_measurement.md)
  records the three-video reviewed-truth result and the decision not to build the measured
  central-pose prototype
- [data/serve_prepend_lookback_189c5af_20260808/](data/serve_prepend_lookback_189c5af_20260808/)
  contains its compressed candidate, counterfactual and baseline evidence
- [serve_prepend_lookback_20260731-091227.md](serve_prepend_lookback_20260731-091227.md) is the
  earlier current-code orientation and build note
- [measure_serve_prepend_lookback.py](measure_serve_prepend_lookback.py) runs the fixture-chain
  measurement against an explicit three-video fixture profile without changing production outputs
- [build_rally_start_audit_guide.py](build_rally_start_audit_guide.py) strictly joins the issue-28
  target rallies to ball-round-1 source rows and builds reviewed pilot rows and compact full-audit
  decision seeds
- [data/rally_start_visibility_audit_20260809/](data/rally_start_visibility_audit_20260809/)
  contains the deterministic 136-row target package, joined 32-row quality/control pilot, and
  per-video seeds with 32 reviewed and 104 pending compact decisions
- [data/rally_start_visibility_review_20260809/](data/rally_start_visibility_review_20260809/)
  contains the keyed primary human decisions for the pilot
- [rally_start_visibility_audit_runbook_20260809.md](rally_start_visibility_audit_runbook_20260809.md)
  records the completed disposable-copy pilot and the full-audit companion commands
- [rally_start_visibility_pilot_report_20260809.md](rally_start_visibility_pilot_report_20260809.md)
  reports the completed four-state pilot and its sampling limits
- [`src/annotator/rally_start_event_annotator.py`](../../../src/annotator/rally_start_event_annotator.py)
  reviews proposal-keyed rally starts while keeping timeline inputs read-only
- [data/serve_prepend_lookback_20260731-040847/](data/serve_prepend_lookback_20260731-040847/)
  contains the gzip CSV/JSON and native NumPy-over-XZ/LZMA-9 evidence pack

The archived design record is context only:
[../../archive/serve_prepend_lookback.md](../../archive/serve_prepend_lookback.md). It is not a
current specification or source for current figures.

The exploratory run used the current committed-mask chain and a sensitivity control that passes a
per-frame `raw_exclusion_mask = False` vector to disable replay/cutaway masking. Other
processing and downstream filters remain active. No production code or default behaviour changed.

The 2026-08-08 follow-up uses the reviewed broadcast timelines. Its evidence-only candidate rule
recovered 0 of 136 unmatched first ShuttleSet strokes with a later matched stroke and produced 14
false positives against that target. The measured prototype is not recommended for production work.
