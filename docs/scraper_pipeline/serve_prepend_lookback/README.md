# Serve-prepend lookback

This directory is the current measurement package for the deferred serve-prepend feature.

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
  target rallies to ball-round-1 source rows and builds pending issue-32 visibility audit templates
- [data/rally_start_visibility_audit_20260809/](data/rally_start_visibility_audit_20260809/)
  contains the deterministic 136-row target package and 32-row quality/control pilot
- [rally_start_visibility_audit_runbook_20260809.md](rally_start_visibility_audit_runbook_20260809.md)
  keeps the viewer on disposable timeline copies and records visibility decisions separately
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
