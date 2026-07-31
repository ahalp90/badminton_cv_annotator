# Serve-prepend lookback

This directory is the current measurement package for the deferred serve-prepend feature.

- [serve_prepend_lookback_20260731-091227.md](serve_prepend_lookback_20260731-091227.md) is the
  authoritative current-code orientation and build note
- [measure_serve_prepend_lookback.py](measure_serve_prepend_lookback.py) runs the fixture-chain
  measurement without changing production outputs
- [data/serve_prepend_lookback_20260731-040847/](data/serve_prepend_lookback_20260731-040847/)
  contains the gzip CSV/JSON and native NumPy-over-XZ/LZMA-9 evidence pack

The archived design record is context only:
[../../archive/serve_prepend_lookback.md](../../archive/serve_prepend_lookback.md). It is not a
current specification or source for current figures.

The exploratory run used the current committed-mask chain and a sensitivity control that passes a
per-frame `raw_exclusion_mask = False` vector to disable replay/cutaway masking. Other
processing and downstream filters remain active. No production code or default behaviour changed.
