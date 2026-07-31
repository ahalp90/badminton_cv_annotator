# Serve-prepend lookback

This directory contains the current measurement package for the deferred
serve-prepend feature.

- [`serve_prepend_lookback_20260731-091227.md`](serve_prepend_lookback_20260731-091227.md)
  is the current-code orientation and build note
- [`measure_serve_prepend_lookback.py`](measure_serve_prepend_lookback.py) runs the current
  fixture-chain measurement without changing production outputs
- [`data/serve_prepend_lookback_20260731-040847`](data/serve_prepend_lookback_20260731-040847/)
  contains the gzip CSV/JSON and native NumPy-over-XZ/LZMA-9 evidence pack

The archived design record is related but not active:
[`../../archive/serve_prepend_lookback.md`](../../archive/serve_prepend_lookback.md) holds the
original design context and deferred decisions. This directory holds current measurements and
implementation orientation.

No production code was changed. The exploratory run used the current committed-mask chain plus
the existing replay/cutaway-unmasked sensitivity control. Other processing and downstream filters
remain active. The raw external review remains in scratch; its verified findings are folded into
the orientation.
