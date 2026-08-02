"""Pin + verify + vendor the rtmlib ONNX models (Batch-0 harness).

Fetches the two model archives the adapter uses, via rtmlib's own downloader,
so the cached ``.onnx`` names match what the adapter loads at runtime; asserts
each *extracted* ``.onnx`` SHA256 against a pinned constant, and vendors the
verified ``.onnx`` to a durable pool dir so extraction no longer depends on the
openmmlab URL staying up.

Fails non-zero on any SHA mismatch: an upstream re-release would silently change
the model, and must be re-pinned deliberately, not auto-accepted.

The pinned hashes are of the extracted ``.onnx`` (the artifact onnxruntime
loads), not the ``.zip`` (which rtmlib deletes after extraction). Extraction is
deterministic, so the ``.onnx`` bytes are stable.

Env:
  XDG_CACHE_HOME           rtmlib cache root (default ~/.cache)
  RTMLIB_MODEL_VENDOR_DIR  where to copy the verified .onnx
                           (default: pool .../rtmlib_onnx)

Run:
  XDG_CACHE_HOME=<cache> PYTHONPATH=src:src/bst_x <venv>/bin/python \\
      src/bst_x/validation_scripts/rtmlib_migration/download_and_verify_models.py
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

from rtmlib.tools.file import download_checkpoint

from preparing_data.rtmlib_pose import DET_URL, POSE_URL

# SHA256 of the extracted .onnx: pose computed 2026-07-02, detector 2026-07-04
# (RTMDet-M restoration; hash identical across two independent downloads).
# A mismatch means upstream re-released the model; stop and re-pin deliberately.
PINNED = {
    DET_URL: "4f4d7e07350b1753299111d1ae500fd64447a5b0e38e4bacbefab6573c742d30",
    POSE_URL: "cff059fd58a2c0d5fabaddcd66a96abcfb327563bcb0149ea59c9de4a8990fe2",
}

VENDOR_DIR = Path(os.environ.get(
    "RTMLIB_MODEL_VENDOR_DIR",
    "/srv/mergerfs/main_pool/320_cosc594_data-bourbaki/rtmlib_onnx",
))


def _sha256(path: Path) -> str:
    """Streaming SHA256 (models are up to ~110 MB)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for url, expected in PINNED.items():
        onnx = Path(download_checkpoint(url, progress=False))  # cached .onnx if present
        actual = _sha256(onnx)
        ok = actual == expected
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {onnx.name}")
        print(f"        expected {expected}")
        print(f"        actual   {actual}")
        if ok:
            dst = VENDOR_DIR / onnx.name
            if not dst.exists() or _sha256(dst) != expected:
                shutil.copy2(onnx, dst)
            print(f"        vendored -> {dst}")
    print(f"\n{'PASS' if all_ok else 'FAIL'}: model SHA verification")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
