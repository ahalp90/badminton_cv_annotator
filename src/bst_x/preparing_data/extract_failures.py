"""Shared failure-handling for the two pose-extraction batch entry points.

Both batch paths (``raw_extract`` and ``prepare_train_on_shuttleset``'s Step 1)
run the same rtmlib adapter over thousands of clips. A clip that decodes zero
frames (unreadable, truncated, or empty mp4) used to crash the whole batch with
an opaque ``np.stack([])`` error. Instead each entry point logs the clip, skips
it, and aborts loudly only if too many clips fail.

The failure log is informational only. A failed clip writes no npys, so the
existing resume logic (``_raw_ndet.npy`` / ``_failed.npy`` presence) retries it
next run; nothing ever reads this log back as a skip marker.

Kept in its own tiny module so both entry points share one filename, one abort
fraction, and one log-writer without either importing the other (they otherwise
pull in disjoint, heavy dependency stacks).
"""
from __future__ import annotations

from pathlib import Path

# One append-mode line per failed clip, written under the batch's save-dir.
# Named here so both entry points write the same file and their abort messages
# can point a human at it.
FAILED_CLIPS_LOG = "failed_clips.log"

# Abort the batch once failures exceed this fraction of the clips SLATED for
# extraction in this invocation. "Slated" excludes clips skipped by resume (they
# never run, so they must not dilute the denominator); each entry point computes
# a fixed slated-count up front. A few unreadable clips are noise; a third of the
# batch failing means something systemic (wrong clips dir, a corrupt download).
FAILURE_ABORT_FRACTION = 0.3


def failed_clips_log_path(save_dir: Path) -> Path:
    """Return the failure-log path for a batch writing under ``save_dir``."""
    return save_dir / FAILED_CLIPS_LOG


def log_failed_clip(save_dir: Path, stem: str, reason: str) -> None:
    """Print a clip failure and append one ``{stem}\\t{reason}`` line to the log.

    Append mode: the log accumulates across the batch (and across reruns). It is
    informational only. A failed clip writes no npys, so resume retries it next
    run and this file is never read back as a skip marker; deleting it changes
    nothing about which clips re-extract.

    :param save_dir: the batch's output dir; the log lands at
        ``save_dir/failed_clips.log``.
    :param stem: clip stem that failed.
    :param reason: short one-line failure reason.
    """
    print(f"  FAILED: {stem} ({reason})")
    with failed_clips_log_path(save_dir).open("a") as fh:
        fh.write(f"{stem}\t{reason}\n")
