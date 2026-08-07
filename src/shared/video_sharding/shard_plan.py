"""Partition a video's frame index space into contiguous half-open shards."""

from __future__ import annotations

# _raw_ndet.npy stores per-frame detection counts as int8 (see raw_extract),
# so any n_max above 127 would silently wrap. Same guard as raw_extract's CLI.
NDET_INT8_CAP = 127


def plan_frame_shards(n_frames: int, n_shards: int) -> list[tuple[int, int]]:
    """Split ``[0, n_frames)`` into ``n_shards`` contiguous half-open ranges.

    Sizes differ by at most one frame (the first ``n_frames % n_shards`` shards
    get the extra frame). The ranges cover every frame exactly once, in order.

    :raises ValueError: on a non-positive frame count or shard count, or more
        shards than frames (an empty shard would stitch as a zero-frame array
        and complicate every downstream guard for no benefit).
    """
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")
    if n_shards <= 0:
        raise ValueError(f"n_shards must be positive, got {n_shards}")
    if n_shards > n_frames:
        raise ValueError(f"n_shards={n_shards} exceeds n_frames={n_frames}")

    base, extra = divmod(n_frames, n_shards)
    shards: list[tuple[int, int]] = []
    start = 0
    for shard_index in range(n_shards):
        end = start + base + (1 if shard_index < extra else 0)
        shards.append((start, end))
        start = end
    return shards
