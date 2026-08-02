# Portions of this file are derived from BST (Badminton Stroke-type Transformer)
# by Jing-Yuan Chang, Copyright (c) 2025 Jing-Yuan Chang, used under the MIT
# Licence. See src/bst_x/THIRD_PARTY_NOTICES.md. This project is otherwise
# licensed LGPL-3.0-or-later.

"""Prepare ShuttleSet training data: pose estimation and collation.

Bridges the gap between the pipeline's clip output and BST's expected input format.
Two steps, each independently skippable:
  Step 1: 2D player pose estimation via rtmlib + court projection
  Step 2: Collate per-clip .npy files into batch-ready arrays

Run from the repo root with both package roots on PYTHONPATH::

    PYTHONPATH=src:src/bst_x \\
        python -m preparing_data.prepare_train_on_shuttleset --help
"""

# Deferred annotations so the RtmlibPoseExtractor type hints don't force the rtmlib
# import at module load. That keeps collate_npy and the pose-style joint helpers
# -- none of which touch the extractor -- importable without the extraction deps
# (e.g. venv-bst-x collation, the CPU goldens). The rtmlib adapter is imported
# lazily in the function that instantiates it, and under TYPE_CHECKING for the
# static hints. Keep this: reverting re-couples the module.
from __future__ import annotations

import argparse
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from beartype import beartype
from jaxtyping import Float, jaxtyped
from typing import TYPE_CHECKING

from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor

from preparing_data.extract_failures import (
    FAILURE_ABORT_FRACTION,
    failed_clips_log_path,
    log_failed_clip,
)
from preparing_data.shuttleset_dataset import (
    get_bone_pairs,
    make_seq_len_same,
    create_bones,
    interpolate_joints,
)
from pipeline.config import (
    CLIP_WINDOW,
    CLIPS_OUTPUT_DIR,
    COCO_N_JOINTS,
    SET_INFO_DIR,
    RESOLUTION_CSV_PATH,
    SHUTTLE_OUTPUT_DIR,
    derive_npy_collated_dir_basename,
)
from classifier_shared.taxonomy import (
    BST_X_TAXONOMIES,
    Taxonomy,
    derive_class_index,
    taxonomy_lookup,
)
from pipeline.data_access import env_path, env_path_or_none, load_repo_dotenv
# Court helpers live in shared.court; re-export check_pos_in_court so the
# heuristics (current.py, sticky_anchor.py) keep their existing import path.
from shared.court import build_all_court_info, check_pos_in_court  # noqa: F401
# Shared doubles-guard head count. No import cycle: base.py pulls in no pipeline
# code, and the heuristics modules import this module only lazily inside functions.
from preparing_data.heuristics.base import (
    DOUBLES_COUNT_MARGIN,
    SITTING_THRESHOLD,
    count_standing_in_court,
    is_sitting,
)

if TYPE_CHECKING:  # type-only: keeps rtmlib out of the runtime import (see module-top note)
    from preparing_data.rtmlib_pose import RtmlibPoseExtractor


@jaxtyped(typechecker=beartype)
def normalize_joints(
    arr: Float[np.ndarray, 'players joints 2'],
    bbox: Float[np.ndarray, 'players 4'],
    v_height=None,
    center_align=False,
) -> Float[np.ndarray, 'players joints 2']:
    """Normalise per-player joints; shapes carried by the annotations (players=2 in
    detect/current, players=1 for sticky_anchor's per-slot calls). ``Float`` not ``Float32``:
    callers feed float32 pose keypoints with float64 bboxes interchangeably.

    Signature defaults are BST-upstream; ``main()`` overrides
    ``center_align=True`` (what the committed extracts used).
    """
    # If v_height == None and center_align == False,
    # this normalization method is same as that used in TemPose.
    if v_height is not None:
        dist = v_height / 4
    else:  # bbox diagonal dist
        dist = np.linalg.norm(bbox[:, 2:] - bbox[:, :2], axis=-1, keepdims=True)

    arr_x = arr[:, :, 0]
    arr_y = arr[:, :, 1]
    # No missing-joint guard on purpose: failed frames/slots are zeroed upstream
    # and never reach this function, and RTMPose regression coords are continuous
    # floats (exact 0.0 doesn't occur). If a future pose backend can emit zero or
    # sentinel coords for missing joints, reintroduce a zero-preserving mask here
    # (and exempt sentinels from center_align) so "missing" can't be read as a
    # real on-court position.
    x_normalized = (arr_x - bbox[:, None, 0]) / dist
    y_normalized = (arr_y - bbox[:, None, 1]) / dist

    if center_align:
        center = (bbox[:, :2] + bbox[:, 2:]) / 2
        c_normalized = (center - bbox[:, :2]) / dist
        x_normalized -= c_normalized[:, None, 0]
        y_normalized -= c_normalized[:, None, 1]

    return np.stack((x_normalized, y_normalized), axis=-1)


@jaxtyped(typechecker=beartype)
def _order_two_on_court(
    keypoints_2d: np.ndarray,
    vid: int,
    all_court_info: dict,
    res_df: pd.DataFrame,
) -> tuple[tuple[np.ndarray, np.ndarray] | None, int]:
    """Decide whether a frame has exactly two on-court players, ordered Top-before-Bottom.

    The ``< 2`` short-circuit precedes ``check_pos_in_court`` because the latter slices
    ``keypoints[:, -2:, :]``, which raises on an empty detection. The flip is strict
    ``>``: on a y-tie the original ascending-index order is kept. The flip relies on
    the ``!= 2`` guard upstream so ``np.flip`` on a 2-element array is a swap.

    :param keypoints_2d: (m, J, 2). The 2D keypoints; the court projection needs 2D
        pixel coords.
    :param vid: clip's source video id, used to look up homography + resolution.
    :param all_court_info: dict from get_court_info.
    :param res_df: resolution DataFrame indexed by video id.
    :return: ``(result, n_counted)``. ``result`` is ``(in_court_pid, pos_normalized)``
        on success (exactly 2 on court, ordered Top-before-Bottom), or ``None`` on
        either failure path. ``n_counted`` is the doubles-guard head count: standing
        detections within ``DOUBLES_COUNT_MARGIN`` of the court, where ``> 2`` is
        doubles evidence. That is a separate signal from the pick gate, which still
        keys on ``check_pos_in_court``'s eps-0.01 in-court count being exactly two.
        ``pos_normalized`` is the full ``(m, 2)`` array, not the 2-row slice -- the
        caller does its own ``pos_normalized[in_court_pid]`` (helper returns the full
        array so the caller's existing index expression stays correct).
    """
    if len(keypoints_2d) < 2:
        # Court check never ran (check_pos_in_court would slice an empty detection),
        # so report the detection count as the head count: it upper-bounds the real
        # count, and < 2 can never be a doubles over-count, so the guard stays sound
        # without projecting.
        return None, len(keypoints_2d)
    in_court, pos_normalized = check_pos_in_court(
        keypoints_2d, vid, all_court_info, res_df
    )
    # in_court: (m), pos_normalized: (m, xy), xy=2
    in_court_pid = np.nonzero(in_court)[0]
    # Head count and pick gate are deliberately separate signals: the gate keys on
    # the eps-0.01 in-court count being exactly two (pick logic untouched), while
    # n_counted is the wider DOUBLES_COUNT_MARGIN, sitting-exempt count the caller
    # reads doubles evidence (> 2) off (D26).
    sitting = is_sitting(keypoints_2d, SITTING_THRESHOLD)
    n_counted = count_standing_in_court(pos_normalized, sitting, DOUBLES_COUNT_MARGIN)
    if len(in_court_pid) != 2:
        return None, n_counted
    # Make sure Top player before Bottom player (comparing y-dim).
    # Strict > so a y-tie keeps the np.nonzero ascending order.
    if pos_normalized[in_court_pid[0], 1] > pos_normalized[in_court_pid[1], 1]:
        in_court_pid = np.flip(in_court_pid)
    return (in_court_pid, pos_normalized), n_counted


def detect_players_2d(
    extractor: RtmlibPoseExtractor,
    video_path: Path,
    all_court_info: dict,
    res_df: pd.DataFrame,
    J=COCO_N_JOINTS,
    normalized_by_v_height=False,
    center_align=False,
) -> tuple[list[bool], np.ndarray, np.ndarray, list[bool]] | None:
    """Detect the two on-court players' 2D pose and court positions per frame.

    :return: ``(failed_ls, players_positions, players_joints, overcount_ls)`` on
        success, or ``None`` if the clip decoded zero frames (unreadable, truncated,
        or empty mp4) -- the house "helper returns None on failure" signal, which
        the caller logs + skips. ``failed_ls`` is a per-frame bool list (True
        where no valid two-player pair was found; that frame is zero-filled).
        ``players_positions`` is ``(t, m, xy)`` with ``m=xy=2``;
        ``players_joints`` is ``(t, m, J, xy)``. ``overcount_ls`` is a per-frame bool
        list: True where more than two standing people projected within the doubles
        count margin on that frame (doubles evidence). A doubles rally puts four feet
        in court and so fails the exactly-two test; overcount rides beside failed_ls
        to tell that apart from an ordinary miss.
    """
    vid = int(video_path.name.split("_", 1)[0])

    failed_ls = []
    players_positions = []
    players_joints = []
    overcount_ls = []

    for det in extractor.iter_video(video_path):
        # float64 to match the old mmpose np.array(list-of-lists) dtype: rtmlib
        # returns float32. gate_dtype_parity asserts the cast is in effect, and
        # the parity gates (gate_cpu_downstream_byteeq, gate_deployed_parity)
        # check downstream output against the committed float64 baseline at atol.
        keypoints = det.keypoints.astype(np.float64)  # (n_people, J, 2)

        # Failed frames are kept as zeros (not dropped) so the clip stays intact.
        # Shuttle coords for these frames are zeroed at collation (Step 2).
        ordered, n_counted = _order_two_on_court(keypoints, vid, all_court_info, res_df)
        # Recorded on every frame, including the failed ones: a doubles frame fails
        # the exactly-two test but is exactly where the over-count evidence lives.
        overcount_ls.append(n_counted > 2)
        if not ordered:
            failed_ls.append(True)
            players_positions.append(np.zeros((2, 2), dtype=float))
            players_joints.append(np.zeros((2, J, 2), dtype=float))
            continue
        in_court_pid, pos_normalized = ordered

        bboxes = det.bboxes.astype(np.float64)  # (n_people, 4)

        failed_ls.append(False)
        players_positions.append(pos_normalized[in_court_pid])
        players_joints.append(
            normalize_joints(
                arr=keypoints[in_court_pid],
                bbox=bboxes[in_court_pid],
                v_height=res_df.loc[vid, "height"] if normalized_by_v_height else None,
                center_align=center_align,
            )
        )

    if not failed_ls:
        # Zero frames decoded: np.stack([]) would crash. Signal the caller with
        # None (house pattern) so it can log + skip. No npys written keeps
        # resume able to retry. Court info was never touched on this path.
        return None

    players_positions = np.stack(players_positions)
    # players_positions: (t, m, xy)
    players_joints = np.stack(players_joints)
    # players_joints: (t, m, J, xy)

    return failed_ls, players_positions, players_joints, overcount_ls


def get_shuttle_result(npy_path: Path) -> np.ndarray:
    """Load a clip's normalised shuttle trajectory, xy only.

    The npy is the converter's output (``shuttle_csvs_to_npy`` in
    ``pipeline/shuttle_extractor.py``), which already did the keep-first
    Frame dedup and the resolution-normalisation once at extract time. Here we
    just slice off column 2 (Visibility); misses stay as their saved (0, 0)
    sentinel, untouched.

    :param npy_path: Flat shuttle npy for the clip (``{stem}.npy``).
    :return: (t, 2) normalised ``[x, y]``, each in [0, 1].
    """
    return np.load(str(npy_path))[:, :2]


def prepare_dataset_npy_from_raw_video(
    my_clips_folder: Path,
    save_root_dir: Path,
    resolution_df: pd.DataFrame,
    all_court_info: dict,
    joints_normalized_by_v_height=False,
    joints_center_align=False,
    device: str = "cuda",
):
    """Run rtmlib 2D pose estimation on clips and save per-clip .npy files.

    For each clip, detects player keypoints (COCO 17-joint), extracts court
    positions via homography, and normalizes joints. Saves _joints.npy
    ((F, P, J, xy)), _pos.npy ((F, P, xy)), _overcount.npy ((F,) bool),
    _failed.npy ((F,)) per clip.

    The resume marker is `_failed.npy` because it is saved last; its presence
    means all four outputs are complete for the clip. Shuttle data is read
    from the canonical shuttle-npy dir at collation (Step 2); this expensive
    GPU step stays focused solely on pose estimation.

    A clip that decodes zero frames (unreadable, truncated, or empty mp4) writes
    no npys, is logged to `{save_root_dir}/failed_clips.log` (append mode), and
    is skipped. Once failures exceed 0.3 of the clips slated for extraction this
    run (the not-yet-done clips; resume-skips are excluded), the batch raises a
    RuntimeError naming the log.

    :param my_clips_folder: Directory containing clip .mp4 files (searched recursively).
    :param save_root_dir: Output directory for per-clip .npy files.
    :param resolution_df: DataFrame with video resolutions, indexed by video ID.
    :param all_court_info: Dict mapping video ID to court info (homography, borders).
    :param joints_normalized_by_v_height: If True, normalize joints by video height
        instead of bounding box diagonal.
    :param joints_center_align: If True, center-align joints within bounding box.
    :param device: onnxruntime device for the rtmlib adapter ("cuda" or "cpu").
    """
    from preparing_data.rtmlib_pose import RtmlibPoseExtractor  # lazy: keeps the module rtmlib-free at import (see top)
    pose_extractor = RtmlibPoseExtractor(device=device)

    # Flat layout: per-clip files sit alongside each other under save_root_dir.
    # Split + label come from clips_master.csv at collation time (Step 2).
    save_root_dir.mkdir(parents=True, exist_ok=True)

    all_mp4_paths = sorted(my_clips_folder.glob("**/*.mp4"))

    # Pre-filter to the not-yet-done clips (resume marker _failed.npy absent) so
    # the abort denominator is fixed up front and resume-skips can't dilute the
    # 0.3 failure fraction. Iterating this filtered list (rather than skipping
    # inside the loop) also makes the tqdm total the real work count.
    to_extract = [
        video_path
        for video_path in all_mp4_paths
        if not Path(str(save_root_dir / video_path.stem) + "_failed.npy").exists()
    ]
    n_slated = len(to_extract)
    abort_threshold = FAILURE_ABORT_FRACTION * n_slated
    failures = 0

    for video_path in tqdm(to_extract, desc="Yield .npy files", unit="video"):
        save_branch = str(save_root_dir / video_path.stem)

        result = detect_players_2d(
            extractor=pose_extractor,
            video_path=video_path,
            all_court_info=all_court_info,
            res_df=resolution_df,
            normalized_by_v_height=joints_normalized_by_v_height,
            center_align=joints_center_align,
        )
        if result is None:
            failures += 1
            log_failed_clip(save_root_dir, video_path.stem, "decoded 0 frames")
            if failures > abort_threshold:
                raise RuntimeError(
                    f"{failures} clip(s) failed to decode, exceeding "
                    f"{FAILURE_ABORT_FRACTION:.0%} of the {n_slated} slated for "
                    f"extraction. See {failed_clips_log_path(save_root_dir)} "
                    f"for the failed stems."
                )
            continue
        failed_ls, players_positions, joints, overcount_ls = result

        np.save(save_branch + "_pos.npy", players_positions)
        np.save(save_branch + "_joints.npy", joints)
        np.save(save_branch + "_overcount.npy", np.array(overcount_ls, dtype=bool))
        # _failed.npy stays the last write: it is the resume marker, so its presence
        # must guarantee _pos / _joints / _overcount are already on disk for the clip.
        np.save(save_branch + "_failed.npy", np.array(failed_ls, dtype=bool))

        # Free Python-side buffers between clips over ~33k iterations;
        # onnxruntime manages its own device memory. The zero-frame failure
        # branch allocates nothing, so it skips this.
        gc.collect()


VALID_POSE_STYLES: tuple[str, ...] = ("J_only", "JnB_interp", "JnB_bone", "Jn2B")


@jaxtyped(typechecker=beartype)
def pad_and_derive_pose_styles(
    seq_len: int,
    joints: Float[np.ndarray, 'time players joints 2'],
    pos: Float[np.ndarray, 'time players 2'],
    shuttle: Float[np.ndarray, 'time 2'],
    bone_pairs: list[tuple[int, int]],
    pose_styles: frozenset[str] = frozenset({"JnB_bone"}),
):
    """Pad to uniform sequence length and compute requested pose augmentations.

    Only the pose styles in ``pose_styles`` are computed and returned; the
    derived arrays (``create_bones``, ``interpolate_joints``) are skipped if
    nothing downstream needs them.

    The three arrays share a 'time' axis (frame count): make_seq_len_same resamples them
    with one index. ``Float`` not ``Float32`` because they arrive as whatever the
    per-clip npys hold and get cast to float32 in the body below.

    :param seq_len: Target sequence length. Shorter clips are zero-padded; longer
        clips are resampled (linspace index sampling) to fit.
    :param bone_pairs: List of (start_joint, end_joint) index pairs for bone computation.
    :param pose_styles: Which pose representations to compute. Subset of
        ``VALID_POSE_STYLES``. Defaults to ``{'JnB_bone'}`` (the only style
        BST training has ever used in this tracker).
    :return: Tuple of (pose_dict, pos, shuttle, video_len) where pose_dict maps
        each requested style name to its (time, players, K, 2) array (K = the
        style's keypoint count) and video_len is the number of real
        (non-padded) frames.
    """
    joints = joints.astype(np.float32)
    pos = pos.astype(np.float32)
    shuttle = shuttle.astype(np.float32)

    joints, pos, shuttle, new_video_len = make_seq_len_same(
        seq_len, joints, pos, shuttle
    )

    pose_dict: dict[str, np.ndarray] = {}

    if "J_only" in pose_styles:
        pose_dict["J_only"] = joints

    # bones is needed for JnB_bone and Jn2B; interpolated joints for JnB_interp and Jn2B.
    needs_bones = bool(pose_styles & {"JnB_bone", "Jn2B"})
    needs_interp = bool(pose_styles & {"JnB_interp", "Jn2B"})
    bones = create_bones(joints, bone_pairs) if needs_bones else None
    joints_interpolated = (
        interpolate_joints(joints, bone_pairs) if needs_interp else None
    )

    if "JnB_bone" in pose_styles:
        pose_dict["JnB_bone"] = np.concatenate((joints, bones), axis=-2)
    if "JnB_interp" in pose_styles:
        pose_dict["JnB_interp"] = joints_interpolated
    if "Jn2B" in pose_styles:
        pose_dict["Jn2B"] = np.concatenate((joints_interpolated, bones), axis=-2)

    return pose_dict, pos, shuttle, new_video_len


def _resolve_clips_and_labels(
    clips_csv: Path,
    set_name: str,
    split_column: str,
    taxonomy: Taxonomy,
    root_dir: Path,
    unknown_root_dir: Path | None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Concern 1: CSV filter + per-row label derivation + file-existence drop.

    Reads ``clips_csv``, restricts to ``split_column == set_name``, and runs
    one loop that does the label call, the unknown-root routing, the
    existence check, and the three appends together. The single-loop
    discipline keeps ``data_branches`` / ``labels`` / ``clip_stems_arr``
    row-aligned in the same ``[0, n)`` space.

    Returns the trio so every later stage indexes one shared row order.
    """
    clips_df = pd.read_csv(clips_csv)
    if split_column not in clips_df.columns:
        raise KeyError(
            f"split_column {split_column!r} not in clips_csv columns: "
            f"{list(clips_df.columns)}"
        )
    clips_df = clips_df[clips_df[split_column] == set_name].copy()

    # derive_class_index applies taxonomy.excluded_base_stroke_types first (returns
    # None -> drop the row), then merge_map, then side-prefixing per
    # taxonomy.has_sides. The unknown_root_dir branch pulls per-clip files
    # from the sibling extract for rows that survived (i.e. taxonomies that
    # retain unknown).
    data_branches: list[str] = []
    labels_ls: list[int] = []
    stems_ls: list[str] = []
    missing = 0
    for raw_type, side, stem in zip(
        clips_df["raw_type_en"],
        clips_df["player_side"],
        clips_df["clip_stem"],
    ):
        try:
            idx = derive_class_index(taxonomy, raw_type, side)
        except ValueError as e:
            # Add clip stem context to the descriptive error derive_class_index
            # already raises. Preserves the chain via `from e`.
            raise ValueError(
                f"label derivation failed for clip {stem!r}: {e}"
            ) from e
        if idx is None:
            continue  # filtered out via excluded_base_stroke_types
        chosen_root = (
            unknown_root_dir
            if (raw_type == "unknown" and unknown_root_dir)
            else root_dir
        )
        branch = str(chosen_root / stem)
        # Skip clips whose flat per-clip files are absent. verify_flatten.py
        # should have ruled this out before the originals were deleted, but
        # the check is cheap and prevents a confusing ENOENT mid-collation.
        if not Path(branch + "_pos.npy").exists():
            missing += 1
            continue
        data_branches.append(branch)
        labels_ls.append(idx)
        stems_ls.append(stem)

    if missing:
        unknown_hint = (
            f" (or {unknown_root_dir} for unknown rows)"
            if unknown_root_dir else ""
        )
        print(
            f"  [{set_name}] WARNING: {missing} clips in master CSV had no "
            f"flat per-clip files under {root_dir}{unknown_hint}; skipped."
        )
    labels = np.asarray(labels_ls, dtype=np.int64)
    clip_stems_arr = np.asarray(stems_ls, dtype=object)
    print(
        f"  [{set_name}] {len(data_branches)} clips after filter "
        f"(taxonomy={taxonomy.name}, split_column={split_column})."
    )
    return data_branches, labels, clip_stems_arr


def _load_clip_npys(
    data_branches: list[str], set_name: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Concern 2: parallel-load joints / pos / failed per clip.

    `result()`s are collected in submission order; that's the row-alignment
    contract every downstream stage depends on. Do not switch to
    `as_completed`. `failed_ls` content is unused -- only `len(failed)` feeds
    the temporal-align truncation.
    """
    print(f"Load .npy files for {set_name} set ...")
    with ThreadPoolExecutor() as executor:
        joint_tasks: list[Future] = []
        pos_tasks: list[Future] = []
        failed_tasks: list[Future] = []
        for branch in data_branches:
            joint_tasks.append(executor.submit(np.load, branch + "_joints.npy"))
            pos_tasks.append(executor.submit(np.load, branch + "_pos.npy"))
            failed_tasks.append(executor.submit(np.load, branch + "_failed.npy"))
        joints_ls = [task.result() for task in joint_tasks]
        pos_ls = [task.result() for task in pos_tasks]
        failed_ls = [task.result() for task in failed_tasks]
    print("Finish loading.")
    return joints_ls, pos_ls, failed_ls


def _align_shuttle_and_truncate(
    data_branches: list[str],
    joints_ls: list[np.ndarray],
    pos_ls: list[np.ndarray],
    failed_ls: list[np.ndarray],
    shuttle_npy_dir: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Concern 3: read shuttle npys + truncate joints/pos/shuttle to a common length.

    Shuttle is read here, not in the pose step: the npys are taxonomy/split-
    agnostic, so decoupling lets the ~1.5-3 day GPU pose job run without them
    and collation re-run cheaply per taxonomy. The keep-first Frame dedup and
    the resolution-normalisation already happened once at the converter
    (pipeline/shuttle_extractor.py); this just loads the saved npy.

    Temporal alignment: the pose extractor and TrackNetV3 use different video
    backends that can disagree by 1-2 frames on the tail of the same .mp4. Truncating
    to the shorter length preserves frame alignment (both decoders start at
    frame 0). Truncation propagates to joints AND pos so pose frame k stays
    paired with shuttle frame k.

    Returns the joints / pos / shuttle triple explicitly
    so concern 4 doesn't need to know that joints_ls / pos_ls were also
    mutated in place.
    """
    shuttle_ls = []
    for i, branch in enumerate(data_branches):
        clip_stem = Path(branch).name  # e.g. '35_1_10_17'
        npy_path = shuttle_npy_dir / (clip_stem + ".npy")
        shuttle = get_shuttle_result(npy_path)
        failed = failed_ls[i]

        min_t = min(len(failed), len(shuttle))
        if len(failed) != len(shuttle):
            joints_ls[i] = joints_ls[i][:min_t]
            pos_ls[i] = pos_ls[i][:min_t]
            shuttle = shuttle[:min_t]

        # Pose-fail frames no longer wipe shuttle. Pose tells you where a
        # player is in the court; shuttle tells you where the bird is. If
        # one player collapses, the bird's coord still has meaning.
        # Full rationale: docs/architecture_notes/frame_zeroing.md.

        shuttle_ls.append(shuttle)

    return joints_ls, pos_ls, shuttle_ls


def _pad_derive_stack_save(
    joints_ls: list[np.ndarray],
    pos_ls: list[np.ndarray],
    shuttle_ls: list[np.ndarray],
    pose_styles: frozenset[str],
    seq_len: int,
    save_dir: Path,
    set_name: str,
    labels: np.ndarray,
    clip_stems_arr: np.ndarray,
) -> None:
    """Concern 4: per-clip pad/augment via ProcessPool, then stack + save.

    `bad_styles` runs BEFORE the ProcessPool so a typo fails fast without
    spinning workers. ProcessPool `.result()`s are collected in
    submission order to preserve row alignment. The non-pose
    stacks (pos / shuttle / videos_len) complete before any save; the
    per-style pose stack is computed inline as `np.save`'s argument, so a
    stack failure on style k would leave styles 0..k-1 written -- the live
    behaviour, preserved here.
    """
    bad_styles = set(pose_styles) - set(VALID_POSE_STYLES)
    if bad_styles:
        raise ValueError(
            f"Unknown pose_styles {sorted(bad_styles)!r}; "
            f"valid choices: {VALID_POSE_STYLES}"
        )

    bone_pairs = get_bone_pairs(skeleton_format="coco")

    print(f"Pad, Create bones and Interpolate (pose_styles={sorted(pose_styles)}) ...")
    with ProcessPoolExecutor() as executor:
        tasks: list[Future] = []
        for joints, pos, shuttle in zip(joints_ls, pos_ls, shuttle_ls):
            tasks.append(
                executor.submit(
                    pad_and_derive_pose_styles,
                    seq_len=seq_len,
                    joints=joints,
                    pos=pos,
                    shuttle=shuttle,
                    bone_pairs=bone_pairs,
                    pose_styles=pose_styles,
                )
            )

        pose_arrs: dict[str, list[np.ndarray]] = {k: [] for k in pose_styles}
        padded_pos: list[np.ndarray] = []
        padded_shuttle: list[np.ndarray] = []
        videos_len: list[int] = []
        for task in tasks:
            pose_dict, p, s, v_len = task.result()
            for k, arr in pose_dict.items():
                pose_arrs[k].append(arr)
            padded_pos.append(p)
            padded_shuttle.append(s)
            videos_len.append(v_len)

    pos_stacked = np.stack(padded_pos)
    shuttle_stacked = np.stack(padded_shuttle)
    videos_len_stacked = np.stack(videos_len)
    print("Finish padding and augmenting.")

    if not save_dir.is_dir():
        save_dir.mkdir()
    set_dir = save_dir / set_name
    if not set_dir.is_dir():
        set_dir.mkdir()

    for k, arrs in pose_arrs.items():
        np.save(str(set_dir / f"{k}.npy"), np.stack(arrs))
    np.save(str(set_dir / "pos.npy"), pos_stacked)
    np.save(str(set_dir / "shuttle.npy"), shuttle_stacked)
    np.save(str(set_dir / "videos_len.npy"), videos_len_stacked)
    np.save(str(set_dir / "labels.npy"), labels)
    # Row-aligned clip stems sidecar so the post-hoc FE-JSON converter can
    # join row index -> stem without re-deriving the CSV filter.
    np.save(str(set_dir / "clip_stems.npy"), clip_stems_arr, allow_pickle=True)
    print("Collation is complete.")


def collate_npy(
    root_dir: Path,
    set_name: str,
    seq_len: int,
    save_dir: Path,
    clips_csv: Path,
    split_column: str,
    taxonomy: Taxonomy,
    shuttle_npy_dir: Path,
    pose_styles: frozenset[str] = frozenset({"JnB_bone"}),
    unknown_root_dir: Path | None = None,
):
    """Collate per-clip .npy files into stacked batch arrays for one split.

    Reads split assignment and label from the master clips CSV, resolves
    per-clip files at FLAT path ``{root_dir}/{clip_stem}_*.npy``, reads
    shuttle trajectories from the canonical shuttle-npy dir, aligns temporal
    dimensions, truncates pose and shuttle to a common length, pads to uniform seq_len,
    computes bone vectors and interpolations, then saves the stacked arrays
    into ``save_dir/set_name/``. A row-aligned ``clip_stems.npy`` sidecar
    is saved alongside ``labels.npy`` so downstream consumers can join row
    index -> stem directly without re-deriving the CSV filter.

    Four staged helpers: ``_resolve_clips_and_labels`` builds the row order,
    ``_load_clip_npys`` parallel-loads joints/pos/failed, ``_align_shuttle_and_truncate``
    reads shuttle npys and truncates the triple to a common length, and
    ``_pad_derive_stack_save`` runs the ProcessPool and writes the stacked
    arrays. The argument guards stay here so a bad call fails before any work.

    :param root_dir: FLAT per-clip dir containing
        ``{clip_stem}_{joints,pos,failed}.npy`` for every clip.
    :param set_name: One of 'train', 'val', 'test'.
    :param seq_len: Target sequence length (frames). Clips are padded/strided to this length.
    :param save_dir: Output directory. A set_name/ subdirectory is created inside.
    :param clips_csv: Master clips CSV (one row per clip) providing split
        assignment (``split_column``), ``raw_type_en``, ``player_side``,
        ``clip_stem``.
    :param split_column: Column in ``clips_csv`` to use for split assignment,
        e.g. ``'split_bst_baseline'`` or ``'split_v2'``.
    :param taxonomy: Taxonomy. ``derive_class_index`` drives per-row class index +
        the unknown-filter rule (via ``excluded_base_stroke_types``); no
        separate drop_unknown flag any more.
    :param shuttle_npy_dir: Flat dir of converter-output shuttle npys
        ({clip}.npy: normalised (t, 3) [x, y, visibility]). Required.
    :param unknown_root_dir: Optional FLAT per-clip dir for rows whose
        ``raw_type_en == 'unknown'``. When set, unknown rows resolve their
        per-clip files from this dir instead of ``root_dir``. Used to point
        the bst_25 / une_v1_15 collations at the sibling
        ``ShuttleSet_keypoints_clean_sticky_anchor_unknown/`` extract. Must
        be None when the taxonomy has ``'unknown'`` in
        ``excluded_base_stroke_types`` (those rows get dropped anyway).
    """
    if set_name not in ("train", "val", "test"):
        raise ValueError(f"Invalid set_name {set_name!r}; expected 'train', 'val', or 'test'.")
    excluded = taxonomy.excluded_base_stroke_types or frozenset()
    if unknown_root_dir and 'unknown' in excluded:
        raise ValueError(
            f"unknown_root_dir set but taxonomy {taxonomy.name!r} excludes "
            f"unknown rows (excluded_base_stroke_types contains 'unknown'). "
            f"Either drop unknown_root_dir or pick a taxonomy that retains unknown."
        )
    if taxonomy.has_unknown and not unknown_root_dir:
        raise ValueError(
            f"taxonomy {taxonomy.name!r} retains unknown in its class list, "
            f"but unknown_root_dir is None. The 1,278 ShuttleSet unknown-class clips "
            f"don't have "
            f"per-clip files under the canonical extract; they need to come "
            f"from the sibling _unknown extract. Pass unknown_root_dir=<that "
            f"sibling dir>, OR pick a taxonomy whose excluded_base_stroke_types "
            f"includes 'unknown' (e.g. {taxonomy.name.replace('_25', '_24').replace('_15', '_14')!r})."
        )

    data_branches, labels, clip_stems_arr = _resolve_clips_and_labels(
        clips_csv=clips_csv,
        set_name=set_name,
        split_column=split_column,
        taxonomy=taxonomy,
        root_dir=root_dir,
        unknown_root_dir=unknown_root_dir,
    )
    joints_ls, pos_ls, failed_ls = _load_clip_npys(data_branches, set_name)
    joints_ls, pos_ls, shuttle_ls = _align_shuttle_and_truncate(
        data_branches=data_branches,
        joints_ls=joints_ls,
        pos_ls=pos_ls,
        failed_ls=failed_ls,
        shuttle_npy_dir=shuttle_npy_dir,
    )
    _pad_derive_stack_save(
        joints_ls=joints_ls,
        pos_ls=pos_ls,
        shuttle_ls=shuttle_ls,
        pose_styles=pose_styles,
        seq_len=seq_len,
        save_dir=save_dir,
        set_name=set_name,
        labels=labels,
        clip_stems_arr=clip_stems_arr,
    )


def main():
    """Parse CLI arguments and run the requested pipeline steps.

    Usage (from the repo root, with both package roots on PYTHONPATH):
        PYTHONPATH=src:src/bst_x \\
            python -m preparing_data.prepare_train_on_shuttleset --dry-run
        PYTHONPATH=src:src/bst_x \\
            python -m preparing_data.prepare_train_on_shuttleset --skip-pose
    """
    # Populate os.environ from <repo>/.env so argparse defaults below can
    # read BST_* vars. Same pattern as pipeline.data_access.
    load_repo_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "Prepare ShuttleSet training data in 2 steps:\n"
            "  Step 1: 2D pose estimation (rtmlib)\n"
            "  Step 2: Collate per-clip .npy files into batch arrays\n"
            "\n"
            "Each step can be skipped independently."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Step control
    parser.add_argument(
        "--skip-pose", action="store_true", help="Skip Step 1 (pose estimation)"
    )
    parser.add_argument(
        "--skip-collate",
        action="store_true",
        help="Skip Step 2 (collation into batch arrays)",
    )

    # Data configuration
    parser.add_argument(
        "--seq-len",
        type=int,
        default=100,
        choices=[30, 100],
        help="Target sequence length in frames (default: 100)",
    )
    parser.add_argument(
        "--taxonomy",
        default="une_v1_14",
        choices=list(BST_X_TAXONOMIES),
        help="Stroke type taxonomy (default: une_v1_14). Drives derive_class_index "
             "per-row index + the unknown-filter rule via "
             "excluded_base_stroke_types.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="onnxruntime device for the rtmlib adapter: 'cuda' (default, "
             "needs onnxruntime-gpu) or 'cpu'.",
    )

    # Path overrides (only the ones that genuinely vary)
    # Defaults read from .env (loaded above via load_repo_dotenv) when the
    # corresponding BST_* env var is set; otherwise fall back to the repo-
    # rooted constants from pipeline/config.py. Same pattern as data_access.
    parser.add_argument(
        "--clips-dir",
        type=Path,
        default=env_path('BST_X_CLIPS_DIR', CLIPS_OUTPUT_DIR),
        help=f"Clip .mp4 input directory (default: BST_X_CLIPS_DIR or {CLIPS_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--shuttle-npy-dir",
        type=Path,
        default=env_path('BST_X_SHUTTLE_NPY_DIR', SHUTTLE_OUTPUT_DIR),
        help=f"Directory with converter-output shuttle npys (default: BST_X_SHUTTLE_NPY_DIR or {SHUTTLE_OUTPUT_DIR})",
    )

    # Step 2 (collation) configuration: drives split + label assignment from
    # the master clips CSV instead of the on-disk folder layout. The flat
    # per-clip dir holds {clip_stem}_*.npy files shared across all ablations;
    # the collated dir is per-cell -- the parent dir carries the taxonomy and
    # the basename carries split + generation tag, so cells that share a
    # taxonomy but differ by split don't collide.
    parser.add_argument(
        "--clips-csv",
        type=Path,
        default=env_path(
            'BST_X_CLIPS_CSV',
            Path(__file__).resolve().parents[3] / "notebooks" / "clips_master.csv",
        ),
        help="Master clips CSV with split + label per clip "
             "(default: BST_X_CLIPS_CSV or <repo>/notebooks/clips_master.csv).",
    )
    parser.add_argument(
        "--split-column",
        default="split_bst_baseline",
        choices=["split_bst_baseline", "split_v2"],
        help="Column in clips_csv giving train/val/test assignment "
             "(default: split_bst_baseline).",
    )
    parser.add_argument(
        "--collation-id",
        required=True,
        help="Required collation generation tag. Suffixes the collated output "
             "dir (npy_[seq{N}_]{split}_{collation_id}) so re-collations of "
             "the same taxonomy + split coexist. Common values: "
             "'taxon_pinned_w_preds', 'wipe_drop'. A training-time ablation tag "
             "is separate and lives in the run manifest, not the collation path.",
    )
    parser.add_argument(
        "--clip-npy-dir",
        type=Path,
        default=env_path_or_none('BST_X_RTMPOSE_NPY_DIR'),
        help="FLAT per-clip dir (Step 1 writer + Step 2 reader). Default reads "
             "BST_X_RTMPOSE_NPY_DIR; if unset, falls back to the per-taxonomy "
             f"preparing_root + 'dataset_npy_{CLIP_WINDOW}_flat'.",
    )
    parser.add_argument(
        "--unknown-clip-npy-dir",
        type=Path,
        default=None,
        help="Optional FLAT per-clip dir for rows with raw_type_en=='unknown'. "
             "Routes unknown rows through this dir while everything else comes "
             "from --clip-npy-dir. Used to point bst_25 / une_v1_15 collations "
             "at the sibling _unknown extract. Must NOT be set when the "
             "taxonomy excludes unknown.",
    )
    parser.add_argument(
        "--pose-styles",
        default="JnB_bone",
        help="Comma-separated pose representations to compute and save at "
             "Step 2. Default 'JnB_bone' (the only style BST training has "
             f"used in this tracker). Valid choices: {','.join(VALID_POSE_STYLES)}.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be done without executing",
    )

    args = parser.parse_args()

    # ---- Resolve taxonomy and derive intermediate paths ----
    taxonomy = taxonomy_lookup(args.taxonomy)
    excluded = taxonomy.excluded_base_stroke_types or frozenset()
    if args.unknown_clip_npy_dir and 'unknown' in excluded:
        parser.error(
            f"--unknown-clip-npy-dir is set but taxonomy {taxonomy.name!r} "
            f"excludes unknown rows (excluded_base_stroke_types={sorted(excluded)}). "
            f"Either drop the flag or pick a taxonomy that retains unknown."
        )
    if taxonomy.has_unknown and not args.unknown_clip_npy_dir:
        parser.error(
            f"--taxonomy {taxonomy.name!r} retains unknown in its class list, "
            f"but --unknown-clip-npy-dir is not set. The 1,278 ShuttleSet "
            f"unknown-class clips don't have per-clip files under the canonical "
            f"extract; they need "
            f"to come from the sibling _unknown extract. Pass "
            f"--unknown-clip-npy-dir <that sibling dir>, OR pick a taxonomy "
            f"whose excluded_base_stroke_types includes 'unknown'."
        )
    # ShuttleSet_data_<tax>/ root reads BST_X_COLLATED_DATA_ROOT when set
    # (matches the FE serving contract in frontend_integration_guide.md);
    # otherwise falls back to the in-repo preparing_data/ convention for
    # local dev where /scratch isn't available.
    collated_data_root = env_path_or_none('BST_X_COLLATED_DATA_ROOT')
    if collated_data_root:
        preparing_root = collated_data_root / f"ShuttleSet_data_{taxonomy.name}"
    else:
        preparing_root = (
            Path(__file__).resolve().parent / f"ShuttleSet_data_{taxonomy.name}"
        )
    preparing_root.mkdir(parents=True, exist_ok=True)

    # Parse + validate --pose-styles.
    pose_styles = frozenset(s.strip() for s in args.pose_styles.split(",") if s.strip())
    bad_styles = pose_styles - set(VALID_POSE_STYLES)
    if bad_styles:
        parser.error(
            f"Unknown --pose-styles entries {sorted(bad_styles)!r}; "
            f"valid: {','.join(VALID_POSE_STYLES)}"
        )

    # Collated dir naming via shared helper (mirrored on the bst_x_train.py
    # reader side); see ``pipeline.config.derive_npy_collated_dir_basename``.
    npy_collated_dir = preparing_root / derive_npy_collated_dir_basename(
        seq_len=args.seq_len,
        split_column=args.split_column,
        collation_id=args.collation_id,
    )
    if args.seq_len == 30:
        default_flat_dir = preparing_root / "dataset_npy_flat"
    else:  # 100
        default_flat_dir = (
            preparing_root / f"dataset_npy_{CLIP_WINDOW}_flat"
        )

    # FLAT per-clip dir. Step 1 writes per-clip files here ({clip_stem}_*.npy),
    # Step 2 reads from here. Split + label come from clips_master.csv at
    # collation time -- the layout is taxonomy- and split-independent.
    flat_clip_npy_dir = args.clip_npy_dir or default_flat_dir

    # ---- Dry run ----
    if args.dry_run:
        print("=== DRY RUN (no files will be created) ===\n")
        print(f"  seq_len:          {args.seq_len}")
        print(f"  taxonomy:         {taxonomy.name} ({taxonomy.n_classes} classes)")
        print(f"  clips_dir:        {args.clips_dir}")
        print(f"  shuttle_npy_dir:  {args.shuttle_npy_dir}")
        print(f"  flat_clip_npy:    {flat_clip_npy_dir}  (Step 1 writer + Step 2 reader)")
        print(f"  npy_collated:     {npy_collated_dir}")
        print(f"  clips_csv:        {args.clips_csv}")
        print(f"  split_column:     {args.split_column}")
        print(f"  excluded_raw:     {sorted(excluded)}")
        print(f"  unknown_clip_dir: {args.unknown_clip_npy_dir}")
        print(f"  collation_id:     {args.collation_id}")
        print(f"  pose_styles:      {sorted(pose_styles)}")
        print(f'  homography:       {SET_INFO_DIR / "homography.csv"}')
        print(f"  resolution:       {RESOLUTION_CSV_PATH}")
        print(f'\n  Step 1 (pose):    {"SKIP" if args.skip_pose else "RUN"}')
        print(f'  Step 2 (collate): {"SKIP" if args.skip_collate else "RUN"}')
        print("\n=== End dry run ===")
        return

    # ---- Load resolution data + court info (needed by all steps) ----
    resolution_df = pd.read_csv(str(RESOLUTION_CSV_PATH)).set_index("id")
    all_court_info = build_all_court_info(SET_INFO_DIR, resolution_df)

    # ---- Step 1: Pose estimation ----
    if not args.skip_pose:
        print("\n--- Step 1: Pose estimation ---")
        prepare_dataset_npy_from_raw_video(
            my_clips_folder=args.clips_dir,
            save_root_dir=flat_clip_npy_dir,
            resolution_df=resolution_df,
            all_court_info=all_court_info,
            joints_normalized_by_v_height=False,
            joints_center_align=True,
            device=args.device,
        )
    else:
        print("Step 1: Skipped (--skip-pose)")

    # ---- Step 2: Collation ----
    if not args.skip_collate:
        print("\n--- Step 2: Collate .npy files ---")
        if not args.clips_csv.exists():
            parser.error(f"--clips-csv path does not exist: {args.clips_csv}")
        if not flat_clip_npy_dir.exists():
            parser.error(
                f"flat per-clip dir does not exist: {flat_clip_npy_dir}\n"
                "  Run Step 1 first (drop --skip-pose) or pass --clip-npy-dir."
            )
        for set_name in ["train", "val", "test"]:
            collate_npy(
                root_dir=flat_clip_npy_dir,
                set_name=set_name,
                seq_len=args.seq_len,
                save_dir=npy_collated_dir,
                clips_csv=args.clips_csv,
                split_column=args.split_column,
                pose_styles=pose_styles,
                taxonomy=taxonomy,
                unknown_root_dir=args.unknown_clip_npy_dir,
                shuttle_npy_dir=args.shuttle_npy_dir,
            )
    else:
        print("Step 2: Skipped (--skip-collate)")

    print("\nAll requested steps complete.")


if __name__ == "__main__":
    main()
