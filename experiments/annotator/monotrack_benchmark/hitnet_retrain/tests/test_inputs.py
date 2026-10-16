"""Check the retrain boundary against released source and concrete saved-input cases."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_training_set as builder
import inputs
import recipe
import run_training
from ai_badminton.pose import process_pose_file, read_player_poses


def detections(frames: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kps = np.empty((frames, 4, 17, 2))
    for frame in range(frames):
        kps[frame, 0] = [30 + frame * 0.1, 75]
        kps[frame, 1] = [60 + frame * 0.1, 80]
        kps[frame, 2] = [50, 25]
        kps[frame, 3] = [1000, 2000]
    scores = np.broadcast_to(np.array([0.5, 0.9, 0.8, 1.0])[None, :, None], (frames, 4, 17)).copy()
    return kps, scores, np.full(frames, 4, dtype=np.int32)


def court() -> inputs.Court:
    return inputs.Court(np.array([[0, 100], [100, 100], [0, 0], [100, 0]], dtype=float))


def test_selection_and_fill_match_shipped_file_processor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    kps, scores, ndet = detections(8)
    scores[1, 1] = scores[1, 0]  # The first detection wins ties.
    kps[2, 1, 15] = [0, 0]  # Source falls back to the right ankle.
    kps[3, 1, 15:] = np.nan  # Source returns the missing right ankle.
    ndet[4] = 0
    ndet[5] = 1  # Higher-score padded slots must not participate.
    lines = []
    for frame, count in enumerate(ndet):
        lines.append(f"frame {frame}\n")
        for detection in range(count):
            lines.append(f"pose {detection}\n")
            for point, confidence in zip(kps[frame, detection], scores[frame, detection], strict=True):
                lines.append(f"{point[0]} {point[1]} {confidence}\n")
    raw = tmp_path / "poses.txt"
    raw.write_text("".join(lines))
    prefix = str(tmp_path / "reference")
    process_pose_file(str(raw), prefix, court(), fullPose=True)
    reference = read_player_poses(prefix)
    selected = inputs.select_players(kps, scores, ndet, court())
    np.testing.assert_array_equal(selected[0, 0], kps[0, 1])
    np.testing.assert_array_equal(selected[1, 0], kps[1, 0])
    np.testing.assert_array_equal(selected[2, 0], kps[2, 1])
    np.testing.assert_array_equal(selected[3, 0], kps[3, 0])
    assert np.isnan(selected[4]).all()
    assert np.isnan(selected[5, 1]).all()
    filled = inputs.fill_players(selected)
    for slot in range(2):
        # The released processor writes decimal CSV before reading the poses back.
        np.testing.assert_allclose(filled[:, slot].reshape(len(kps), 34), reference[slot].to_numpy(), rtol=0, atol=1e-12)


def test_features_and_targets_match_notebook() -> None:
    frames = 40
    kps, scores, ndet = detections(frames)
    geometry = court()
    players = inputs.fill_players(inputs.select_players(kps, scores, ndet, geometry))
    shuttle = np.column_stack((np.arange(frames) + 10, np.arange(frames) + 30)).astype(float)
    shuttle[10:15] = 0
    rows = inputs.feature_rows(shuttle, players, geometry)
    labels = np.zeros(frames, dtype=np.int8)
    labels[[6, 21, 34]] = [1, 2, 1]
    rallies = [{"role": "train", "offset": 0, "frames": frames}]
    starts, targets = recipe.role_windows(rows, labels, rallies, "train")
    actual = rows[starts[:, None] + np.arange(12)].reshape(len(starts), -1)
    blocks, target_blocks = [], []
    for offset in range(12):
        end = frames - 12 + offset + 1
        blocks.append(np.hstack((shuttle[offset:end], players[offset:end].reshape(end - offset, 68),
                                 np.tile(np.asarray(geometry.corners).flatten(), (end - offset, 1)))))
        target_blocks.append(labels[offset:end])
    reference = recipe.scale_data(np.hstack(blocks)).astype(np.float32)
    np.testing.assert_array_equal(actual, reference)
    np.testing.assert_array_equal(targets, np.vstack(target_blocks[6:]).max(axis=0))
    np.testing.assert_array_equal(actual[0, 70:78], rows[0, 70:78])


def test_windows_do_not_cross_rallies() -> None:
    rows = np.ones((32, 78))
    labels = np.zeros(32, dtype=np.int8)
    labels[16] = 2
    rallies = [{"role": "train", "offset": 0, "frames": 16},
               {"role": "validation", "offset": 16, "frames": 16}]
    starts, targets = recipe.role_windows(rows, labels, rallies, "train")
    np.testing.assert_array_equal(starts, np.arange(5))
    assert not targets.any()


def test_build_preserves_labels_and_counts_failed_courts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stages, courts = tmp_path / "stages", tmp_path / "courts"
    courts.mkdir()
    video = {"item_id": "fixture", "fps": 30, "role": "train", "frame_count": 90, "resolution": [100, 100],
             "spans": []}
    kps, scores, ndet = detections(90)
    for directory in (stages / "pose/fixture", stages / "shuttle/fixture"):
        directory.mkdir(parents=True)
    for name, values in (("kps", kps), ("kp_scores", scores), ("ndet", ndet)):
        inputs.write_array(stages / "pose/fixture" / f"pose_{name}.npy.xz", values)
    track = np.tile([0.4, 0.6, 1.0], (90, 1))
    inputs.write_array(stages / "shuttle/fixture/shuttle_track.npy.xz", track)
    inputs.write_json(stages / "shuttle/fixture/test_stride8_inpaint_mask.json.gz",
                      {"inpaint_status": "applied", "n_rows": 90, "inpaint_selected": [[15, 19]]})
    # The third court swaps physical halves. GT labels must remain unchanged.
    corners = [[0, 0], [100, 0], [100, 100], [0, 100]]
    for span_id in range(3):
        start = 30 * span_id
        video["spans"].append({"span_id": span_id, "rally_id": str(span_id), "start_frame": start,
                               "end_frame": start + 30, "label_frames": [start + 10, start + 20],
                               "label_sides": ["Bot", "Top"]})
        result = {"status": "failed"} if span_id == 1 else {
            "status": "completed", "corners_native_px": corners if span_id == 0 else corners[2:] + corners[:2],
        }
        inputs.write_json(courts / f"fixture_{span_id:04d}.json.gz", result)
    result = builder.build_video((video, stages, courts))
    assert len(result["labels"]) == 2
    assert result["rallies"][1]["excluded"] == "no_court"
    assert result["rallies"][1]["contacts"] == 2
    for labels in result["labels"]:
        assert labels[10] == 1 and labels[20] == 2
    assert np.all(result["zeroed"][0][15:19, :2] == 0)
    np.testing.assert_array_equal(result["kept"][1], result["zeroed"][1])
    # A missing court result is an incomplete dataset, not an authorised skip.
    (courts / "fixture_0001.json.gz").unlink()
    with pytest.raises(FileNotFoundError):
        builder.build_video((video, stages, courts))


def test_augmentation_matches_notebook() -> None:
    notebook = json.loads((inputs.MONOTRACK / "modified-tracknet/train-hitnet.ipynb").read_text())
    namespace: dict = {"np": np, "random": random, "num_consec": 12, "BATCH_SIZE": 64}
    # The cell also imports scipy/sklearn; extract only its numerical functions.
    import ast
    import math

    namespace.update(math=math, eps=1e-6)
    source = "\n".join("".join(notebook["cells"][index]["source"]) for index in (2, 11))
    tree = ast.parse(source)
    helpers = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) or (
        isinstance(node, ast.Assign) and isinstance(node.value, ast.Lambda))], type_ignores=[])
    exec(compile(helpers, "notebook-augmentation", "exec"), namespace)  # noqa: S102 — execute vendored reference
    values = np.random.default_rng(0).uniform(1, 2, (9, 936))
    values[:, ::31] = 0
    for operation in recipe.choices:
        for seed in (0, 17):
            random.seed(seed)
            np.random.seed(seed)
            expected = namespace[operation.__name__](values)
            random.seed(seed)
            np.random.seed(seed)
            actual = operation(values)
            np.testing.assert_array_equal(actual, expected)


def test_winner_export_requires_complete_grid(tmp_path: Path) -> None:
    output, exported = tmp_path / "output", tmp_path / "weights"
    (output / "features").mkdir(parents=True)
    inputs.write_json(output / "features/summary.json.gz", {"by_split": {"train": {"no_court_rallies": 50}}})
    (output / "config.toml").write_text("[inputs]\n")
    for arm, dropout, l2_reg in run_training.GRID:
        directory = output / "trials" / run_training.trial_name(arm, dropout, l2_reg)
        directory.mkdir(parents=True)
        loss = 0.2 if dropout == 0 and l2_reg == 1e-6 else 0.5
        inputs.write_json(directory / "summary.json.gz",
                          {"arm": arm, "dropout": dropout, "l2": l2_reg, "smoke": False, "best_val_loss": loss})
        (directory / "hitnet.h5").write_bytes(directory.name.encode())
        (directory / "history.csv.gz").write_bytes(b"history")
    winners = run_training.export_winners(output, exported)
    assert winners["kept"]["l2"] == 1e-6 and winners["zeroed"]["l2"] == 1e-6
    assert (exported / "hitnet_kept.h5").read_bytes() == b"kept_d0_l21e-06"
    (output / "trials" / run_training.trial_name(*run_training.GRID[-1]) / "summary.json.gz").unlink()
    with pytest.raises(FileNotFoundError):
        run_training.export_winners(output, tmp_path / "incomplete")
    assert not (tmp_path / "incomplete").exists()
