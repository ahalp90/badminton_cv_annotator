"""Pin the shuttle CSV->NPY converter to the old collation maths.

C19 moves the dedup + Visibility-drop + resolution-normalisation off the
collation hot path and onto the one-time converter in
``pipeline.shuttle_extractor``. Collation now reads the saved npy instead of
reparsing the CSV. This test pins that the npy the converter writes carries
exactly what the old ``get_shuttle_result`` used to compute from the CSV, so
the move is byte-for-byte behaviour-preserving.

The oracle inlines the OLD get_shuttle_result maths (pre-C19): read CSV,
drop_duplicates('Frame'), set_index, drop Visibility, to_numpy().astype(float),
divide x by width and y by height. We assert the real converter output matches
it, and (post-rewire) that the real get_shuttle_result reading the npy matches
it too.
"""
import numpy as np
import pandas as pd

from pipeline.shuttle_extractor import shuttle_csvs_to_npy
from preparing_data.prepare_train_on_shuttleset import get_shuttle_result


# Real TrackNetV3 CSV column order: Frame, Visibility, X, Y (raw pixel coords).
# Rows exercise the three cases the converter must handle: ordinary frames,
# a (0,0) visibility-0 miss frame, and a duplicate Frame (keep-first).
_CSV_ROWS = {
    "10_1_1_1": [
        # Frame, Visibility, X, Y
        (0, 1, 100, 200),
        (1, 0, 0, 0),        # miss frame: visibility 0, (0,0) sentinel
        (2, 1, 150, 250),
        (2, 1, 999, 999),    # duplicate Frame=2; keep-first keeps (150, 250)
        (3, 1, 300, 400),
    ],
    "20_2_3_4": [
        (0, 1, 50, 60),
        (1, 1, 70, 80),
    ],
}

# id -> (width, height); vid id is the stem's leading integer.
_RESOLUTIONS = {10: (1280, 720), 20: (1920, 1080)}


def _old_get_shuttle_result(csv_path, v_width, v_height):
    """The pre-C19 get_shuttle_result maths, inlined as the oracle."""
    df = pd.read_csv(csv_path).drop_duplicates("Frame")
    df = df.set_index("Frame").drop(columns="Visibility")
    shuttle_camera = df.to_numpy().astype(float)  # (t, 2): [X, Y]
    x_norm = shuttle_camera[:, 0] / v_width
    y_norm = shuttle_camera[:, 1] / v_height
    return np.stack((x_norm, y_norm), axis=-1)


def _build_fixtures(tmp_path):
    """Write fixture clip .mp4 stubs, {stem}_ball.csv files, and a resolution CSV."""
    clips_dir = tmp_path / "clips"
    csv_dir = tmp_path / "shuttle_csv"
    npy_dir = tmp_path / "shuttle_npy"
    clips_dir.mkdir()
    csv_dir.mkdir()

    for stem, rows in _CSV_ROWS.items():
        # Empty .mp4 stub so clips_dir.rglob('*.mp4') discovers the clip.
        clip_path = clips_dir / "train" / "SomeClass" / f"{stem}.mp4"
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        clip_path.write_bytes(b"")

        df = pd.DataFrame(rows, columns=["Frame", "Visibility", "X", "Y"])
        df.to_csv(csv_dir / f"{stem}_ball.csv", index=False)

    res_rows = [(vid, w, h) for vid, (w, h) in _RESOLUTIONS.items()]
    res_df = pd.DataFrame(res_rows, columns=["id", "width", "height"])
    res_csv = tmp_path / "resolution.csv"
    res_df.to_csv(res_csv, index=False)

    return clips_dir, csv_dir, npy_dir, res_csv


def test_shuttle_npy_matches_old_collation_maths(tmp_path):
    clips_dir, csv_dir, npy_dir, res_csv = _build_fixtures(tmp_path)

    shuttle_csvs_to_npy(
        clips_dir=clips_dir,
        csv_dir=csv_dir,
        npy_output_dir=npy_dir,
        resolution_csv_path=res_csv,
    )

    for vid_id, (v_width, v_height) in _RESOLUTIONS.items():
        stem = next(s for s in _CSV_ROWS if int(s.split("_")[0]) == vid_id)
        oracle = _old_get_shuttle_result(csv_dir / f"{stem}_ball.csv", v_width, v_height)
        saved = np.load(npy_dir / f"{stem}.npy")

        # Converter xy == old collation maths.
        assert np.array_equal(oracle, saved[:, :2]), stem
        # Visibility (deduped, keep-first) passes through as column 2.
        deduped_vis = (
            pd.read_csv(csv_dir / f"{stem}_ball.csv")
            .drop_duplicates("Frame")["Visibility"]
            .to_numpy()
            .astype(float)
        )
        assert np.array_equal(deduped_vis, saved[:, 2]), stem

        # The rewired get_shuttle_result (reads the npy, slices xy) == oracle.
        assert np.array_equal(get_shuttle_result(npy_dir / f"{stem}.npy"), oracle), stem
