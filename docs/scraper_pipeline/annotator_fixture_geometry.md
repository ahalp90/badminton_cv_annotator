# Annotator fixture geometry derivation

Every calibration fixture's camera-space court bounds, and the net band's position,
come from two tracked ShuttleSet CSVs, derived live at import time in
`src/annotator/calibration/fixtures.py`. The one exception is the net band's width: a
fixed policy of 0.5 m each side of centre (`_NET_BAND_HALF_WIDTH_M = 0.5`,
`fixtures.py:133`), a one-metre total band, not a CSV value. `_load_calibration_geometry()`
runs once when the module loads (`fixtures.py:269`) and the three fixtures (`SSET_01`,
`SSET_15`, `SSET_21`) read the result back out by video id. This document is the current
derivation, distilled from the fuller investigation record at
`local_scratch/autograder_architecture/now_tracked/calibration_geometry/s6_b7a_mapping_report.txt`
(kept locally; not tracked in Git). That record used an older fixture name, `pilot`, for
video id 1 before the rename to `sset_01`; every number below already uses the
current names.

## Source CSVs

- `training/data/shuttleset/annotations/set/homography.csv` — one row per ShuttleSet
  video id. Each row carries the four camera-space court corners (`upleft_x/y`,
  `upright_x/y`, `downleft_x/y`, `downright_x/y`, at 1280x720) and a serialised 3x3
  `homography_matrix` string.
- `training/data/shuttleset/annotations/my_raw_video_resolution.csv` — `width`/`height`
  per video id. This is the resolution `fixtures.py` stores on each `Fixture`.

Both are read and validated by `_read_source_frame` (`fixtures.py:141`): the `id`
column must exist, parse to finite integers, and contain no duplicate id. `_source_row`
(`fixtures.py:165`) then requires exactly one matching row per requested video id.
Any violation raises a `ValueError` naming the offending video id and file.

## Resolution scaling

`homography.csv` coordinates are recorded at 1280x720; fixtures store geometry at
1920x1080. `_HOMOGRAPHY_TO_FIXTURE_MULTIPLIER = 1.5` (`fixtures.py:131`,
`1920/1280 = 1080/720 = 1.5`) scales every camera-space coordinate used below before
it lands in a fixture.

## Court box bounds (image-space scaling, no homography projection)

`get_corner_camera` (`shared/court.py:56`) reads the four raw corner columns into a
`(2, 4)` array. `_derive_calibration_geometry` (`fixtures.py:184`) multiplies that
array by `1.5`, then takes the per-axis minimum and maximum, rounded to one decimal
(`_rounded_bounds`, `fixtures.py:238`). This is a direct linear scale of the CSV
corner values — it does not go through the homography matrix. The result is
`court_geo[0]` (x bounds) and `court_geo[1]` (y bounds).

## Net band (uses the full homography matrix)

1. `get_H` (`shared/court.py:49`) parses the row's `homography_matrix` string into a
   3x3 matrix; `fixtures.py` requires it finite or raises.
2. The four camera corners are projected through that matrix
   (`project` + `convert_homogeneous`, `shared/court.py:78`/`62`) into true
   court-plane coordinates. Their min/max give a court-space bounding box; its centre
   is `(x_centre, y_centre)`.
3. `REF_COURT_M[0] = 13.4` (`shared/court.py:39`) is the full court length in metres.
   `court_units_per_metre = (court_y_max - court_y_min) / 13.4` converts real metres
   into the homography's court-space units.
4. The net band half-width is a fixed `0.5` m (`_NET_BAND_HALF_WIDTH_M`,
   `fixtures.py:133`), converted to court units:
   `net_half_band = court_units_per_metre * 0.5`.
5. Two court-space points, `(x_centre, y_centre - net_half_band)` and
   `(x_centre, y_centre + net_half_band)`, are projected back into camera space
   through the inverse homography, scaled by `1.5`, and rounded to one decimal. This
   is `net_band`, also stored as `court_geo[2]`.

## Validation

- `_finite_float` (`fixtures.py:173`) requires every corner and resolution value to
  parse and be finite, or raises naming the field and video id.
- The homography matrix must be finite and invertible; a singular matrix raises
  (`fixtures.py:209`).
- Every projected corner and net-band value must be finite, or the load raises
  (`fixtures.py:210`, `fixtures.py:224`).
- Resolution `width`/`height` must be positive (`fixtures.py:230`).

## The three fixture results

Reproduces the values asserted in
`tests/test_annotator_fixtures.py::test_calibration_geometry_matches_tracked_sources`.

### sset_01 (video id 1)

- Homography row: `downleft_x=307.2`, `downright_x=973.0`, `upright_y=307.4`,
  `downleft_y=671.2`.
- x bounds: `round(307.2 * 1.5, 1), round(973.0 * 1.5, 1)` = `(460.8, 1459.5)`.
- y bounds: `round(307.4 * 1.5, 1), round(671.2 * 1.5, 1)` = `(461.1, 1006.8)`.
- Court-space centre: approximately `(175.0, 480.0)`; half-band
  `(810.0 - 150.0) * 0.5 / 13.4 = 24.626865...`.
- Net band, projected back and scaled: `(664.6, 703.7)`.
- Resolution: `(1920.0, 1080.0)`.

### sset_15 (video id 15)

- Homography row: `downleft_x=293.0`, `downright_x=981.4`, `upright_y=252.0`,
  `downleft_y=662.8`.
- x bounds: `(439.5, 1472.1)`.
- y bounds: `(378.0, 994.2)`.
- Court-space centre: approximately `(175.0, 480.0)`; half-band
  `24.626866...`, giving court-space points `(175.0, 455.373134...)` and
  `(175.0, 504.626866...)`.
- Net band, projected back and scaled: `(583.9, 626.6)`.
- Resolution: `(1920.0, 1080.0)`.

### sset_21 (video id 21)

- Homography row: `downleft_x=289.4`, `downright_x=986.8`, `upright_y=302.2`,
  `downleft_y=659.0`.
- x bounds: `(434.1, 1480.2)`.
- y bounds: `(453.3, 988.5)`.
- Court-space centre: approximately `(177.5, 480.0)`; half-band `24.626866...`.
- Net band, projected back and scaled: `(644.6, 682.5)`.
- Resolution: `(1920.0, 1080.0)`.

## No unsourced literal remains

An earlier, pre-cleanup representation carried a fourth court-geometry value, a
player-height band `(84.0, 336.0)`, identical across all three fixtures with no
tracked derivation (`s6_b7a_mapping_report.txt` records it as `UNSOURCED` after
checking `pilot_geometry.py`, `fixtures.py`, the pose sidecars, and both CSVs above).
That value is not part of the current `CalibrationGeometry`: `Fixture.court_geo` is a
3-tuple (`x_bounds`, `y_bounds`, `net_band`), and every element is now either read from
one of the two tracked CSVs or the one documented fixed policy above (the net band's
one-metre width). Nothing in the current geometry pipeline is an unexplained literal.
