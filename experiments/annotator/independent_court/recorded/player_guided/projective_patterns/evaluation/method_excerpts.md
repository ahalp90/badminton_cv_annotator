# Exact method excerpts

Read the [evidence guide](README.md) first. These are literal source excerpts,
not a standalone runnable module. Module names and whole-file MD5s identify the
source snapshots. Imports and unrelated functions are omitted. `np` means NumPy,
`cv2` means OpenCV; other names refer to the named project modules. No new result
is produced by publishing these excerpts.

## court_corners.py

Whole-source MD5: `ae6a8494792a5161341e6c8687f337c9`.

### COURT_WIDTH_M

```python
COURT_WIDTH_M = 6.10  # doubles sideline to doubles sideline
```

### COURT_LENGTH_M

```python
COURT_LENGTH_M = 13.40  # baseline to baseline
```

### CORNER_COURT_M

```python
CORNER_COURT_M = np.array(
    [[0.0, 0.0], [COURT_WIDTH_M, 0.0], [COURT_WIDTH_M, COURT_LENGTH_M], [0.0, COURT_LENGTH_M]],
    dtype=np.float32,
)
```

### _SINGLES_INSET

```python
_SINGLES_INSET = 0.46
```

### _LONG_SERVICE_INSET

```python
_LONG_SERVICE_INSET = 0.76
```

### _SHORT_SERVICE_OFFSET

```python
_SHORT_SERVICE_OFFSET = 1.98
```

### _NET_Y

```python
_NET_Y = COURT_LENGTH_M / 2.0
```

### _CENTRE_X

```python
_CENTRE_X = COURT_WIDTH_M / 2.0
```

### _FAR_SHORT_Y

```python
_FAR_SHORT_Y = _NET_Y - _SHORT_SERVICE_OFFSET  # 4.72
```

### _NEAR_SHORT_Y

```python
_NEAR_SHORT_Y = _NET_Y + _SHORT_SERVICE_OFFSET  # 8.68
```

### PAINTED_SEGMENTS_M

```python
PAINTED_SEGMENTS_M: tuple[tuple[np.ndarray, np.ndarray], ...] = tuple(
    (np.array(a, dtype=np.float32), np.array(b, dtype=np.float32))
    for a, b in (
        # x-family: constant x, varying y
        ((0.0, 0.0), (0.0, COURT_LENGTH_M)),  # left doubles sideline
        ((_SINGLES_INSET, 0.0), (_SINGLES_INSET, COURT_LENGTH_M)),  # left singles sideline
        ((_CENTRE_X, 0.0), (_CENTRE_X, _FAR_SHORT_Y)),  # centre line, far half
        ((_CENTRE_X, _NEAR_SHORT_Y), (_CENTRE_X, COURT_LENGTH_M)),  # centre line, near half
        ((COURT_WIDTH_M - _SINGLES_INSET, 0.0), (COURT_WIDTH_M - _SINGLES_INSET, COURT_LENGTH_M)),  # right singles
        ((COURT_WIDTH_M, 0.0), (COURT_WIDTH_M, COURT_LENGTH_M)),  # right doubles sideline
        # y-family: constant y, varying x
        ((0.0, 0.0), (COURT_WIDTH_M, 0.0)),  # far baseline
        ((0.0, _LONG_SERVICE_INSET), (COURT_WIDTH_M, _LONG_SERVICE_INSET)),  # far doubles long service
        ((0.0, _FAR_SHORT_Y), (COURT_WIDTH_M, _FAR_SHORT_Y)),  # far short service
        ((0.0, _NEAR_SHORT_Y), (COURT_WIDTH_M, _NEAR_SHORT_Y)),  # near short service
        ((0.0, COURT_LENGTH_M - _LONG_SERVICE_INSET), (COURT_WIDTH_M, COURT_LENGTH_M - _LONG_SERVICE_INSET)),
        ((0.0, COURT_LENGTH_M), (COURT_WIDTH_M, COURT_LENGTH_M)),  # near baseline
    )
)
```

## vp_pruning.py

Whole-source MD5: `0fc8080c36157da21949994fec4862e8`.

### Settings

```python
@dataclass(frozen=True)
class Settings:
    angle_deg: float = 1.5
    direction_lines: int = 128
    pencils: int = 16
    overlap: float = 0.8
    rectangles: int = 16_384
    candidate_batch: int = 256
    pencil_selection: str = "ranked"
```

### normalisation

```python
def normalisation(size: tuple[int, int]) -> np.ndarray:
    """Map isotropic, centred coordinates back to working pixels."""
    width, height = size
    diagonal = np.hypot(width, height)
    return np.array([[diagonal, 0, width / 2], [0, diagonal, height / 2], [0, 0, 1]])
```

### angular_residuals

```python
def angular_residuals(lines: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Line/ray angles in degrees at each line's foot from the coordinate origin.

    :return: One residual per homogeneous point and observed line. Undefined
        rays receive 90 degrees, including a finite VP at the line's foot.
    """
    normals = lines[:, :2]
    feet = -lines[:, 2, None] * normals / np.sum(normals**2, axis=1)[:, None]
    directions = np.stack((lines[:, 1], -lines[:, 0]), axis=1)
    rays = points[:, None, :2] - points[:, None, 2:] * feet[None]
    dot = np.abs(np.einsum("pli,li->pl", rays, directions))
    cross = np.abs(rays[..., 0] * directions[:, 1] - rays[..., 1] * directions[:, 0])
    angles = np.degrees(np.arctan2(cross, dot))
    return np.where(np.linalg.norm(rays, axis=2) > 1e-12, angles, 90.0)
```

### retain_pencils

```python
def retain_pencils(
    masks: np.ndarray, counts: np.ndarray, supports: np.ndarray, candidate_ids: np.ndarray, settings: Settings,
) -> tuple[list[int], np.ndarray]:
    """Retain competing pencils by raw support or additional observation coverage."""
    if settings.pencil_selection not in ("ranked", "coverage"):
        raise ValueError(f"Unknown pencil selection: {settings.pencil_selection}")
    eligible = counts >= 2
    status = np.where(eligible, "capped", "below_two_lines").astype("U16")
    covered = np.zeros(masks.shape[1], dtype=bool)
    retained: list[int] = []
    for _ in range(settings.pencils):
        indices = np.flatnonzero(eligible)
        if not len(indices):
            break
        novelty = (
            (masks[indices] & ~covered).sum(axis=1) if settings.pencil_selection == "coverage" else counts[indices]
        )
        order = np.lexsort((candidate_ids[indices], -supports[indices], -counts[indices], -novelty))
        index = int(indices[order[0]])
        retained.append(index)
        covered |= masks[index]
        union = np.count_nonzero(masks[indices] | masks[index], axis=1)
        intersection = np.count_nonzero(masks[indices] & masks[index], axis=1)
        redundant = indices[intersection / union > settings.overlap]
        eligible[redundant] = False
        status[redundant] = "redundant"
        status[index] = "retained"
    return retained, status
```

### estimate

```python
def estimate(segments: np.ndarray, size: tuple[int, int], settings: Settings) -> tuple[np.ndarray, dict]:
    """Estimate competing homogeneous VPs from fragments without court labels."""
    observations = assignment.prepare_observations(segments, size)
    merge_settings = replace(detector.DEFAULT_SETTINGS, max_family_lines=300)
    all_lines = detector._merge_lines(observations.segments.reshape(-1, 4), merge_settings)
    lines = all_lines[:settings.direction_lines]
    transform = normalisation(size)
    normalised_lines = lines @ transform
    pairs = np.asarray(list(combinations(range(len(lines)), 2)), dtype=int).reshape(-1, 2)
    intersections = np.cross(normalised_lines[pairs[:, 0]], normalised_lines[pairs[:, 1]])
    infinity = np.column_stack((normalised_lines[:, 1], -normalised_lines[:, 0], np.zeros(len(lines))))
    candidates = np.concatenate((intersections, infinity))
    norms = np.linalg.norm(candidates, axis=1)
    nondegenerate = norms > 1e-12
    candidate_ids = np.flatnonzero(nondegenerate)
    candidates = candidates[nondegenerate] / norms[nondegenerate, None]
    masks, counts, supports = [], [], []
    for offset in range(0, len(candidates), settings.candidate_batch):
        angles = angular_residuals(normalised_lines, candidates[offset:offset + settings.candidate_batch])
        batch_masks = angles <= settings.angle_deg
        masks.extend(batch_masks)
        counts.extend(batch_masks.sum(axis=1))
        supports.extend(np.maximum(0, 1 - angles / settings.angle_deg).sum(axis=1))
    masks = np.asarray(masks, dtype=bool).reshape(-1, len(lines)) if len(lines) else np.empty((0, 0), dtype=bool)
    counts = np.asarray(counts, dtype=int)
    supports = np.asarray(supports)
    retained, status = retain_pencils(masks, counts, supports, candidate_ids, settings)
    selected = candidates[retained]
    native = selected @ transform.T
    provenance = []
    for line in lines:
        distances = np.abs(observations.segments @ line[:2] + line[2]).max(axis=1)
        angles = np.abs(observations.directions @ line[:2])
        compatible = (distances <= merge_settings.merge_distance) & (
            angles <= np.sin(np.deg2rad(merge_settings.merge_angle_deg))
        )
        provenance.append(observations.fragment_ids[compatible].tolist())
    details = {
        "raw_fragments": len(segments), "visible_fragments": len(observations.segments),
        "visible_raw_ids": observations.fragment_ids.tolist(),
        "preparation_groups": len(observations.groups),
        "merge_input_cap": 300, "merge_input_excluded": max(0, len(observations.segments) - 300),
        "merged_direction_count": len(all_lines), "direction_cap_excluded": len(all_lines) - len(lines),
        "direction_lines": lines.tolist(), "compatible_raw_ids": provenance,
        "provenance_note": "Geometric compatibility after fitting, not exact merge membership",
        "pair_candidates": len(pairs), "infinity_candidates": len(infinity),
        "degenerate_candidates": int((~nondegenerate).sum()),
        "candidate_ids": candidate_ids.tolist(), "support_counts": counts.tolist(),
        "candidate_status": status.tolist(), "retained_candidate_ids": candidate_ids[retained].tolist(),
        "retained_support_masks": masks[retained].tolist(), "points_working": native.tolist(),
        "normalised_to_working": transform.tolist(),
    }
    return native, details
```

## run_svd_fixed.py

Whole-source MD5: `ca28990becaedc52faa8fad9a2a97c59`.

### svd_direction

```python
def svd_direction(lines: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fit a unit homogeneous point to the supplied normalised line rows.

    :param lines: One homogeneous line per row, with three coefficients per line.
    :return: Direction and singular-value diagnostics; membership stays fixed.
    """
    assert lines.ndim == 2 and lines.shape[1] == 3 and len(lines) >= 2
    # Unit 2D normals remove arbitrary line scale without weighting by the offset.
    lines = lines / np.linalg.norm(lines[:, :2], axis=1)[:, None]
    _, values, right = np.linalg.svd(lines, full_matrices=True)
    singular_values = np.pad(values, (0, 3 - len(values)))
    point = right[-1]
    return point, {
        'line_count': len(lines),
        'singular_values': singular_values.tolist(),
        'normalised_nullspace_gap': float((singular_values[1] - singular_values[2]) / singular_values[0]),
        'algebraic_rms': float(np.sqrt(np.mean(np.square(lines @ point)))),
    }
```

### fit_groups

```python
def fit_groups(lines: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Fit each original membership mask independently without changing its rows."""
    points, records = [], []
    for index, mask in enumerate(masks):
        point, record = svd_direction(lines[mask])
        points.append(point)
        records.append({'group_index': index, 'support_line_ids': np.flatnonzero(mask).tolist(), **record})
    return np.asarray(points), records
```

### fit_pairs

```python
def fit_pairs(points: np.ndarray, corners: np.ndarray) -> dict:
    """Evaluate every ordered pair; retain failed solver attempts explicitly."""
    started = perf_counter()
    records = []
    for pair_id, indices in enumerate(permutations(range(len(points)), 2)):
        record = {'pair_id': pair_id, 'groups': list(indices)}
        try:
            fit = control_fit(points[list(indices)], corners)
            if not np.isfinite(fit['max_corner_working_px']):
                raise ValueError('Non-finite corner error')
            record.update(fit)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            record['failure'] = f'{type(error).__name__}: {error}'
        records.append(record)
    finite = [record for record in records if 'failure' not in record]
    converged = [record for record in finite if record['converged']]
    return {
        'records': records,
        'attempted': len(records),
        'failed': len(records) - len(finite),
        'converged': len(converged),
        'best_finite': min(finite, key=lambda row: row['max_corner_working_px'], default=None),
        'best_converged': min(converged, key=lambda row: row['max_corner_working_px'], default=None),
        'elapsed_s': perf_counter() - started,
    }
```

## diagnose_directions.py

Whole-source MD5: `5ffe3d4ff4baaf5ff5cd4dd8a369102b`.

### control_fit

```python
def control_fit(points: np.ndarray, corners: np.ndarray) -> dict:
    """Find a control fit with fixed direction columns; this does not use observed lines."""
    directions = points.T / np.linalg.norm(points, axis=1)
    court = detector.CORNER_COURT_M.astype(float)
    coefficients = np.zeros((4, 2, 4))
    for coordinate in range(2):
        coefficients[:, coordinate, :2] = court * (
            directions[coordinate] - corners[:, coordinate, None] * directions[2])
        coefficients[:, coordinate, coordinate + 2] = 1.
    initial = np.linalg.lstsq(coefficients.reshape(8, 4), corners.ravel(), rcond=None)[0]

    def transform(parameters: np.ndarray) -> np.ndarray:
        return np.column_stack((directions * parameters[:2], [parameters[2], parameters[3], 1.]))

    def residual(parameters: np.ndarray) -> np.ndarray:
        projected, _ = detector.project(transform(parameters)[None], court)
        return (projected[0] - corners).ravel()

    fit = least_squares(residual, initial, x_scale='jac', max_nfev=200)
    errors = residual(fit.x).reshape(4, 2)
    return {'max_corner_working_px': float(np.linalg.norm(errors, axis=1).max()),
            'rms_coordinate_working_px': float(np.sqrt(np.square(errors).mean())),
            'converged': bool(fit.success), 'evaluations': fit.nfev,
            'homography_working': transform(fit.x).tolist()}
```

## projective_seed.py

Whole-source MD5: `4cdb8e863318d5a8982a56f68005a1dc`.

### corner_errors

```python
def corner_errors(corners: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Compare physical courts while allowing the canonical 180-degree relabelling."""
    direct = np.linalg.norm(corners - reference, axis=-1).max(axis=-1)
    rotated = np.linalg.norm(corners - reference[[2, 3, 0, 1]], axis=-1).max(axis=-1)
    return np.minimum(direct, rotated)
```

## inspect_appearance.py

Whole-source MD5: `1df7afd8e59ad019f20d92555162fbcb`.

### ridge_mask

```python
def ridge_mask(frame: np.ndarray, segments: np.ndarray) -> np.ndarray:
    accepted = detector._filter_painted_stripes(frame, segments)
    identities = {tuple(segment) for segment in accepted}
    return np.asarray([tuple(segment) in identities for segment in segments], dtype=bool)
```

### profiles

```python
def profiles(frame: np.ndarray, homographies: np.ndarray) -> list[dict]:
    """Measure finite visible intervals, retaining missing profiles separately."""
    height, width = frame.shape[:2]
    projected, _ = detector.project(homographies, detector.SEGMENTS_M)
    endpoints, visible = detector._visible_samples(projected.reshape(-1, 12, 2, 2), (width, height), 2)
    passed = np.zeros(visible.shape, dtype=bool)
    passed[visible] = ridge_mask(frame, endpoints[visible].reshape(-1, 4))
    results = []
    for interval_pass, interval_visible in zip(passed, visible, strict=True):
        markings = []
        for intervals in assignment.MARKING_INTERVALS:
            available = interval_visible[list(intervals)]
            markings.append(float(interval_pass[list(intervals)][available].mean()) if available.any() else None)
        available_scores = [value for value in markings if value is not None]
        results.append({'score': float(np.mean(available_scores)) if available_scores else None,
                        'marking_ridge': markings, 'interval_visible': interval_visible.tolist(),
                        'interval_ridge': interval_pass.tolist()})
    return results
```

## run_automatic.py

Whole-source MD5: `e5287191ff275d383e02cbc6d8a4b034`.

### winner_ids

```python
def winner_ids(entries: list[dict]) -> tuple[str | None, str | None]:
    """Preserve the two existing rankings within the same camera-eligible pool."""
    eligible = [entry for entry in entries if entry['gates']['camera_error'] is not None
                and entry['gates']['camera_error'] <= CAMERA_ERROR_LIMIT and entry['profile']['score'] is not None]
    line = max(eligible, key=lambda entry: entry['stripe']['exclusive']['score'], default=None)
    paint = max(eligible, key=lambda entry: (entry['profile']['score'], entry['stripe']['exclusive']['score']), default=None)
    return (None if line is None else line['candidate_id'], None if paint is None else paint['candidate_id'])
```

## assignment.py

Whole-source MD5: `170deb40fa498b9633dc050cb9f089ac`.

### MARKINGS

```python
MARKINGS = (
    "left_doubles", "left_singles", "centre", "right_singles", "right_doubles",
    "far_baseline", "far_long_service", "far_short_service",
    "near_short_service", "near_long_service", "near_baseline",
)
```

### MARKING_INTERVALS

```python
MARKING_INTERVALS = ((0,), (1,), (2, 3), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,))
```

## detector.py

Whole-source MD5: `e26a6498bc1b20812569c3c6f46144bf`.

### SEGMENTS_M

```python
SEGMENTS_M = np.asarray(PAINTED_SEGMENTS_M, dtype=np.float64)
```

### RIDGE_SAMPLES

```python
RIDGE_SAMPLES = 24
```

### RIDGE_CENTRE_SHIFTS

```python
RIDGE_CENTRE_SHIFTS = np.array([-4, -2, 0, 2, 4], dtype=np.float32)
```

### RIDGE_SIDE_DISTANCE

```python
RIDGE_SIDE_DISTANCE = 6.0
```

### RIDGE_MIN_CONTRAST

```python
RIDGE_MIN_CONTRAST = 10.0
```

### RIDGE_MIN_FRACTION

```python
RIDGE_MIN_FRACTION = 0.4
```

### _filter_painted_stripes

```python
def _filter_painted_stripes(frame: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Keep bright stripes with darker pixels on both sides, regardless of colour.

    Canny marks stripe edges, so sample several nearby centres along the normal.
    This is an optional evidence filter; weak or crowded markings can be lost.
    Input fragments from the extractors have finite endpoints and positive length.
    """
    if not len(segments):
        return segments
    endpoints = segments.reshape(-1, 2, 2).astype(np.float32)
    vectors = endpoints[:, 1] - endpoints[:, 0]
    normals = np.stack((-vectors[:, 1], vectors[:, 0]), axis=1)
    normals /= np.linalg.norm(vectors, axis=1)[:, None]
    fractions = np.linspace(0, 1, RIDGE_SAMPLES, dtype=np.float32)
    centres = endpoints[:, None, 0] + vectors[:, None] * fractions[None, :, None]
    shifted = centres[:, :, None] + normals[:, None, None] * RIDGE_CENTRE_SHIFTS[None, None, :, None]
    sides = RIDGE_SIDE_DISTANCE * normals[:, None, None]
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = frame.shape[:2]
    intensities = []
    in_frame = []
    # Each sample tests five possible centres and the pixels on either side.
    for points in (shifted, shifted - sides, shifted + sides):
        maps = points.reshape(len(segments), -1, 2)
        sampled = cv2.remap(grey, maps[..., 0], maps[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        intensities.append(sampled.reshape(points.shape[:-1]))
        in_frame.append(
            (points[..., 0] >= 0) & (points[..., 0] < width)
            & (points[..., 1] >= 0) & (points[..., 1] < height)
        )
    centre, first_side, second_side = intensities
    contrast = np.minimum(centre - first_side, centre - second_side)
    visible = in_frame[0] & in_frame[1] & in_frame[2]
    contrast = np.where(visible, contrast, -np.inf)
    ridge_samples = contrast.max(axis=2) >= RIDGE_MIN_CONTRAST
    return segments[ridge_samples.mean(axis=1) >= RIDGE_MIN_FRACTION]
```

### _visible_samples

```python
def _visible_samples(endpoints: np.ndarray, size: tuple[int, int], count: int) -> tuple[np.ndarray, np.ndarray]:
    """Clip finite projected markings before sampling them uniformly in image space."""
    starts = endpoints[:, :, 0]
    vectors = endpoints[:, :, 1] - starts
    lower = np.zeros(starts.shape[:2])
    upper = np.ones(starts.shape[:2])
    visible = np.ones(starts.shape[:2], dtype=bool)
    for axis, limit in enumerate(size):
        stationary = np.abs(vectors[..., axis]) < 1e-8
        visible &= ~stationary | ((starts[..., axis] >= 0) & (starts[..., axis] <= limit - 1))
        divisor = np.where(stationary, 1, vectors[..., axis])
        first = -starts[..., axis] / divisor
        last = (limit - 1 - starts[..., axis]) / divisor
        lower = np.maximum(lower, np.where(stationary, -np.inf, np.minimum(first, last)))
        upper = np.minimum(upper, np.where(stationary, np.inf, np.maximum(first, last)))
    visible &= upper > lower
    clipped_length = (upper - lower) * np.linalg.norm(vectors, axis=-1)
    visible &= clipped_length >= 12
    fractions = lower[..., None] + (upper - lower)[..., None] * np.linspace(0, 1, count)
    samples = starts[..., None, :] + fractions[..., None] * vectors[..., None, :]
    return samples, visible
```
