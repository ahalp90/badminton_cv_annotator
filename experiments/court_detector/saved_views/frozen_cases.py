"""Load the frozen research views: pinned packs, their provenance sidecar and frames.

The saved-view runner, research scripts and tests read these fixtures. The court
detector does not: its callers pass a view's image, lines and provenance in
`ViewInputs`. The loader checks pack bytes and case IDs before returning
immutable provenance records. Source frames are read from those records, never
inferred from case names.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterator, Mapping
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import cv2

from court_detector.image_sources import CaseProvenance, ImageKind
from court_detector.measurements import ViewContext, read_json_gz, view_context

# The frozen packs and their source frames live beside this research runner.
ROOT = Path(__file__).resolve().parent / "data"

SIDECAR_FILENAME = "case_provenance.json.gz"
SIDECAR_SCHEMA = "independent-court-case-provenance/1"
SIDECAR_MD5 = "c1893e6217065d6038cc9b8f05cd1b98"
PACK_MD5_BY_NAME: Mapping[str, str] = MappingProxyType(
    {
        "broadcast_extension_inputs.json.gz": "cf71a4e217f2ce7dee2b3e81f0ec845c",
        "gx_extension_inputs.json.gz": "45bea3597cece373f2653da17c38e8d7",
        "marking_refit_inputs.json.gz": "82c710ce0c8c082cdfd3aeecb4d5f144",
    }
)

CASE_PACKS = {
    "gx": "packs/gx_extension_inputs.json.gz",
    "amateur": "packs/marking_refit_inputs.json.gz",
    "broadcast": "packs/broadcast_extension_inputs.json.gz",
}
REGRESSION_CASE_ORDER = (
    ("gxBQ_window_00_frame_0", "gx", "GX0"),
    ("gxBQ_window_00_frame_5", "gx", "GX5"),
    ("am2_window_00_frame_150", "amateur", "Am2-150"),
    ("am2_window_01_frame_28019", "amateur", "Am2-28019"),
    ("am3_window_00_frame_0", "amateur", "Am3-0"),
    ("shuttleset_03_scene_0017", "broadcast", "SS03-17"),
    ("shuttleset_03_scene_0019", "broadcast", "SS03-19"),
    ("shuttleset_03_scene_0016", "broadcast", "SS03-16"),
    ("shuttleset_21_scene_0020", "broadcast", "SS21-20"),
)
UNUSED_CASE_ORDER = (
    ("gxBQ_window_00_frame_689", "gx", "gxBQ_window_00_frame_689"),
    ("gxBQ_window_01_frame_5111", "gx", "gxBQ_window_01_frame_5111"),
    ("gxBQ_window_02_frame_5766", "gx", "gxBQ_window_02_frame_5766"),
    ("gxBQ_window_03_frame_77876", "gx", "gxBQ_window_03_frame_77876"),
    ("gxBQ_window_04_frame_86088", "gx", "gxBQ_window_04_frame_86088"),
    ("yellow_short_frame_14", "amateur", "yellow_short_frame_14"),
    ("letterboxed_short_frame_45", "amateur", "letterboxed_short_frame_45"),
    ("centre_short_frame_36", "amateur", "centre_short_frame_36"),
    ("am1_window_00_frame_54", "amateur", "am1_window_00_frame_54"),
    ("am3_window_01_frame_10514", "amateur", "am3_window_01_frame_10514"),
    ("am4_window_00_frame_0", "amateur", "am4_window_00_frame_0"),
    ("am4_window_01_frame_13782", "amateur", "am4_window_01_frame_13782"),
    ("shuttleset_03_scene_0029", "broadcast", "shuttleset_03_scene_0029"),
    ("shuttleset_03_scene_0034", "broadcast", "shuttleset_03_scene_0034"),
    ("shuttleset_03_scene_0038", "broadcast", "shuttleset_03_scene_0038"),
    ("shuttleset_21_scene_0000", "broadcast", "shuttleset_21_scene_0000"),
    ("shuttleset_21_scene_0010", "broadcast", "shuttleset_21_scene_0010"),
    ("shuttleset_21_scene_0039", "broadcast", "shuttleset_21_scene_0039"),
)
CASE_ORDER = REGRESSION_CASE_ORDER
ALL_CASE_ORDER = REGRESSION_CASE_ORDER + UNUSED_CASE_ORDER
REGRESSION_CASE_IDS = tuple(case_id for case_id, _, _ in REGRESSION_CASE_ORDER)
UNUSED_CASE_IDS = tuple(case_id for case_id, _, _ in UNUSED_CASE_ORDER)
ALL_CASE_IDS = tuple(case_id for case_id, _, _ in ALL_CASE_ORDER)
CASE_IDS = REGRESSION_CASE_IDS
CASE_LABELS = {case_id: label for case_id, _, label in ALL_CASE_ORDER}
PACK_OF = {case_id: pack for case_id, pack, _ in ALL_CASE_ORDER}


class _FrozenCaseMapping(Mapping[str, CaseProvenance]):
    """Read-only mapping returned by the loader."""

    __slots__ = ("_cases",)

    def __init__(self, cases: Mapping[str, CaseProvenance]) -> None:
        object.__setattr__(self, "_cases", MappingProxyType(dict(cases)))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("case provenance mapping is immutable")

    def __getitem__(self, case_id: str) -> CaseProvenance:
        return self._cases[case_id]

    def __iter__(self) -> Iterator[str]:
        return iter(self._cases)

    def __len__(self) -> int:
        return len(self._cases)


JsonObject = Mapping[str, Any]


def _fail(context: str, message: str) -> ValueError:
    return ValueError(f"{context}: {message}")


def _object(value: Any, context: str) -> JsonObject:
    if not isinstance(value, Mapping):
        raise _fail(context, "must be an object")
    return cast(JsonObject, value)


def _frame(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _fail(context, "must be a non-negative integer")
    return value


def _md5(value: Any, context: str) -> str:
    if not isinstance(value, str) or len(value) != 32:
        raise _fail(context, "must be a 32-character MD5 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise _fail(context, "must be a 32-character MD5 hex digest") from error
    return value.lower()


def _file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_gzip_json(path: Path) -> JsonObject:
    if not path.is_file():
        raise FileNotFoundError(f"provenance input does not exist: {path}")
    try:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            value = json.load(source)
    except (OSError, EOFError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read gzip JSON: {path}") from error
    return _object(value, str(path))


def _pack_cases(pack: JsonObject) -> tuple[dict[str, JsonObject], tuple[str, ...]]:
    raw_cases = pack.get("cases")
    if not isinstance(raw_cases, list):
        raise _fail("pack", "cases must be a list")
    cases: dict[str, JsonObject] = {}
    ordered_ids: list[str] = []
    for position, value in enumerate(raw_cases):
        context = f"pack.cases[{position}]"
        case = _object(value, context)
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise _fail(f"{context}.id", "must be a non-empty string")
        if case_id in cases:
            raise _fail("pack.cases", f"duplicate case ID {case_id!r}")
        cases[case_id] = case
        ordered_ids.append(case_id)
    return cases, tuple(ordered_ids)


def _image_provenance(value: Any, context: str) -> tuple[ImageKind, tuple[int, ...]]:
    image = _object(value, context)
    allowed = {"kind", "frame_index", "frame_indices"}
    if set(image) - allowed:
        unknown = sorted(set(image) - allowed)
        raise _fail(context, f"unknown fields: {', '.join(unknown)}")
    raw_kind = image.get("kind")
    if not isinstance(raw_kind, str):
        raise _fail(f"{context}.kind", "must be a string")
    try:
        kind = ImageKind(raw_kind)
    except ValueError as error:
        raise _fail(f"{context}.kind", f"unknown image kind {raw_kind!r}") from error

    has_frame = "frame_index" in image
    has_frames = "frame_indices" in image
    if has_frame == has_frames:
        raise _fail(context, "must contain exactly one of frame_index or frame_indices")
    if kind is ImageKind.SOURCE_FRAME:
        if not has_frame:
            raise _fail(context, "source_frame requires frame_index")
        return kind, (_frame(image["frame_index"], f"{context}.frame_index"),)
    if not has_frames:
        raise _fail(context, "composite requires frame_indices")
    raw_frames = image["frame_indices"]
    if not isinstance(raw_frames, list) or not raw_frames:
        raise _fail(f"{context}.frame_indices", "must be a non-empty list")
    frames = tuple(_frame(item, f"{context}.frame_indices[{index}]") for index, item in enumerate(raw_frames))
    if len(set(frames)) != len(frames):
        raise _fail(f"{context}.frame_indices", "must not contain duplicates")
    return kind, frames


def _case_provenance(value: Any, case_id: str, context: str) -> CaseProvenance:
    record = _object(value, context)
    required = {"image", "box_frame_index"}
    if set(record) != required:
        missing = required - set(record)
        extra = set(record) - required
        detail = []
        if missing:
            detail.append(f"missing {', '.join(sorted(missing))}")
        if extra:
            detail.append(f"unknown {', '.join(sorted(extra))}")
        raise _fail(context, "; ".join(detail))
    image_kind, image_frames = _image_provenance(record["image"], f"{context}.image")
    box_frame = _frame(record["box_frame_index"], f"{context}.box_frame_index")
    return CaseProvenance(case_id, image_kind, image_frames, box_frame)


def _sidecar_entry(sidecar: JsonObject, pack_name: str) -> JsonObject:
    required = {"schema", "packs"}
    if set(sidecar) != required:
        missing = required - set(sidecar)
        extra = set(sidecar) - required
        detail = []
        if missing:
            detail.append(f"missing {', '.join(sorted(missing))}")
        if extra:
            detail.append(f"unknown {', '.join(sorted(extra))}")
        raise _fail("sidecar", "; ".join(detail))
    if sidecar["schema"] != SIDECAR_SCHEMA:
        raise _fail("sidecar.schema", f"unsupported schema {sidecar['schema']!r}")
    packs = _object(sidecar["packs"], "sidecar.packs")
    if set(packs) != set(PACK_MD5_BY_NAME):
        missing = sorted(set(PACK_MD5_BY_NAME) - set(packs))
        extra = sorted(set(packs) - set(PACK_MD5_BY_NAME))
        raise _fail("sidecar.packs", f"pack-set mismatch (missing={missing!r}, extra={extra!r})")
    entry = _object(packs.get(pack_name), f"sidecar.packs[{pack_name!r}]")
    required_entry = {"md5", "cases"}
    if set(entry) != required_entry:
        missing = required_entry - set(entry)
        extra = set(entry) - required_entry
        detail = []
        if missing:
            detail.append(f"missing {', '.join(sorted(missing))}")
        if extra:
            detail.append(f"unknown {', '.join(sorted(extra))}")
        raise _fail(f"sidecar.packs[{pack_name!r}]", "; ".join(detail))
    return entry


def load_frozen_case_provenance(pack_path: Path) -> Mapping[str, CaseProvenance]:
    """Load one pinned frozen pack and return its immutable case mapping.

    :param pack_path: Path to one of the three checked frozen pack files.
    :return: Read-only mapping in pack case order.
    """
    if not isinstance(pack_path, Path):
        raise TypeError("pack_path must be a pathlib.Path")
    expected_md5 = PACK_MD5_BY_NAME.get(pack_path.name)
    if expected_md5 is None:
        raise ValueError(f"unsupported frozen pack basename: {pack_path.name!r}")
    actual_md5 = _file_md5(pack_path)
    if actual_md5 != expected_md5:
        raise ValueError(f"{pack_path.name}: pack MD5 mismatch (expected {expected_md5}, got {actual_md5})")

    pack = _read_gzip_json(pack_path)
    _, ordered_ids = _pack_cases(pack)
    sidecar_path = pack_path.parent.parent / SIDECAR_FILENAME
    actual_sidecar_md5 = _file_md5(sidecar_path)
    if actual_sidecar_md5 != SIDECAR_MD5:
        raise ValueError(
            f"{sidecar_path.name}: sidecar MD5 mismatch (expected {SIDECAR_MD5}, got {actual_sidecar_md5})"
        )
    sidecar = _read_gzip_json(sidecar_path)
    entry = _sidecar_entry(sidecar, pack_path.name)
    if _md5(entry["md5"], "sidecar.md5") != expected_md5:
        raise _fail(f"sidecar.packs[{pack_path.name!r}].md5", "does not match pinned pack MD5")
    sidecar_cases = _object(entry["cases"], f"sidecar.packs[{pack_path.name!r}].cases")
    if set(sidecar_cases) != set(ordered_ids):
        missing = sorted(set(ordered_ids) - set(sidecar_cases))
        extra = sorted(set(sidecar_cases) - set(ordered_ids))
        raise _fail("sidecar cases", f"case-set mismatch (missing={missing!r}, extra={extra!r})")

    parsed = {
        case_id: _case_provenance(sidecar_cases[case_id], case_id, f"sidecar case {case_id!r}")
        for case_id in ordered_ids
    }
    return _FrozenCaseMapping(parsed)


def relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def load_source(root: Path, case_id: str) -> dict:
    source_pack = read_json_gz(root / CASE_PACKS[PACK_OF[case_id]])
    return next(source for source in source_pack["cases"] if source["id"] == case_id)


def frame_path(root: Path, source: dict, provenance: CaseProvenance) -> Path:
    case_id = source["id"]
    if case_id.startswith("gxBQ"):
        return root / "frames/gx" / source["image"]
    if case_id.startswith("shuttleset"):
        return root / "frames/original" / source["image"]
    video = case_id.split("_", 1)[0]
    frame = int(case_id.rsplit("_", 1)[1])
    if provenance.image_kind.value != "source_frame" or provenance.image_frame_indices != (frame,):
        raise ValueError(
            f"{case_id}: amateur frame path uses frame {frame}, but provenance identifies "
            f"{provenance.image_kind.value} frames {provenance.image_frame_indices}"
        )
    return root / "frames/amateur" / video / f"frame_{frame:08d}.png"


@lru_cache(maxsize=3)
def _load_provenance_pack(pack_path: Path) -> Mapping[str, CaseProvenance]:
    return load_frozen_case_provenance(pack_path)


def load_case_provenance(root: Path, case_id: str) -> CaseProvenance:
    """Load one case's typed provenance from its frozen input pack."""
    return _load_provenance_pack(root / CASE_PACKS[PACK_OF[case_id]])[case_id]


def prepare_view(root: Path, case_id: str) -> ViewContext:
    source = load_source(root, case_id)
    provenance = load_case_provenance(root, case_id)
    frame_file = frame_path(root, source, provenance)
    frame = cv2.imread(str(frame_file))
    if frame is None:
        raise FileNotFoundError(frame_file)
    return view_context(case_id, source, provenance, frame, relative_path(frame_file, root))
