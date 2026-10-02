"""Strict provenance adapter for the three frozen independent-court packs.

The packs are legacy records, so their image and selected person-box frames
are kept in one checked sidecar.  The adapter validates the pack bytes and
case IDs before returning immutable case records.  It deliberately does not
infer provenance from case names.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

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


class ImageKind(StrEnum):
    """Kind of pixels used for the measured image."""

    SOURCE_FRAME = "source_frame"
    COMPOSITE = "composite"


class BoxRelation(StrEnum):
    """Relationship between selected person boxes and the measured image."""

    SAME_IMAGE = "same_image"
    NEARBY_SOURCE_FRAME = "nearby_source_frame"
    COMPOSITE = "composite"


@dataclass(frozen=True, slots=True)
class CaseProvenance:
    """Validated immutable provenance for one frozen case."""

    case_id: str
    image_kind: ImageKind
    image_frame_indices: tuple[int, ...]
    box_frame_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("case_id must be a non-empty string")
        if not isinstance(self.image_kind, ImageKind):
            raise TypeError("image_kind must be an ImageKind")
        if not isinstance(self.image_frame_indices, tuple) or not self.image_frame_indices:
            raise ValueError("image_frame_indices must be a non-empty tuple")
        if any(isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 for frame in self.image_frame_indices):
            raise ValueError("image_frame_indices must contain non-negative integers")
        if self.image_kind is ImageKind.SOURCE_FRAME and len(self.image_frame_indices) != 1:
            raise ValueError("source_frame requires exactly one image frame")
        if len(set(self.image_frame_indices)) != len(self.image_frame_indices):
            raise ValueError("image_frame_indices must not contain duplicates")
        if isinstance(self.box_frame_index, bool) or not isinstance(self.box_frame_index, int) or self.box_frame_index < 0:
            raise ValueError("box_frame_index must be a non-negative integer")

    @property
    def box_relation(self) -> BoxRelation:
        """Return the relation derived from image kind and frame identity."""

        if self.image_kind is ImageKind.COMPOSITE:
            return BoxRelation.COMPOSITE
        if self.box_frame_index == self.image_frame_indices[0]:
            return BoxRelation.SAME_IMAGE
        return BoxRelation.NEARBY_SOURCE_FRAME

    @property
    def has_same_image_boxes(self) -> bool:
        """Return whether person boxes describe the measured image exactly."""

        return self.box_relation is BoxRelation.SAME_IMAGE

    @property
    def unavailable_reason(self) -> str | None:
        """Return a plain reason when same-image person boxes are unavailable."""

        if self.has_same_image_boxes:
            return None
        if self.box_relation is BoxRelation.COMPOSITE:
            return "same-image person boxes unavailable: measured image is a composite"
        return (
            "same-image person boxes unavailable: boxes come from nearby source frame "
            f"{self.box_frame_index}, while the image is source frame {self.image_frame_indices[0]}"
        )


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


def require_same_image_boxes(case: CaseProvenance) -> CaseProvenance:
    """Require same-image person boxes and return the validated case."""

    if not isinstance(case, CaseProvenance):
        raise TypeError("case must be a CaseProvenance")
    if not case.has_same_image_boxes:
        raise ValueError(case.unavailable_reason or "same-image person boxes unavailable")
    return case


__all__ = [
    "BoxRelation",
    "CaseProvenance",
    "ImageKind",
    "load_frozen_case_provenance",
    "require_same_image_boxes",
]
