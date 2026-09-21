"""Focused tests for the strict frozen-case provenance adapter."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import experiments.annotator.independent_court.case_provenance as provenance_module
from experiments.annotator.independent_court.case_provenance import (
    SIDECAR_MD5,
    BoxRelation,
    ImageKind,
    load_frozen_case_provenance,
    require_same_image_boxes,
)

ROOT = Path(__file__).parents[1]
FROZEN = ROOT / "scratch/court_det_fix/frozen_views"
PACKS = FROZEN / "packs"
SIDECAR = FROZEN / "case_provenance.json.gz"
PACK_NAMES = (
    "broadcast_extension_inputs.json.gz",
    "gx_extension_inputs.json.gz",
    "marking_refit_inputs.json.gz",
)


def _read_sidecar(path: Path = SIDECAR) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def _copy_frozen_set(tmp_path: Path) -> Path:
    frozen = tmp_path / "frozen_views"
    packs = frozen / "packs"
    packs.mkdir(parents=True)
    for name in PACK_NAMES:
        shutil.copyfile(PACKS / name, packs / name)
    shutil.copyfile(SIDECAR, frozen / SIDECAR.name)
    return packs


def _write_sidecar(path: Path, value: dict) -> None:
    path.write_bytes(gzip.compress(json.dumps(value, sort_keys=True).encode("utf-8"), mtime=0))


def test_sidecar_is_deterministic_and_has_only_the_strict_shape() -> None:
    raw = SIDECAR.read_bytes()
    assert raw == SIDECAR.read_bytes()
    assert int.from_bytes(raw[4:8], "little") == 0
    assert hashlib.md5(raw, usedforsecurity=False).hexdigest() == SIDECAR_MD5

    sidecar = _read_sidecar()
    assert set(sidecar) == {"schema", "packs"}
    assert sidecar["schema"] == "independent-court-case-provenance/1"
    assert set(sidecar["packs"]) == set(PACK_NAMES)
    assert all(set(entry) == {"md5", "cases"} for entry in sidecar["packs"].values())
    assert all(
        set(case) == {"image", "box_frame_index"}
        for entry in sidecar["packs"].values()
        for case in entry["cases"].values()
    )


def test_exact_population_and_known_cases() -> None:
    mappings = [load_frozen_case_provenance(PACKS / name) for name in PACK_NAMES]
    all_cases = [case for mapping in mappings for case in mapping.values()]

    assert len(all_cases) == 47
    assert sum(case.has_same_image_boxes for case in all_cases) == 17
    assert sum(case.box_relation is BoxRelation.NEARBY_SOURCE_FRAME for case in all_cases) == 10
    assert sum(case.image_kind is ImageKind.COMPOSITE for case in all_cases) == 20

    nearby_ids = {
        case.case_id
        for case in all_cases
        if case.box_relation is BoxRelation.NEARBY_SOURCE_FRAME
    }
    assert nearby_ids == {
        "gxBQ_window_00_frame_5",
        "gxBQ_window_01_frame_5111",
        "gxBQ_window_02_frame_5766",
        "gxBQ_window_03_frame_77876",
        "gxBQ_window_04_frame_86088",
        "yellow_short_frame_14",
        "letterboxed_short_frame_58",
        "centre_short_frame_64",
        "centre_short_frame_71",
        "am4_window_00_frame_319",
    }

    gx5 = mappings[1]["gxBQ_window_00_frame_5"]
    assert gx5.image_kind is ImageKind.SOURCE_FRAME
    assert gx5.image_frame_indices == (5,)
    assert gx5.box_frame_index == 6
    assert gx5.unavailable_reason == (
        "same-image person boxes unavailable: boxes come from nearby source frame "
        "6, while the image is source frame 5"
    )

    broadcast = mappings[0]["shuttleset_03_scene_0017"]
    assert broadcast.image_kind is ImageKind.COMPOSITE
    assert broadcast.image_frame_indices == (38343, 38525, 38671)
    assert broadcast.box_frame_index == 38343
    assert broadcast.box_relation is BoxRelation.COMPOSITE
    assert broadcast.unavailable_reason == "same-image person boxes unavailable: measured image is a composite"


def test_require_same_image_boxes_returns_same_case_or_fails() -> None:
    mapping = load_frozen_case_provenance(PACKS / "gx_extension_inputs.json.gz")
    same = mapping["gxBQ_window_00_frame_0"]
    assert require_same_image_boxes(same) is same
    with pytest.raises(ValueError, match="nearby source frame 6"):
        require_same_image_boxes(mapping["gxBQ_window_00_frame_5"])


def test_pack_mutation_fails_pinned_hash_gate(tmp_path: Path) -> None:
    packs = _copy_frozen_set(tmp_path)
    pack_path = packs / "gx_extension_inputs.json.gz"
    mutated = bytearray(pack_path.read_bytes())
    mutated[len(mutated) // 2] ^= 1
    pack_path.write_bytes(mutated)

    with pytest.raises(ValueError, match="MD5 mismatch"):
        load_frozen_case_provenance(pack_path)


def test_sidecar_case_set_mismatch_fails_loudly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    packs = _copy_frozen_set(tmp_path)
    sidecar_path = packs.parent / SIDECAR.name
    sidecar = _read_sidecar(sidecar_path)
    del sidecar["packs"]["gx_extension_inputs.json.gz"]["cases"]["gxBQ_window_00_frame_0"]
    _write_sidecar(sidecar_path, sidecar)
    monkeypatch.setattr(provenance_module, "SIDECAR_MD5", hashlib.md5(sidecar_path.read_bytes(), usedforsecurity=False).hexdigest())

    with pytest.raises(ValueError, match="case-set mismatch"):
        load_frozen_case_provenance(packs / "gx_extension_inputs.json.gz")


def test_schema_valid_classification_change_fails_sidecar_hash_gate(tmp_path: Path) -> None:
    packs = _copy_frozen_set(tmp_path)
    sidecar_path = packs.parent / SIDECAR.name
    sidecar = _read_sidecar(sidecar_path)
    case = sidecar["packs"]["gx_extension_inputs.json.gz"]["cases"]["gxBQ_window_00_frame_0"]
    case["box_frame_index"] = 1
    _write_sidecar(sidecar_path, sidecar)

    with pytest.raises(ValueError, match="sidecar MD5 mismatch"):
        load_frozen_case_provenance(packs / "gx_extension_inputs.json.gz")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda case: case["image"].update(kind="unknown"), "unknown image kind"),
        (lambda case: case.update(image={"kind": "composite", "frame_indices": []}), "non-empty list"),
        (lambda case: case.update(relation="same_image"), "unknown relation"),
        (lambda case: case.pop("box_frame_index"), "missing box_frame_index"),
    ],
)
def test_malformed_or_unknown_provenance_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change, message: str
) -> None:
    packs = _copy_frozen_set(tmp_path)
    sidecar_path = packs.parent / SIDECAR.name
    sidecar = _read_sidecar(sidecar_path)
    case = sidecar["packs"]["gx_extension_inputs.json.gz"]["cases"]["gxBQ_window_00_frame_0"]
    change(case)
    _write_sidecar(sidecar_path, sidecar)
    monkeypatch.setattr(provenance_module, "SIDECAR_MD5", hashlib.md5(sidecar_path.read_bytes(), usedforsecurity=False).hexdigest())

    with pytest.raises(ValueError, match=message):
        load_frozen_case_provenance(packs / "gx_extension_inputs.json.gz")


def test_loader_requires_a_pinned_path_basename_and_path_type(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="pathlib.Path"):
        load_frozen_case_provenance(str(PACKS / PACK_NAMES[0]))  # type: ignore[arg-type]
    unsupported = tmp_path / "other.json.gz"
    unsupported.write_bytes(b"not a pack")
    with pytest.raises(ValueError, match="unsupported frozen pack basename"):
        load_frozen_case_provenance(unsupported)


def test_pack_and_case_records_are_immutable() -> None:
    mapping = load_frozen_case_provenance(PACKS / "gx_extension_inputs.json.gz")
    with pytest.raises(TypeError):
        mapping["new"] = mapping["gxBQ_window_00_frame_0"]  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        mapping["gxBQ_window_00_frame_0"].box_frame_index = 2  # type: ignore[misc]


def test_pack_file_hashes_are_the_sidecar_hashes() -> None:
    sidecar = _read_sidecar()
    for name in PACK_NAMES:
        actual = hashlib.md5((PACKS / name).read_bytes(), usedforsecurity=False).hexdigest()
        assert actual == sidecar["packs"][name]["md5"]
