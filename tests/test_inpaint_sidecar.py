"""Tests for the TrackNetV3 inpaint fill-mask sidecar writer."""

from __future__ import annotations

import ast
import builtins
import copy
import gzip
import importlib.util
import json
import os
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKNET_DIR = REPO_ROOT / 'src' / 'shared' / 'tracknetv3'
WRITER_PATH = TRACKNET_DIR / 'write_inpaint_metadata.py'
WRITER_SPEC = importlib.util.spec_from_file_location('authoritative_inpaint_metadata', WRITER_PATH)
if WRITER_SPEC is None or WRITER_SPEC.loader is None:
    raise RuntimeError(f'Could not load {WRITER_PATH}')
WRITER_MODULE = importlib.util.module_from_spec(WRITER_SPEC)
WRITER_SPEC.loader.exec_module(WRITER_MODULE)
write_inpaint_metadata = WRITER_MODULE.write_inpaint_metadata


def _write_sidecar(
    tmp_path: Path,
    *,
    mask: list[int],
    frames: list[int],
    inpaintnet: object | None = object(),
    eval_mode: str = 'weight',
    tracknet_seq_len: int = 8,
    tracknet_frames: list[int] | None = None,
    visibility: list[int] | None = None,
    video_name: str = '1.mp4',
) -> tuple[Path, dict[str, object], str]:
    video_dir = tmp_path / 'videos'
    output_dir = tmp_path / 'output'
    video_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / video_name
    csv_path = output_dir / 'renamed_ball.csv'
    final_visibility = visibility if visibility is not None else [1] * len(frames)
    tracknet_frame_ids = tracknet_frames if tracknet_frames is not None else list(frames)
    tracknet_pred_dict = {'Frame': tracknet_frame_ids, 'Inpaint_Mask': mask}
    pred_dict = {'Frame': frames, 'Visibility': final_visibility}

    write_inpaint_metadata(
        csv_path,
        tracknet_pred_dict=tracknet_pred_dict,
        pred_dict=pred_dict,
        video_file=video_path,
        eval_mode=eval_mode,
        tracknet_seq_len=tracknet_seq_len,
        h=288.0,
        inpaintnet=inpaintnet,
        tracknet_ckpt='TrackNet_best.pt',
        inpaintnet_ckpt='InpaintNet_best.pt',
    )

    sidecar_path = output_dir / (
        f'{Path(video_name).stem}_stride'
        f'{tracknet_seq_len if eval_mode == "nonoverlap" else 1}_inpaint_mask.json.gz'
    )
    with gzip.open(sidecar_path, 'rt', encoding='utf-8') as sidecar_file:
        raw_text = sidecar_file.read()
    return sidecar_path, json.loads(raw_text), raw_text


@pytest.mark.parametrize(
    ('mask', 'frames', 'expected'),
    [
        ([], [], []),
        ([1, 1, 1, 1], [0, 1, 2, 3], [[0, 4]]),
        ([0, 1, 0], [0, 1, 2], [[1, 2]]),
        ([0, 0, 1], [0, 1, 2], [[2, 3]]),
        ([1, 1], [5, 7], [[5, 6], [7, 8]]),
    ],
    ids=['empty', 'all_ones', 'single_frame', 'final_frame', 'gapped_frames'],
)
def test_spans_use_sorted_final_frame_ids(
    tmp_path: Path,
    mask: list[int],
    frames: list[int],
    expected: list[list[int]],
) -> None:
    _sidecar_path, payload, _raw_text = _write_sidecar(tmp_path, mask=mask, frames=frames)
    assert payload['inpaint_selected'] == expected


def test_spans_pair_the_raw_mask_with_final_frames(tmp_path: Path) -> None:
    _sidecar_path, payload, _raw_text = _write_sidecar(
        tmp_path,
        mask=[1, 0, 1],
        frames=[5, 6, 7],
        tracknet_frames=[100, 101, 102],
    )
    assert payload['inpaint_selected'] == [[5, 6], [7, 8]]


def test_unsorted_frames_are_sorted_after_mask_pairing(tmp_path: Path) -> None:
    _sidecar_path, payload, _raw_text = _write_sidecar(
        tmp_path,
        mask=[1, 1, 0, 1],
        frames=[7, 5, 6, 10],
    )
    assert payload['inpaint_selected'] == [[5, 6], [7, 8], [10, 11]]


def test_selected_invisible_row_stays_in_raw_spans(tmp_path: Path) -> None:
    _sidecar_path, payload, _raw_text = _write_sidecar(
        tmp_path,
        mask=[0, 1, 0],
        frames=[0, 1, 2],
        visibility=[1, 0, 1],
    )
    assert payload['inpaint_selected'] == [[1, 2]]


@pytest.mark.parametrize(
    ('mask', 'frames'),
    [
        ([1], [0, 1]),
        ([1, 0, 1], [0, 1, 0]),
        ([1, 0], [-1, 0]),
    ],
    ids=['length_mismatch', 'duplicate_frame', 'negative_frame'],
)
def test_invalid_applied_masks_raise(
    tmp_path: Path,
    mask: list[int],
    frames: list[int],
) -> None:
    with pytest.raises(ValueError):
        _write_sidecar(tmp_path, mask=mask, frames=frames)


def test_disabled_status_allows_empty_mask_with_frames(tmp_path: Path) -> None:
    _sidecar_path, payload, raw_text = _write_sidecar(
        tmp_path,
        mask=[],
        frames=[0, 1, 2],
        inpaintnet=None,
    )
    assert payload['inpaint_status'] == 'disabled'
    assert payload['inpaint_selected'] == []
    assert payload['inpaintnet_ckpt'] is None
    assert payload['n_rows'] == 3
    # The empty span list stays inline, under the four-space header indent.
    assert raw_text.startswith('{\n    "schema": "inpaint_fill_mask/1",\n')
    assert raw_text.endswith('    "inpaint_selected": []\n}\n')


@pytest.mark.parametrize(
    ('eval_mode', 'expected_stride'),
    [('nonoverlap', 5), ('weight', 1), ('average', 1)],
)
def test_stride_follows_mode_and_checkpoint_sequence_length(
    tmp_path: Path,
    eval_mode: str,
    expected_stride: int,
) -> None:
    sidecar_path, payload, _raw_text = _write_sidecar(
        tmp_path / eval_mode,
        mask=[0],
        frames=[0],
        eval_mode=eval_mode,
        tracknet_seq_len=5,
    )
    assert payload['stride'] == expected_stride
    assert sidecar_path.name == f'1_stride{expected_stride}_inpaint_mask.json.gz'


def test_applied_header_and_format_are_exact_except_runtime_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FixedDateTime:
        @classmethod
        def now(cls, _timezone: timezone) -> datetime:
            return datetime(2026, 7, 22, 11, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(WRITER_MODULE, 'datetime', FixedDateTime)
    sidecar_path, payload, raw_text = _write_sidecar(
        tmp_path,
        mask=[1, 1, 0, 1, 1],
        frames=[5, 6, 8, 11, 12],
        tracknet_frames=[50, 51, 52, 53, 54],
        video_name='input_video.mp4',
    )
    expected = {
        'schema': 'inpaint_fill_mask/1',
        'index_space': 'frame',
        'inpaint_status': 'applied',
        'n_rows': 5,
        'eval_mode': 'weight',
        'stride': 1,
        'th_h_px': 14.4,
        'tracknet_ckpt': 'TrackNet_best.pt',
        'inpaintnet_ckpt': 'InpaintNet_best.pt',
        'input_video': 'input_video.mp4',
        'extracted_utc': '2026-07-22T11:00:00Z',
        'inpaint_selected': [[5, 7], [11, 13]],
    }
    assert payload == expected
    assert sidecar_path.name == 'input_video_stride1_inpaint_mask.json.gz'
    # Layout is part of the contract: four-space header indent, one span
    # per line, nothing minified and nothing double-indented.
    assert raw_text.startswith('{\n    "schema": "inpaint_fill_mask/1",\n')
    assert '\n    "n_rows": 5,\n' in raw_text
    assert '    "inpaint_selected": [\n        [5, 7],\n        [11, 13]\n    ]\n}\n' in raw_text
    assert '            5' not in raw_text


def test_input_video_keeps_extension(tmp_path: Path) -> None:
    _sidecar_path, payload, _raw_text = _write_sidecar(
        tmp_path,
        mask=[0],
        frames=[0],
        video_name='clip.webm',
    )
    assert payload['input_video'] == 'clip.webm'


def test_writer_uses_canonical_identity_and_provenance_for_proxy_input(tmp_path: Path) -> None:
    canonical_dir = tmp_path / 'canonical'
    proxy_dir = tmp_path / 'proxy'
    output_dir = tmp_path / 'output'
    canonical_dir.mkdir()
    proxy_dir.mkdir()
    output_dir.mkdir()
    canonical = canonical_dir / 'fixture.webm'
    proxy = proxy_dir / 'fixture.avi'
    canonical.touch()
    proxy.touch()
    (canonical_dir / 'sources.toml').write_text(
        '''dataset = "fixture-dataset"

[videos."fixture.webm"]
video_id = "fixture-id"
title = "Fixture title"
url = "https://example.test/fixture"
fps = 30
''',
        encoding='utf-8',
    )

    write_inpaint_metadata(
        output_dir / 'fixture_ball.csv',
        tracknet_pred_dict={'Frame': [0], 'Inpaint_Mask': []},
        pred_dict={'Frame': [0], 'Visibility': [1]},
        video_file=proxy,
        eval_mode='nonoverlap',
        tracknet_seq_len=8,
        h=288.0,
        inpaintnet=None,
        input_video_identity=canonical,
    )

    with gzip.open(
        output_dir / 'fixture_stride8_inpaint_mask.json.gz',
        'rt',
        encoding='utf-8',
    ) as sidecar_file:
        payload = json.load(sidecar_file)
    assert payload['input_video'] == 'fixture.webm'
    assert payload['dataset'] == 'fixture-dataset'
    assert payload['video_id'] == 'fixture-id'


def test_writer_does_not_mutate_prediction_inputs(tmp_path: Path) -> None:
    tracknet_pred_dict = {'Frame': [100, 101, 102], 'Inpaint_Mask': [1, 0, 1]}
    pred_dict = {'Frame': [2, 0, 1], 'Visibility': [1, 0, 1]}
    original_tracknet = copy.deepcopy(tracknet_pred_dict)
    original_pred = copy.deepcopy(pred_dict)

    video_dir = tmp_path / 'videos'
    output_dir = tmp_path / 'output'
    video_dir.mkdir()
    output_dir.mkdir()
    write_inpaint_metadata(
        output_dir / 'output.csv',
        tracknet_pred_dict=tracknet_pred_dict,
        pred_dict=pred_dict,
        video_file=video_dir / '1.mp4',
        eval_mode='weight',
        tracknet_seq_len=8,
        h=288.0,
        inpaintnet=object(),
    )

    assert tracknet_pred_dict == original_tracknet
    assert pred_dict == original_pred


def test_writer_validation_happens_before_csv_write(tmp_path: Path) -> None:
    output_dir = tmp_path / 'output'
    video_dir = tmp_path / 'videos'
    output_dir.mkdir()
    video_dir.mkdir()
    csv_path = output_dir / '1_ball.csv'

    with pytest.raises(ValueError):
        write_inpaint_metadata(
            csv_path,
            tracknet_pred_dict={'Frame': [0], 'Inpaint_Mask': [1, 0]},
            pred_dict={'Frame': [0]},
            video_file=video_dir / '1.mp4',
            eval_mode='weight',
            tracknet_seq_len=8,
            h=288.0,
            inpaintnet=object(),
        )

    assert not csv_path.exists()


def _manifest_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    video_dir = tmp_path / 'video_source'
    output_dir = tmp_path / 'sidecar_output'
    working_dir = tmp_path / 'working_directory'
    video_dir.mkdir()
    output_dir.mkdir()
    working_dir.mkdir()
    monkeypatch.chdir(working_dir)
    return video_dir / '1.mp4', output_dir / '1_ball.csv', working_dir


def _write_manifest(video_path: Path, text: str) -> None:
    video_path.parent.joinpath('sources.toml').write_text(text, encoding='utf-8')


def _write_manifest_case(video_path: Path, csv_path: Path) -> dict[str, object]:
    write_inpaint_metadata(
        csv_path,
        tracknet_pred_dict={'Frame': [0], 'Inpaint_Mask': [0]},
        pred_dict={'Frame': [0]},
        video_file=video_path,
        eval_mode='weight',
        tracknet_seq_len=8,
        h=288.0,
        inpaintnet=None,
    )
    sidecar_path = csv_path.parent / '1_stride1_inpaint_mask.json.gz'
    with gzip.open(sidecar_path, 'rt', encoding='utf-8') as sidecar_file:
        return json.load(sidecar_file)


def test_manifest_provenance_comes_from_video_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video_path, csv_path, working_dir = _manifest_paths(tmp_path, monkeypatch)
    _write_manifest(
        video_path,
        '''dataset = "shuttleset"\n\n[videos."1.mp4"]\nvideo_id = 1\ntitle = "Kento_MOMOTA"\nurl = "https://example.test/video"\nfps = 25.0\n''',
    )
    working_dir.joinpath('sources.toml').write_text(
        '''dataset = "Wrong working-directory dataset"\n\n[videos."1.mp4"]\nvideo_id = 999\n''',
        encoding='utf-8',
    )

    payload = _write_manifest_case(video_path, csv_path)
    assert {
        key: payload[key]
        for key in ('dataset', 'video_id', 'title', 'url', 'fps')
    } == {
        'dataset': 'shuttleset',
        'video_id': 1,
        'title': 'Kento_MOMOTA',
        'url': 'https://example.test/video',
        'fps': 25.0,
    }


def test_manifest_fields_are_omitted_when_manifest_is_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video_path, csv_path, working_dir = _manifest_paths(tmp_path, monkeypatch)
    working_dir.joinpath('sources.toml').write_text(
        '''dataset = "Wrong working-directory dataset"\n\n[videos."1.mp4"]\nvideo_id = 999\n''',
        encoding='utf-8',
    )
    payload = _write_manifest_case(video_path, csv_path)
    assert not {'dataset', 'video_id', 'title', 'url', 'fps'} & payload.keys()


def test_manifest_fields_are_omitted_for_unlisted_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video_path, csv_path, _working_dir = _manifest_paths(tmp_path, monkeypatch)
    _write_manifest(
        video_path,
        '''dataset = "shuttleset"\n\n[videos."2.mp4"]\nvideo_id = 2\ntitle = "Other"\nurl = "https://example.test/other"\nfps = 25.0\n''',
    )
    payload = _write_manifest_case(video_path, csv_path)
    assert not {'dataset', 'video_id', 'title', 'url', 'fps'} & payload.keys()


def test_manifest_partial_entry_copies_only_present_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video_path, csv_path, _working_dir = _manifest_paths(tmp_path, monkeypatch)
    _write_manifest(
        video_path,
        '''dataset = "shuttleset"\n\n[videos."1.mp4"]\ntitle = "Partial"\n''',
    )
    payload = _write_manifest_case(video_path, csv_path)
    assert payload['dataset'] == 'shuttleset'
    assert payload['title'] == 'Partial'
    assert not {'video_id', 'url', 'fps'} & payload.keys()


def test_scraped_manifest_extra_eligibility_is_tolerated_by_both_readers(tmp_path: Path) -> None:
    video_path = tmp_path / 'clip.mp4'
    video_path.write_bytes(b'video')
    video_path.parent.joinpath('sources.toml').write_text(
        '''dataset = "scraped"\n\n[videos."clip.mp4"]\nvideo_id = "clip"\ntitle = "A title"\nurl = "https://example.test/clip"\ncommentary_eligible = true\n''',
        encoding='utf-8',
    )
    expected = {
        'dataset': 'scraped',
        'video_id': 'clip',
        'title': 'A title',
        'url': 'https://example.test/clip',
    }

    assert WRITER_MODULE._read_source_provenance(str(video_path)) == expected

    video_path.parent.joinpath('sources.toml').write_text(
        'dataset = "scraped"\n\n[videos]\n',
        encoding='utf-8',
    )
    assert WRITER_MODULE._read_source_provenance(str(video_path)) == {}


def test_malformed_manifest_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video_path, csv_path, _working_dir = _manifest_paths(tmp_path, monkeypatch)
    _write_manifest(video_path, '[videos."1.mp4"\n')
    with pytest.raises(tomllib.TOMLDecodeError):
        _write_manifest_case(video_path, csv_path)


def test_manifest_permission_error_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video_path, csv_path, _working_dir = _manifest_paths(tmp_path, monkeypatch)
    manifest_path = video_path.parent / 'sources.toml'
    manifest_path.write_text('dataset = "shuttleset"\n', encoding='utf-8')
    real_open = builtins.open

    def raise_for_manifest(path, *args, **kwargs):
        if os.fspath(path) == os.fspath(manifest_path):
            raise PermissionError(os.fspath(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, 'open', raise_for_manifest)
    with pytest.raises(PermissionError):
        _write_manifest_case(video_path, csv_path)


@pytest.mark.parametrize(
    'manifest_text',
    [
        'videos = []\n',
        'videos = { "1.mp4" = "not a table" }\n',
    ],
    ids=['videos_not_table', 'entry_not_table'],
)
def test_manifest_tables_are_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest_text: str,
) -> None:
    video_path, csv_path, _working_dir = _manifest_paths(tmp_path, monkeypatch)
    _write_manifest(video_path, manifest_text)
    with pytest.raises(TypeError):
        _write_manifest_case(video_path, csv_path)


@pytest.mark.parametrize(
    ('field', 'value'),
    [('fps', '"25"'), ('url', '25')],
)
def test_manifest_field_types_are_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    video_path, csv_path, _working_dir = _manifest_paths(tmp_path, monkeypatch)
    _write_manifest(
        video_path,
        f'dataset = "shuttleset"\n\n[videos."1.mp4"]\n{field} = {value}\n',
    )
    with pytest.raises(TypeError):
        _write_manifest_case(video_path, csv_path)


def test_manifest_accepts_string_video_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video_path, csv_path, _working_dir = _manifest_paths(tmp_path, monkeypatch)
    _write_manifest(video_path, 'dataset = "shuttleset"\n\n[videos."1.mp4"]\nvideo_id = "001"\n')
    payload = _write_manifest_case(video_path, csv_path)
    assert payload['video_id'] == '001'


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding='utf-8'), filename=os.fspath(path))


def _predict_video_calls(tree: ast.Module) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'predict_video'
    ]


def _keyword(call: ast.Call, name: str) -> ast.expr:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    raise AssertionError(f'{name} was not passed to predict_video')


def _is_basename_call(node: ast.expr, argument_name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'basename'
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Attribute)
        and isinstance(node.args[0].value, ast.Name)
        and node.args[0].value.id == 'args'
        and node.args[0].attr == argument_name
    )


def _assignment_value(tree: ast.Module, name: str) -> ast.expr:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node.value
    raise AssertionError(f'{name} was not resolved before the batch loop')


def _is_none_constant(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def test_all_entry_points_propagate_checkpoint_basenames() -> None:
    standalone_tree = _parse(TRACKNET_DIR / 'predict.py')
    standalone_call = _predict_video_calls(standalone_tree)[0]
    assert _is_basename_call(_keyword(standalone_call, 'tracknet_ckpt'), 'tracknet_file')
    standalone_inpaint = _keyword(standalone_call, 'inpaintnet_ckpt')
    assert isinstance(standalone_inpaint, ast.IfExp)
    assert _is_basename_call(standalone_inpaint.body, 'inpaintnet_file')
    assert _is_none_constant(standalone_inpaint.orelse)

    for batch_path in (TRACKNET_DIR / 'batch_predict.py',):
        batch_tree = _parse(batch_path)
        batch_call = _predict_video_calls(batch_tree)[0]
        # The call site must pass the resolved variables THEMSELVES, so the
        # keyword's Name.id has to match the assignment being checked below.
        tracknet_arg = _keyword(batch_call, 'tracknet_ckpt')
        inpaint_arg = _keyword(batch_call, 'inpaintnet_ckpt')
        assert isinstance(tracknet_arg, ast.Name) and tracknet_arg.id == 'tracknet_ckpt'
        assert isinstance(inpaint_arg, ast.Name) and inpaint_arg.id == 'inpaintnet_ckpt'
        assert _is_basename_call(_assignment_value(batch_tree, 'tracknet_ckpt'), 'tracknet_file')
        inpaint_assignment = _assignment_value(batch_tree, 'inpaintnet_ckpt')
        assert isinstance(inpaint_assignment, ast.IfExp)
        assert _is_basename_call(
            inpaint_assignment.body,
            'inpaintnet_file',
        )
        assert _is_none_constant(inpaint_assignment.orelse)


def _direct_call_statement_indexes(function: ast.FunctionDef, name: str) -> list[int]:
    """Indexes into ``function.body`` of bare ``name(...)`` statements."""
    return [
        index
        for index, statement in enumerate(function.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == name
    ]


def test_writer_call_precedes_csv_write_in_predict_video() -> None:
    # A live ordering test needs a GPU inference run, so the ordering
    # constraint (one unconditional sidecar write immediately before the
    # CSV write, both in predict_video's own statement list, so a writer
    # exception must abort the CSV) is asserted structurally instead.
    for predict_path in (TRACKNET_DIR / 'predict.py',):
        tree = _parse(predict_path)
        predict_video = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == 'predict_video'
        )
        # Direct-body statements only: a call nested in try/if/loop would
        # not appear here, so these asserts also pin unconditionality.
        writer_indexes = _direct_call_statement_indexes(predict_video, 'write_inpaint_metadata')
        csv_indexes = _direct_call_statement_indexes(predict_video, 'write_pred_csv')
        assert writer_indexes and len(writer_indexes) == 1, (
            f'{predict_path} needs exactly one unconditional write_inpaint_metadata statement'
        )
        assert csv_indexes and len(csv_indexes) == 1, (
            f'{predict_path} needs exactly one unconditional write_pred_csv statement'
        )
        assert writer_indexes[0] + 1 == csv_indexes[0], (
            f'{predict_path}: the sidecar write must sit immediately before the CSV write'
        )
