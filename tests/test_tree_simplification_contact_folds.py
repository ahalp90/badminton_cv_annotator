"""Matched contact folds preserve validation and downstream group membership."""

from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

import joblib
import numpy as np
import pytest

from annotator.training.contact import ContactFitConfig
from annotator.training.sequences import DEFAULT_RALLY_BOUNDARY_POLICY
from experiments.annotator import development as experiment
from experiments.annotator.tree_simplification import contact_folds, pools

VALIDATION = ('sset_18', 'sset_22', 'sset_24', 'sset_25', 'sset_30', 'sset_31', 'sset_39', 'sset_40')


def development() -> experiment.Development:
    groups = {}
    for fold, identities in contact_folds.CONTACT_FOLDS.items():
        groups.update({identity: fold[0] for identity in identities})
    groups.update({identity: 'V' for identity in VALIDATION})
    return experiment.Development(groups, dict.fromkeys(groups, 30.0), {
        identity: int(identity.removeprefix('sset_')) for identity in groups
    })


def prepared_fixture(tmp_path: Path) -> SimpleNamespace:
    prepared, original, reference = tmp_path / 'prepared', tmp_path / 'original', tmp_path / 'contact-run'
    for directory in (prepared, original, reference):
        directory.mkdir()
    groups = development().group_by_video
    settings = {
        'contact': asdict(ContactFitConfig(device='cuda')),
        'preprocessing': {'fixture': True}, 'contact_prediction': {},
    }
    experiment.write_json(prepared / 'preparation.json.gz', {
        'inputs': {'groups': groups, 'preprocessing': settings['preprocessing'],
                   'paths': {'shots_master': str(tmp_path / 'shots.csv')}},
    })
    experiment.write_json(reference / 'settings.json.gz', {'prepared': str(prepared), 'settings': settings})
    (reference / 'contacts').mkdir(parents=True)
    for name in ('held_A', 'held_B', 'held_C', 'held_D', 'v_32'):
        joblib.dump({'original': name}, reference / 'contacts' / f'{name}.joblib')
    manifest = {
        'schema': pools.SCHEMA, 'prepared': str(prepared), 'contact_run': str(reference), 'settings': settings,
        'max_edits': 3, 'deletion_candidates': 6,
        'boundary_policy': asdict(DEFAULT_RALLY_BOUNDARY_POLICY),
    }
    experiment.write_json(original / 'pools.json.gz', manifest)
    (prepared / 'videos').mkdir()
    (original / 'videos').mkdir()
    for identity, group in groups.items():
        split = 'validation' if group == 'V' else 'train'
        video = SimpleNamespace(identity=identity, split=split, fps=30.0,
                                rallies=(SimpleNamespace(frames=(30, 10)),))
        joblib.dump(SimpleNamespace(video=video, group=group, features=SimpleNamespace(rows=np.zeros(2))),
                    prepared / 'videos' / f'{identity}.joblib')
        metadata = {'video': identity, 'group': group, 'split': split}
        experiment.write_json(prepared / 'videos' / f'{identity}.json.gz', metadata)
        joblib.dump({'identity': identity}, original / 'videos' / f'{identity}.joblib')
        experiment.write_json(original / 'videos' / f'{identity}.json.gz', metadata)
    return SimpleNamespace(prepared=prepared, original=original, reference=reference, manifest=manifest)


def test_folds_partition_only_original_training_and_retain_downstream_groups() -> None:
    context = development()
    assignments = contact_folds.fold_assignments(context)
    assert len(assignments) == 32
    assert not assignments.keys() & set(VALIDATION)
    assert set(assignments.values()) == set(contact_folds.CONTACT_FOLDS)
    for identity, fold in assignments.items():
        assert context.group_by_video[identity] == fold[0]
    changed = replace(context, group_by_video={**context.group_by_video, 'sset_18': 'A'})
    with pytest.raises(ValueError, match='original 32 training and 8 validation'):
        contact_folds.fold_assignments(changed)
    changed = replace(context, group_by_video={**context.group_by_video, 'sset_01': 'B'})
    with pytest.raises(ValueError, match='disjoint subdivisions'):
        contact_folds.fold_assignments(changed)


def test_orchestration_fits_28_only_and_reuses_completed_folds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = prepared_fixture(tmp_path)
    context = development()
    monkeypatch.setattr(contact_folds, 'load_original_videos', lambda _path: ())
    monkeypatch.setattr(contact_folds, 'development_for', lambda _videos: context)
    fits, prepared = [], []

    def fit(videos: Any, config: Any, *, fit_video_order: Any) -> Any:
        identities = tuple(video.identity for video in videos)
        assert len(identities) == 28 and not set(identities) & set(VALIDATION)
        assert config == ContactFitConfig(**fixture.manifest['settings']['contact'])
        assert identities == tuple(identity for identity in context.videos(('A', 'B', 'C', 'D'))
                                   if identity in identities)
        assert fit_video_order == context.id_order(identities)
        assert all(video.contact_frames.tolist() == [10, 30] for video in videos)
        fits.append(identities)
        return SimpleNamespace(model={'fitted': len(fits)}, selection=SimpleNamespace(labels=np.asarray([0, 1])))

    def prepare(*args: Any, **kwargs: Any) -> None:
        assert args == (fixture.prepared, fixture.reference, tmp_path / 'output' / 'pools')
        assert kwargs['validation_source'] == fixture.original
        assert set(kwargs['contact_paths']) == set(contact_folds.fold_assignments(context))
        for identity, path in kwargs['contact_paths'].items():
            assert path.name == f'held_{contact_folds.fold_assignments(context)[identity]}.joblib'
            assert path.is_file()
        prepared.append(kwargs)

    monkeypatch.setattr(contact_folds, 'fit_contact_model', fit)
    monkeypatch.setattr(contact_folds, 'prepare_pools', prepare)
    for _attempt in range(2):
        contact_folds.prepare_contact_folds(fixture.prepared, fixture.reference, fixture.original, tmp_path / 'output')
    assert len(fits) == 8 and len(set(fits)) == 8 and len(prepared) == 2
    training = set(context.videos(('A', 'B', 'C', 'D')))
    for fold, identities in zip(contact_folds.CONTACT_FOLDS, fits, strict=True):
        assert training - set(identities) == set(contact_folds.CONTACT_FOLDS[fold])
        receipt = experiment.read_json(tmp_path / 'output' / 'contacts' / f'held_{fold}.json.gz')
        assert receipt['rows'] == 2 and receipt['positive_rows'] == 1
    assert joblib.load(fixture.reference / 'contacts' / 'v_32.joblib') == {'original': 'v_32'}
    assert not (tmp_path / 'output' / 'contacts' / 'v_32.joblib').exists()
    manifest_path = tmp_path / 'output' / 'contact-folds.json.gz'
    manifest = experiment.read_json(manifest_path)
    manifest['folds']['A1'].reverse()
    experiment.write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match='Contact folds or settings changed'):
        contact_folds.prepare_contact_folds(fixture.prepared, fixture.reference, fixture.original, tmp_path / 'output')
    assert len(fits) == 8


def test_pool_override_copies_validation_bytes_and_binds_resume_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = prepared_fixture(tmp_path)
    assignments = contact_folds.fold_assignments(development())
    contact_paths = {}
    for identity, fold in assignments.items():
        path = tmp_path / 'contacts' / f'{fold}.joblib'
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'contact-model')
        contact_paths[identity] = path
    output = tmp_path / 'output'
    submitted = []

    class Executor:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def submit(self, _function: Any, job: pools.PoolJob) -> SimpleNamespace:
            submitted.append(job)
            return SimpleNamespace(result=lambda: {'video': job.video})

    monkeypatch.setattr(pools, 'ProcessPoolExecutor', Executor)
    monkeypatch.setattr(pools, 'as_completed', lambda futures: futures)
    options: dict[str, Any] = {'contact_paths': contact_paths, 'contact_folds': contact_folds.CONTACT_FOLDS,
               'validation_source': fixture.original}
    pools.prepare_pools(fixture.prepared, fixture.reference, output, **options)
    assert len(submitted) == 32
    assert all(job.split == 'train' and job.group == assignments[job.video][0] for job in submitted)
    assert all(job.contact_path == contact_paths[job.video] for job in submitted)
    for identity in VALIDATION:
        for suffix in ('.joblib', '.json.gz'):
            assert (output / 'videos' / f'{identity}{suffix}').read_bytes() == (
                fixture.original / 'videos' / f'{identity}{suffix}').read_bytes()
    manifest = experiment.read_json(output / 'pools.json.gz')
    assert manifest['settings'] == fixture.manifest['settings']
    submitted.clear()
    default_output = tmp_path / 'default-pools'
    pools.prepare_pools(fixture.prepared, fixture.reference, default_output)
    assert experiment.read_json(default_output / 'pools.json.gz') == fixture.manifest
    assert len(submitted) == 40
    for job in submitted:
        name = 'v_32' if job.split == 'validation' else f'held_{job.group}'
        assert job.contact_path == fixture.reference / 'contacts' / f'{name}.joblib'
    swapped = {**contact_paths, 'sset_01': contact_paths['sset_05']}
    with pytest.raises(ValueError, match='Pool settings changed'):
        pools.prepare_pools(fixture.prepared, fixture.reference, output, **{**options, 'contact_paths': swapped})
    changed = dict(contact_folds.CONTACT_FOLDS)
    changed['A1'] = tuple(reversed(changed['A1']))
    with pytest.raises(ValueError, match='Pool settings changed'):
        pools.prepare_pools(fixture.prepared, fixture.reference, output, **{**options, 'contact_folds': changed})
    with pytest.raises(ValueError, match='every training video exactly once'):
        pools.prepare_pools(fixture.prepared, fixture.reference, output,
                            **{**options, 'contact_paths': {'sset_01': contact_paths['sset_01']}})
    original_manifest = {**fixture.manifest, 'max_edits': 2}
    experiment.write_json(fixture.original / 'pools.json.gz', original_manifest)
    with pytest.raises(ValueError, match='different preparation or proposal settings'):
        pools.prepare_pools(fixture.prepared, fixture.reference, output, **options)
