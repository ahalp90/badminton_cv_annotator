"""YAML-boundary pin for run_tracker's manifest write.

Hyp's ``adaptive_focal`` and ``augmentation`` defaults are frozendicts: shared
class-level defaults, made read-only so no instance can mutate them. PyYAML's
safe_dump representers are exact-type and reject dict subclasses (frozendict
included), so resolve_run_paths casts those two fields back to plain dict before
track_run writes the manifest. This pins that boundary end to end: a config
payload shaped the way resolve_run_paths builds it survives track_run's real
YAML write and reads back as plain dicts.
"""

from __future__ import annotations

import pytest
import yaml

from bst_x_train import Hyp
from run_tracker import track_run


def test_manifest_write_round_trips_hyp_mapping_fields_to_plain_dict(tmp_path):
    hyp = Hyp()

    # Mirror resolve_run_paths' manifest-boundary cast: the two mapping fields
    # go back to plain dict so safe_dump accepts them. An identity no-op while
    # the defaults are plain dicts, load-bearing once they are frozendicts.
    config_payload = dict(hyp._asdict())
    config_payload['augmentation'] = dict(hyp.augmentation)
    if hyp.adaptive_focal is not None:
        config_payload['adaptive_focal'] = dict(hyp.adaptive_focal)

    # track_run owns the real safe_dump (via _write_manifest); tmp_path is no git
    # repo, so the SHA/dirty probes just return None. This is the production
    # serialiser, not a stand-in.
    run_dir, _ = track_run(
        config=config_payload,
        run_id='run_yaml_boundary_pin',
        experiments_dir=tmp_path,
        project_root=tmp_path,
    )

    loaded_cfg = yaml.safe_load((run_dir / 'manifest.yaml').read_text())['config']

    assert type(loaded_cfg['augmentation']) is dict
    assert loaded_cfg['augmentation'] == dict(hyp.augmentation)
    assert type(loaded_cfg['adaptive_focal']) is dict
    assert loaded_cfg['adaptive_focal'] == dict(hyp.adaptive_focal)


def test_safe_dump_rejects_uncast_frozendict():
    """safe_dump rejects dict subclasses like frozendict, which is why the manifest
    writer casts to plain dict first. If a PyYAML upgrade ever accepts them this test
    fails, flagging that the cast in resolve_run_paths is no longer needed."""
    hyp = Hyp()
    with pytest.raises(yaml.representer.RepresenterError):
        yaml.safe_dump({'augmentation': hyp.augmentation})
