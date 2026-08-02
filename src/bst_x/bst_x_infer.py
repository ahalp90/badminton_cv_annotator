# Portions of this file are derived from BST (Badminton Stroke-type Transformer)
# by Jing-Yuan Chang, Copyright (c) 2025 Jing-Yuan Chang, used under the MIT
# Licence. See src/bst_x/THIRD_PARTY_NOTICES.md. This project is otherwise
# licensed LGPL-3.0-or-later.

# BST inference for ShuttleSet. Two faces:
#
#   1. Library: infer() + Task — load a checkpoint and predict, for a live
#      single-clip backend (e.g. a Gradio GUI).
#   2. CLI --fe mode: post-hoc batch dump of per-stroke logits + top-k for an
#      already-trained run, writing the same npz schema bst_x_train emits at
#      end-of-serial. Folds in the retired eval_dump_predictions.py; lets the
#      FE-shape converter / calibration run against a run without retraining.
#
# Run from the repo root with both package roots on PYTHONPATH:
#   PYTHONPATH=src:src/bst_x \
#       python -m bst_x_infer --fe \
#           --run-dir .../experiments/bst_x/shuttleset/run_<id> --serial 5
#   The dump lands in <run-dir>/inference_runs/<timestamp>/ (npz +
#   inference_manifest.yaml); pass --fe-output-dir to redirect it elsewhere.
#
# See bst_x_train.py for detailed PyTorch/TF comparison comments.

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from preparing_data.shuttleset_dataset import Dataset_npy_collated
from classifier_shared.taxonomy import Taxonomy, taxonomy_lookup
from pipeline.config import collation_id_from_manifest, derive_npy_collated_dir_basename
from pipeline.data_access import load_repo_dotenv, resolve_collated_data_root
from bst_x_common import (
    write_prediction_npz,
    build_bst_x_network,
    dump_topk_predictions,
    flatten_pose_features,
    to_device,
)


@torch.no_grad()
def infer(
    model: nn.Module,
    loader,
    device
):
    model.eval()
    pred_ls = []

    for (human_pose, pos, shuttle), video_len, labels in loader:
        human_pose, shuttle, pos, video_len = to_device(
            device, human_pose, shuttle, pos, video_len,
        )

        human_pose = flatten_pose_features(human_pose)
        logits = model(human_pose, shuttle, pos=pos, video_len=video_len)

        pred = torch.argmax(logits, dim=1).cpu()

        pred_ls.append(pred)

    return torch.cat(pred_ls)


class Task:
    """Live single-clip inference helper (Gradio-style backend).

    Build the head at ``taxonomy.n_classes`` and decode predictions against
    ``taxonomy.classes``. Labels on disk are already in that index space (no
    runtime remap), so the head dim is just the taxonomy size.
    """

    def __init__(self, n_joints=17, pose_style='JnB_bone') -> None:
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
        self.n_joints = n_joints
        # pose_style lives here (not on prepare_loader) so get_network_architecture
        # can build without a prior loader step; kills the old call-order trap.
        self.pose_style = pose_style

    def prepare_loader(
        self,
        npy_collated_dir: Path,
        batch_size=128,
    ):
        dataset = Dataset_npy_collated(npy_collated_dir, 'test', self.pose_style)

        self.infer_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size
        )

    def get_network_architecture(
        self,
        *,
        taxonomy: Taxonomy,
        model_name: str = 'BST_X',
        seq_len: int = 100,
    ):
        """Build the inference model at the taxonomy head dim.

        The weights being loaded were trained against ``taxonomy.classes``;
        a mismatch between the weight file's head dim and
        ``taxonomy.n_classes`` raises a clear shape error inside
        ``load_state_dict``. For a legacy run, pass the taxonomy the run
        recorded (``taxonomy_lookup(manifest['config']['taxonomy'])``).

        :param taxonomy: the taxonomy the weights were trained under.
        """
        self.taxonomy = taxonomy
        self.net, _n_bones = build_bst_x_network(
            model_name,
            n_joints=self.n_joints,
            pose_style=self.pose_style,
            n_classes=taxonomy.n_classes,
            seq_len=seq_len,
            device=self.device,
        )

    def load_weight(self, weight_path: Path):
        self.net.load_state_dict(torch.load(str(weight_path), map_location=self.device, weights_only=True))

    def infer(self):
        return infer(self.net, self.infer_loader, self.device)


# ==========================================================================
# --fe batch dump: per-stroke logits + top-k npz for an existing run
# ==========================================================================

def _resolve_collated_dir(
    manifest: dict, config: dict, collated_data_root: Path | None, run_dir: Path,
) -> Path:
    """Resolve the collated dir the run trained on, for a post-hoc dump.

    Prefers the recorded ``extra.data_provenance.npy_collated_dir`` (carries the
    historical basename verbatim, including pre-split-fold names like
    ``npy_wipe_drop``); falls back to deriving it from the recorded config. The
    root comes from ``resolve_collated_data_root`` (``--collated-data-root``
    override, then ``BST_X_COLLATED_DATA_ROOT``, then the in-repo
    ``preparing_data/`` convention).
    """
    extra = manifest.get('extra') or {}
    provenance = extra.get('data_provenance') or {}
    recorded_dir = provenance.get('npy_collated_dir')
    collation_id = collation_id_from_manifest(manifest)
    if not recorded_dir and not collation_id:
        # Neither a recorded dir nor a collation tag: deriving would format the
        # None into an ``npy_..._None`` path that dies later naming that path,
        # not the real cause. Fail here on the actual problem.
        raise ValueError(
            f'{run_dir}/manifest.yaml records neither npy_collated_dir nor a '
            f'collation id; cannot resolve the collated dir.'
        )
    basename = recorded_dir or derive_npy_collated_dir_basename(
        seq_len=config['seq_len'],
        split_column=config['split_column'],
        collation_id=collation_id,
    )
    root = resolve_collated_data_root(collated_data_root)
    return root / f"ShuttleSet_data_{config['taxonomy']}" / basename


def dump_run_predictions(
    *,
    run_dir: Path,
    serial: int,
    fe_output_dir: Path | None = None,
    splits: tuple[str, ...] = ('val', 'test'),
    collated_data_root: Path | None = None,
    model_name: str = 'BST_X',
    n_joints: int = 17,
    batch_size: int = 128,
) -> Path:
    """Dump per-split prediction npz for an already-trained run.

    Each dump lands in its own timestamped dir so post-hoc inference never
    clobbers the run's training-time ``predictions/`` and re-dumps don't
    collide: ``<base>/inference_runs/<YYYYmmdd_HHMMSS>/``. ``base`` defaults to
    ``run_dir`` (co-located with the run), or ``fe_output_dir/<run_id>`` when an
    override is passed. A small ``inference_manifest.yaml`` records the source
    weights / serial / splits / time alongside the npz.

    Same npz schema as ``bst_x_train``'s end-of-serial dump (logits, y_true,
    y_pred_top1, topk_idx, clip_stems, class_list, run_id, serial_no,
    taxonomy_name). New-schema runs only: labels.npy is in active class space,
    so there's no remap.

    :return: the timestamped output dir holding this dump's npz + manifest.
    """
    manifest = yaml.safe_load((run_dir / 'manifest.yaml').read_text())
    config = manifest['config']
    taxonomy = taxonomy_lookup(config['taxonomy'])

    target = next(
        (s for s in manifest.get('serials', []) if s['serial_no'] == serial), None
    )
    # Raise inside the library so an importer (api/notebook) gets an exception,
    # not a process exit; __main__ maps these back to sys.exit for the CLI.
    if not target:
        raise ValueError(f'serial {serial} not found in {run_dir}/manifest.yaml')
    weights_path = run_dir / 'weights' / Path(target['weights_path']).name
    if not weights_path.is_file():
        raise FileNotFoundError(f'weights file missing: {weights_path}')

    collated_dir = _resolve_collated_dir(manifest, config, collated_data_root, run_dir)
    if not collated_dir.is_dir():
        raise FileNotFoundError(f'collated dir missing: {collated_dir}')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net, _n_bones = build_bst_x_network(
        model_name,
        n_joints=n_joints,
        pose_style=config['pose_style'],
        n_classes=taxonomy.n_classes,
        seq_len=config['seq_len'],
        device=device,
    )
    net.load_state_dict(
        torch.load(str(weights_path), map_location=device, weights_only=True)
    )

    print(f'run_dir: {run_dir}')
    print(f'weights: {weights_path}')
    print(f'collated_dir: {collated_dir}')
    print(f'taxonomy: {taxonomy.name} ({taxonomy.n_classes} classes)')

    # Own timestamped dir per dump: co-located in the run by default, or under
    # fe_output_dir/<run_id> when overridden. Never the run's training-time
    # predictions/ dir, so a re-dump can't clobber it.
    now = datetime.now()
    base = (fe_output_dir / run_dir.name) if fe_output_dir else run_dir
    out_dir = base / 'inference_runs' / f'{now:%Y%m%d_%H%M%S}'
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for split in splits:
        dataset = Dataset_npy_collated(collated_dir, split, config['pose_style'])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        dump = dump_topk_predictions(net, loader, device, k=5)
        out_path = out_dir / f'{split}_serial_{serial}.npz'
        write_prediction_npz(
            out_path, dump, dataset, taxonomy, run_dir.name, serial,
        )
        written.append(out_path.name)
        print(f'saved: {out_path} ({len(dump["y_true"])} rows)')

    # Small provenance manifest so a dump self-describes when/from-what, beyond
    # what each npz already carries.
    (out_dir / 'inference_manifest.yaml').write_text(yaml.safe_dump({
        'source_run_id': run_dir.name,
        'created_at': now.isoformat(timespec='seconds'),
        'serial_no': serial,
        'splits': list(splits),
        'taxonomy': taxonomy.name,
        'weights_path': str(weights_path),
        'collated_dir': str(collated_dir),
        'npz_files': written,
    }, sort_keys=False))
    print(f'wrote: {out_dir / "inference_manifest.yaml"}')
    return out_dir


if __name__ == '__main__':
    # Load .env so BST_X_COLLATED_DATA_ROOT resolves the same way the collator
    # and bst_x_train do. No-op without .env; shell exports win.
    load_repo_dotenv()

    parser = argparse.ArgumentParser(
        description='BST inference. --fe runs the post-hoc batch dump of '
                    'per-stroke logits + top-k for an existing run.',
    )
    parser.add_argument(
        '--fe', action='store_true',
        help='FE/batch dump mode. Requires --run-dir.',
    )
    parser.add_argument(
        '--fe-output-dir', type=Path, default=None,
        help='Optional override for where the dump lands. Default writes into '
             '<run-dir>/inference_runs/<timestamp>/; with an override, '
             '<fe-output-dir>/<run_id>/inference_runs/<timestamp>/.',
    )
    parser.add_argument(
        '--run-dir', type=Path, default=None,
        help='experiments/bst_x/shuttleset/run_<id>/ whose weights to dump. Required when --fe is set.',
    )
    parser.add_argument('--serial', type=int, default=5,
                        help='Serial number whose weights to evaluate. Default 5: '
                             'the last serial of a standard 5-serial run.')
    parser.add_argument('--splits', default='val,test',
                        help='Comma-separated splits to dump (default: val,test).')
    parser.add_argument(
        '--collated-data-root', type=Path, default=None,
        help='Root holding ShuttleSet_data_<tax>/. Defaults to '
             'BST_X_COLLATED_DATA_ROOT, then the in-repo preparing_data/.',
    )
    parser.add_argument('--model-name', default='BST_X',
                        help='BST variant; defaults to BST_X (the project name for BST_CG_AP). '
                             'Must match the variant the run was trained with.')
    args = parser.parse_args()

    # --fe-output-dir is an optional override that only makes sense in --fe mode.
    if args.fe_output_dir and not args.fe:
        parser.error('--fe-output-dir requires --fe (no implicit dump mode)')
    if not args.fe:
        parser.error(
            'bst_x_infer CLI currently only implements --fe (batch dump) mode. '
            'For live single-clip inference, import infer() / Task instead.'
        )
    if not args.run_dir:
        parser.error('--fe requires --run-dir <experiments/bst_x/shuttleset/run_...>')

    # dump_run_predictions raises inside the library; the CLI turns those into
    # a clean exit-with-message.
    try:
        dump_run_predictions(
            run_dir=args.run_dir.resolve(),
            serial=args.serial,
            fe_output_dir=args.fe_output_dir,
            splits=tuple(s.strip() for s in args.splits.split(',') if s.strip()),
            collated_data_root=args.collated_data_root,
            model_name=args.model_name,
        )
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(str(exc))
