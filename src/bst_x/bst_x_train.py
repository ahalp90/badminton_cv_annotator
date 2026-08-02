# Portions of this file are derived from BST (Badminton Stroke-type Transformer)
# by Jing-Yuan Chang, Copyright (c) 2025 Jing-Yuan Chang, used under the MIT
# Licence. See src/bst_x/THIRD_PARTY_NOTICES.md. This project is otherwise
# licensed LGPL-3.0-or-later.

# BST training script for ShuttleSet.
#
# Run from the repo root with both package roots on PYTHONPATH:
#   PYTHONPATH=src:src/bst_x \
#       python -m bst_x_train

import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # TensorBoard logging
from torcheval.metrics.functional import multiclass_f1_score
from beartype import beartype
from jaxtyping import Bool, Float32, Int64, jaxtyped

from transformers import get_cosine_schedule_with_warmup  # from HuggingFace, not a custom module
from frozendict import frozendict

from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass
from typing import NamedTuple
from contextlib import redirect_stdout
import argparse
import math
import time
from datetime import datetime, timedelta
import sys

from preparing_data.shuttleset_dataset import prepare_npy_collated_loaders, \
                                              pad_class_labels
from preparing_data.augmentations import CoupledFlip, ConstrainedJitter
from result_utils import show_f1_results, plot_confusion_matrix
from pipeline.config import (
    CLIP_WINDOW,
    COCO_N_JOINTS,
    derive_npy_collated_dir_basename,
)
from classifier_shared.taxonomy import Taxonomy, taxonomy_lookup
from pipeline.data_access import load_repo_dotenv, resolve_collated_data_root
from run_tracker import track_run, track_serial
from bst_x_common import (
    Tee,
    write_prediction_npz,
    build_bst_x_network,
    compute_data_provenance,
    dump_topk_predictions,
    flatten_pose_features,
    to_device,
)
from loss.adaptive_focal import (
    AdaptiveFocalLoss,
    accumulate_class_counts,
    per_class_f1_from_counts,
)
from bst_x_reporting import (
    _log_epoch_tb,
    _write_hparams_summary,
    _print_taxonomy_block,
    print_head_tail,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLIPS_CSV = REPO_ROOT / 'notebooks' / 'clips_master.csv'


# ==========================================================================
# Hyperparameters — edit these to change experiment configuration.
# Active LR + aux schedule rationale: docs/architecture_notes/bst_x_overview.md.
# Dated retune history: docs/architecture_notes/historical_bst.md section 3.
# ==========================================================================
# collation_id picks which on-disk collation generation to read (path + manifest
# tag, e.g. 'taxon_pinned_w_preds'); it discriminates re-collations of the same
# taxonomy + split. ablation_id is a separate, nullable training-time tag (augs /
# loss / wiring on a fixed collation): manifest-only, never in the path. See
# pipeline.config.derive_npy_collated_dir_basename for the disentanglement.
class Hyp(NamedTuple):
    n_epochs: int = 80
    batch_size: int = 128
    lr: float = 5e-4
    # AdamW decoupled weight decay. 0.01 is PyTorch's AdamW default; the sweep
    # overrides it per cell (norm/bias/embeddings excluded from decay). See
    # docs/architecture_notes/hp_and_aug_speculations_30_05_2026.md (Q2).
    weight_decay: float = 0.01
    warm_up_step: int = 100
    taxonomy: str = 'une_v1_14'
    seq_len: int = 100
    early_stop_n_epochs: int = 40
    pose_style: str = 'JnB_bone'
    train_partial: float = 1.0
    use_aux_schedule: bool = True
    aux_fade_end_epoch: int = 15
    clips_csv: str = str(DEFAULT_CLIPS_CSV)
    split_column: str = 'split_v2'
    collation_id: str = 'taxon_pinned_w_preds'
    ablation_id: str | None = None
    label_smoothing: float = 0.0  # CDB-F1 cell forces LS=0; LS softens targets so confident-correct samples have p_t < 1.0, contaminating focal's per-sample hardness signal
    # Manual per-class CE weights, renormalised to mean 1.0 in the loss build so
    # overall loss scale stays comparable to uniform CE; None for uniform CE.
    class_weights: dict | None = None
    # Class-F1-driven adaptive focal loss (CDB-F1). Mutually exclusive with
    # class_weights, and forces label_smoothing=0 (LS contaminates focal's
    # hardness estimate). None disables; pass a dict to engage:
    #   adaptive_focal={
    #       'tau': 1.0, 'gamma': 1.0, 'momentum': 0.9,
    #       'warm_up_epochs': 5, 'f1_floor': 0.0,
    #   }
    # Full design + paper-verified equations: docs/architecture_notes/class_f1_focal_design.md.
    # NamedTuple defaults are class-level and shared across every instance; frozendict enforces the read-only contract.
    adaptive_focal: frozendict | None = frozendict({
        # tau=1, gamma=1 is the swept sweet spot (floor-lift on wrist_smash);
        # see class_f1_focal_design.md.
        'tau': 1.0,
        'gamma': 1.0,
        'momentum': 0.9,
        'warm_up_epochs': 5,
        'f1_floor': 0.0,
    })
    # Train-time augmentations. Replaces the inherited (broken) joints-only
    # RandomTranslation_batch. Flip is the literature-norm dataset-doubler;
    # constrained jitter is the corrected, pos+shuttle-only,
    # layered-conditional-bound formulation. Full design + verified code
    # traces in docs/architecture_notes/augmentation_framework.md.
    augmentation: frozendict = frozendict({
        'p_flip':   0.5,
        'p_jitter': 0.3,
        'cap_y':    0.05,
        'cap_x':    0.10,
        'eps':      0.15,
    })


hyp = Hyp()


# ==========================================================================
# Training and evaluation functions
# ==========================================================================

def aux_schedule_factor(epoch: int, fade_end_epoch: int) -> float:
    """Cosine warm-start-to-fade schedule for CG/AP auxiliary modules.

    Factor is 1.0 at epoch 1, 0.5 at mid-fade, and 0.0 at fade_end_epoch.
    Stays pinned at 0.0 for all epochs beyond fade_end_epoch, giving the
    transformer backbone a pure-solo phase to find its own best representation.

    Decoupling fade_end from n_epochs matters when the historical peak F1
    falls well inside the schedule: setting fade_end_epoch near (or before)
    that peak guarantees CG/AP contribution is meaningfully reduced in the
    peak region, so the experiment actually tests the hypothesis rather than
    running a near-baseline with a mild perturbation.

    :param epoch: current epoch, 1-indexed (matches the training loop).
    :param fade_end_epoch: epoch at which factor first reaches 0.0; stays 0 after.
    :return: scalar in [0, 1].
    """
    if epoch >= fade_end_epoch:
        return 0.0
    progress = (epoch - 1) / (fade_end_epoch - 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@jaxtyped(typechecker=beartype)
@dataclass(frozen=True)
class EpochStats:
    """One training epoch's aggregates. ``tp``/``fp``/``fn`` stay on the train
    device; the jitter counters are plain ints over the epoch's clips."""
    train_loss: float
    tp: Int64[Tensor, 'classes']
    fp: Int64[Tensor, 'classes']
    fn: Int64[Tensor, 'classes']
    jitter_n_effective: int
    jitter_n_oob: int
    jitter_n_total: int


@jaxtyped(typechecker=beartype)
@dataclass(frozen=True)
class ValStats:
    """One validation pass. The caller ``.item()``s ``f1_macro`` / ``f1_min``."""
    val_loss: float
    f1_macro: Float32[Tensor, '']
    f1_min: Float32[Tensor, '']
    f1_per_class: Float32[Tensor, 'classes']
    present: Bool[Tensor, 'classes']
    accuracy: float
    top2_accuracy: float


def accumulate_tp_fp_fn(
    logits: Tensor, labels: Tensor, n_classes: int,
    tp: Tensor, fp: Tensor, fn: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Add this batch's per-class TP / FP / FN onto the running totals.

    Shared by the train and validate loops. Context-agnostic: the caller owns
    the grad context (train wraps the call in ``torch.no_grad()``; validate runs
    under the ``@torch.no_grad()`` decorator). The returned tensors stay on the
    inputs' device, so the accumulators never round-trip to CPU per batch.
    """
    preds = logits.argmax(dim=1)
    batch_tp, batch_fp, batch_fn = accumulate_class_counts(preds, labels, n_classes)
    return tp + batch_tp, fp + batch_fp, fn + batch_fn


def macro_min_over_present(f1: Tensor, present: Tensor) -> tuple[Tensor, Tensor]:
    """Macro (mean) and min F1 over the present classes only.

    Shared zero-support guard for ``validate`` and ``Task.test``: any class with
    no ground truth this pass would otherwise score F1=0 by construction,
    dragging macro down by 1/n and pinning min at 0. ``present`` is a boolean
    mask over classes; an all-absent pass returns ``(0.0, 0.0)``.
    """
    if present.any():
        masked = f1[present]
        return masked.mean(), masked.min()
    return torch.tensor(0.0), torch.tensor(0.0)


def train_one_epoch(
    model: nn.Module,
    loader,
    coupled_flip: CoupledFlip,
    constrained_jitter: ConstrainedJitter,
    n_classes: int,
    loss_fn,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LambdaLR,  # learning rate scheduler
    device,
) -> EpochStats:
    """Train for one epoch, accumulating per-class TP / FP / FN alongside loss.

    Per-class counts feed ``AdaptiveFocalLoss.update_alpha`` at the call site.
    They're cheap (three batched ``bincount`` calls per batch) and stay
    accumulated even when the loss has no use for them, so the train loop
    keeps a uniform return signature regardless of which loss is active.

    Augmentations fire flip-then-jitter per the framework doc. Both ops
    roll independently per-clip so within a batch some clips are
    flipped, some jittered, some both, some neither. Jitter accumulates
    two diagnostic counters across the epoch:

    - ``jitter_n_effective``: clips that received a non-zero shift, for
      ``Aug/jitter_effective_rate`` (case-1 dropout indicator).
    - ``jitter_n_oob``: clips whose effective shift pushed at least one
      previously-real shuttle frame off-screen, triggering the
      ``(0, 0)`` sentinel; for ``Aug/shuttle_oob_rate``.

    :return: an ``EpochStats`` (field types/shapes on the dataclass).
    """
    model.train()  # enable dropout (not global TF-style layer trainability flag)
    total_loss = 0.0
    tp = torch.zeros(n_classes, dtype=torch.long, device=device)
    fp = torch.zeros(n_classes, dtype=torch.long, device=device)
    fn = torch.zeros(n_classes, dtype=torch.long, device=device)
    jitter_n_effective = 0
    jitter_n_oob = 0
    jitter_n_total = 0

    for (human_pose, pos, shuttle), video_len, labels in loader:
        human_pose, shuttle, pos, video_len, labels = to_device(
            device, human_pose, shuttle, pos, video_len, labels,
        )

        # Augmentations: flip first (clean spatial transform), then jitter.
        # Each rolls independently per-clip; coupled_flip mirrors all three
        # streams in their own coord frames and recomputes bones from the
        # post-flip+post-swap joints; constrained_jitter shifts pos+shuttle
        # only with layered-conditional bounds and zero-frame preservation.
        human_pose, pos, shuttle = coupled_flip(human_pose, pos, shuttle)
        human_pose, pos, shuttle, n_eff, n_oob = constrained_jitter(
            human_pose, pos, shuttle,
        )
        jitter_n_effective += n_eff
        jitter_n_oob += n_oob
        jitter_n_total += human_pose.shape[0]

        human_pose = flatten_pose_features(human_pose)
        logits = model(human_pose, shuttle, pos=pos, video_len=video_len)
        loss: Tensor = loss_fn(logits, labels)

        # Manual gradient step: zero grads, backprop, update weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()       # update learning rate according to cosine schedule

        total_loss += loss.item()  # .item() extracts Python float from single-element tensor

        # Per-class confusion counts on argmax preds. no_grad() because preds
        # are detached labels; nothing here needs an autograd graph.
        with torch.no_grad():
            tp, fp, fn = accumulate_tp_fp_fn(logits, labels, n_classes, tp, fp, fn)

    train_loss = total_loss / len(loader)
    return EpochStats(
        train_loss=train_loss,
        tp=tp, fp=fp, fn=fn,
        jitter_n_effective=jitter_n_effective,
        jitter_n_oob=jitter_n_oob,
        jitter_n_total=jitter_n_total,
    )


@torch.no_grad()  # disables gradient computation — saves memory during eval
def validate(
    model: nn.Module,
    loss_fn,
    loader,
    device,
    n_classes: int,
) -> ValStats:
    model.eval()  # disable dropout (not global TF-style layer trainability flag)
    total_loss = 0.0
    # Accumulate per-class TP/FP/FN on device (mirrors train_one_epoch);
    # one .cpu() after the loop, not four per batch.
    cum_tp = torch.zeros(n_classes, dtype=torch.long, device=device)
    cum_fp = torch.zeros(n_classes, dtype=torch.long, device=device)
    cum_fn = torch.zeros(n_classes, dtype=torch.long, device=device)
    cum_top2 = 0  # ground truth among the two highest logits, summed over samples
    cum_n = 0     # total samples seen

    for (human_pose, pos, shuttle), video_len, labels in loader:
        human_pose, shuttle, pos, video_len, labels = to_device(
            device, human_pose, shuttle, pos, video_len, labels,
        )

        human_pose = flatten_pose_features(human_pose)
        logits = model(human_pose, shuttle, pos=pos, video_len=video_len)
        loss: Tensor = loss_fn(logits, labels)
        total_loss += loss.item()

        cum_tp, cum_fp, cum_fn = accumulate_tp_fp_fn(
            logits, labels, n_classes, cum_tp, cum_fp, cum_fn,
        )

        # Top-2 accuracy needs the two highest logits, so it's the one metric
        # not already in the confusion counts; accumulate it here.
        cum_n += labels.size(0)
        top2_idx = logits.topk(2, dim=1).indices
        cum_top2 += int((top2_idx == labels.unsqueeze(1)).any(dim=1).sum())

    cum_tp = cum_tp.cpu()
    cum_fp = cum_fp.cpu()
    cum_fn = cum_fn.cpu()
    val_loss = total_loss / len(loader)

    # Per-class F1 via the shared eps-guarded helper, the same definition the
    # train side feeds update_alpha. Value-identical to the old NaN-fill
    # hand-roll: the eps only rescues zero-count denominators (where NaN-fill
    # gave 0 anyway), and at float32 it vanishes against any nonzero count.
    f1_score = per_class_f1_from_counts(cum_tp, cum_fp, cum_fn)

    # Only classes present in the val set count toward macro/min.
    present = (cum_tp + cum_fn) > 0
    f1_score_avg, f1_score_min = macro_min_over_present(f1_score, present)

    # Accuracy is exactly correct/total: every sample is a TP for its class (if
    # right) or an FN for it (if wrong), so the correct count is sum(cum_tp).
    accuracy = float(cum_tp.sum() / cum_n) if cum_n else 0.0
    top2_accuracy = cum_top2 / cum_n if cum_n else 0.0
    return ValStats(
        val_loss=val_loss,
        f1_macro=f1_score_avg,
        f1_min=f1_score_min,
        f1_per_class=f1_score,
        present=present,
        accuracy=accuracy,
        top2_accuracy=top2_accuracy,
    )


# ==========================================================================
# Training loop with TensorBoard logging and early stopping
# ==========================================================================

def _build_loss_fn(
    n_classes: int,
    class_ls: list[str],
    taxonomy: Taxonomy,
    device,
    hyp: Hyp,
):
    """Resolve hyp's three loss branches (CE / class-weighted CE / adaptive-focal)
    into a single ``loss_fn`` instance.

    Owns the two fail-loud guards: ``adaptive_focal`` is mutually exclusive
    with ``class_weights``; ``adaptive_focal`` requires ``label_smoothing=0.0``.

    ``label_smoothing`` softens targets from [0,1] to reduce overconfidence.
    BST paper / TemPose default is 0.1; we sweep this knob to test whether it's
    bottlenecking the small-support classes that lose ground when the cleaner
    Phase-2 pose data lifts the head of the F1 distribution. See
    docs/architecture_notes/hparams_sweep_speculations.md.

    ``class_weights``: optional manual per-class loss multipliers. Used as a
    smoke test for whether loss-side reweighting can move the bottleneck F1
    classes (wrist_smash + its confusion partner smash). Renormalised to mean
    1.0 so the overall loss magnitude stays comparable to uniform CE (keeps LR /
    grad-clip behaviour aligned across cells). None = uniform.

    ``adaptive_focal``: class-F1-driven CDB-loss with optional focal
    modulation. Replaces the static class_weights knob with an EMA-smoothed
    per-class weight that re-prioritises classes whose train F1 stays low.
    Mutually exclusive with class_weights and forces label_smoothing=0 (LS
    softens targets, contaminating focal's per-sample hardness estimate).
    """
    if hyp.adaptive_focal is not None:
        if hyp.class_weights:
            raise ValueError(
                'adaptive_focal and class_weights are mutually exclusive; '
                'set one of them to None.'
            )
        if hyp.label_smoothing != 0.0:
            raise ValueError(
                'adaptive_focal requires label_smoothing=0.0 (LS softens '
                'targets so confident-correct samples have p_t < 1.0, '
                "contaminating focal's per-sample hardness signal). "
                f'Got label_smoothing={hyp.label_smoothing}.'
            )
        af_cfg = hyp.adaptive_focal
        loss_fn = AdaptiveFocalLoss(
            class_names=class_ls,
            tau=af_cfg.get('tau', 1.0),
            gamma=af_cfg.get('gamma', 1.0),
            momentum=af_cfg.get('momentum', 0.9),
            warm_up_epochs=af_cfg.get('warm_up_epochs', 5),
            f1_floor=af_cfg.get('f1_floor', 0.0),
        ).to(device)
        print(
            f"[loss] adaptive focal (CDB-F1): "
            f"tau={loss_fn.tau}, gamma={loss_fn.gamma}, "
            f"momentum={loss_fn.momentum}, "
            f"warm_up_epochs={loss_fn.warm_up_epochs}, "
            f"f1_floor={loss_fn.f1_floor}"
        )
        return loss_fn
    if hyp.class_weights:
        weights = torch.ones(n_classes, device=device)
        for cls_name, multiplier in hyp.class_weights.items():
            if cls_name not in class_ls:
                raise ValueError(
                    f"class_weights key '{cls_name}' not in the taxonomy "
                    f"{taxonomy.name!r} class list ({len(class_ls)} classes): "
                    f"{class_ls}."
                )
            weights[class_ls.index(cls_name)] = multiplier
        weights = weights * (n_classes / weights.sum())  # renormalise mean to 1.0
        print("[loss] class-weighted CE (renormalised, mean=1.0):")
        for i, c in enumerate(class_ls):
            print(f"    {c:25s} weight={weights[i].item():.3f}")
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=hyp.label_smoothing)
    return nn.CrossEntropyLoss(label_smoothing=hyp.label_smoothing)


def _split_param_groups(model: nn.Module):
    """Decay vs no-decay walk for AdamW. Returns the two raw lists; the caller
    builds the AdamW param-group dicts (so ``hyp.weight_decay`` / ``hyp.lr``
    reads stay co-located with the optimiser construction).

    Excludes norm gains, biases, and the learned tokens / positional embeddings
    from decay: decaying an LN/BN gain pulls its scale toward zero, and decaying
    a sinusoidally-seeded positional embedding erodes the positional signal.
    Matters at the lambda 0.1-0.4 the sweep covers; standard transformer recipe
    (Wang & Aitchison don't decay normalisation layers). ``ndim<=1`` catches
    every norm gain/beta and bias; the two name hints catch the five ndim>=2
    BST-owned params a shape rule misses. Verified split for BST_CG_AP:
    27 decay / 55 no-decay tensors (model-pinned: a different variant or a
    requires_grad change moves the count).
    """
    no_decay_name_hints = ('embedding_', 'learned_token_')
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        norm_or_bias = param.ndim <= 1
        token_or_posemb = any(hint in name for hint in no_decay_name_hints)
        if norm_or_bias or token_or_posemb:
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def _build_augmentations(n_bones, hyp: Hyp):
    """Build the locked Task-2 augmentation pair: centreline flip across all
    three streams (COCO bilateral joint-index swap + bone recompute) plus
    constrained pos+shuttle jitter (layered conditional bounds, joints
    untouched). ``hyp`` supplies pose_style and the augmentation dict.

    Bone recompute requires the JnB_bone pose style; other styles (J_only,
    JnB_interp, Jn2B) need their own recompute helpers which are out of scope
    per docs/architecture_notes/augmentation_framework.md.
    """
    if hyp.pose_style != 'JnB_bone':
        raise NotImplementedError(
            f'Augmentation framework currently supports pose_style=JnB_bone only; '
            f'got {hyp.pose_style!r}. Bone recompute via the bone_pairs table is the '
            f'mechanism that propagates the flip+swap into bones; J_only has no bones, '
            f'JnB_interp uses joint-pair midpoints, Jn2B uses both. Lift the equivalents '
            f'to torch in preparing_data/augmentations.py before re-enabling the others.'
        )
    # Direct index rather than .get(key, default): the Hyp always carries all
    # five aug keys (dict literal + all-or-nothing CLI override), so a missing
    # key is a malformed config and should fail loud, not train on a default.
    aug_cfg = hyp.augmentation
    coupled_flip = CoupledFlip(
        p=aug_cfg['p_flip'],
        n_joints=COCO_N_JOINTS,
        n_bones=n_bones,
    )
    constrained_jitter = ConstrainedJitter(
        p_roll=aug_cfg['p_jitter'],
        cap_y=aug_cfg['cap_y'],
        cap_x=aug_cfg['cap_x'],
        eps=aug_cfg['eps'],
    )
    print(
        f"[aug] coupled flip p={coupled_flip.p}, "
        f"constrained jitter p={constrained_jitter.p_roll} "
        f"(cap_y={constrained_jitter.cap_y}, cap_x={constrained_jitter.cap_x}, "
        f"eps={constrained_jitter.eps})"
    )
    return coupled_flip, constrained_jitter


def _build_optimiser(model, n_batches, hyp: Hyp):
    """AdamW (decoupled weight decay) + cosine schedule. _split_param_groups owns
    the decay rules; this helper owns the per-group weight_decay + hyp.lr wiring so
    the optimiser construction stays co-located with its hparams. ``n_batches`` =
    len(train_loader), for the scheduler's total-steps count. Returns
    ``(optimizer, scheduler)``.
    """
    decay, no_decay = _split_param_groups(model)
    assert (len(decay), len(no_decay)) == (27, 55), (
        f'decay split drifted to {(len(decay), len(no_decay))}, expected (27, 55); '
        'see the _split_param_groups docstring inventory pin'
    )
    print(f'[optim] AdamW lr={hyp.lr} weight_decay={hyp.weight_decay} '
          f'(decay={len(decay)} tensors, no_decay={len(no_decay)})')
    optimizer = optim.AdamW(
        [{'params': decay, 'weight_decay': hyp.weight_decay},
         {'params': no_decay, 'weight_decay': 0.0}],
        lr=hyp.lr,
    )
    # Cosine schedule: LR ramps up during warmup, then decays following a cosine curve.
    # HF formula: lr_factor = 0.5 * (1 + cos(pi * 2 * num_cycles * progress))
    #   num_cycles=0.5 -> LR ends at 0 (full standard cosine descent)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=hyp.warm_up_step,
        num_training_steps=(hyp.n_epochs * n_batches),  # total batches across all epochs
        num_cycles=0.5
    )
    return optimizer, scheduler


def train_network(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    save_path: Path,
    n_bones,
    n_classes: int,
    class_ls: list[str],
    taxonomy: Taxonomy,
    hyp: Hyp,
    tb_dir: Path | None = None,
):
    # tb_dir lands the event files under experiments/<run_id>/tb/serial_N/ so
    # TB folders pair with the run they came from. Default SummaryWriter() writes
    # to ./runs/<host_time>/, which is what older runs used.
    writer = SummaryWriter(log_dir=str(tb_dir)) if tb_dir else SummaryWriter()

    coupled_flip, constrained_jitter = _build_augmentations(n_bones, hyp)
    loss_fn = _build_loss_fn(n_classes, class_ls, taxonomy, device, hyp)
    optimizer, scheduler = _build_optimiser(model, len(train_loader), hyp)

    # Track top-2 of each metric (for HParams summary + verifying early-stop vs crash)
    best_macro = second_macro = 0.0
    best_macro_epoch = second_macro_epoch = 0
    best_min = second_min = 0.0
    best_min_epoch = second_min_epoch = 0
    best_val_loss, best_val_loss_epoch = float('inf'), 0
    early_stop_count = 0

    # Per-class val F1 snapshot, captured at the best-macro epoch (not a
    # per-class argmax across epochs) so the recorded breakdown matches the
    # checkpoint that actually gets saved. Surfaced to the serial manifest.
    best_val_f1_per_class = None
    best_val_present = None
    best_val_accuracy = None
    best_val_top2 = None
    best_macro_epoch_snap = None

    for epoch in range(1, hyp.n_epochs+1):
        # Auxiliary module schedule: cosine fade of CG/AP from 1.0 -> 0.0 across the run.
        # When disabled, factor stays at 1.0 -> identical to unscheduled BST_CG_AP.
        if hyp.use_aux_schedule:
            aux_factor = aux_schedule_factor(epoch, hyp.aux_fade_end_epoch)
        else:
            aux_factor = 1.0
        model.set_schedule_factors(cg_factor=aux_factor, ap_factor=aux_factor)

        t0 = time.time()
        epoch_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            coupled_flip=coupled_flip,
            constrained_jitter=constrained_jitter,
            n_classes=n_classes,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        # End-of-epoch per-class train F1 feeds AdaptiveFocalLoss; otherwise
        # the values are still computed (cheap) and logged to TB for context.
        train_per_class_f1 = per_class_f1_from_counts(
            epoch_stats.tp, epoch_stats.fp, epoch_stats.fn,
        )
        if isinstance(loss_fn, AdaptiveFocalLoss):
            loss_fn.update_alpha(train_per_class_f1)

        val_stats = validate(
            model=model,
            loss_fn=loss_fn,
            loader=val_loader,
            device=device,
            n_classes=n_classes,
        )
        t1 = time.time()
        print(f'Epoch({epoch}/{hyp.n_epochs}): train_loss={epoch_stats.train_loss:.3f}, '
              f'val_loss={val_stats.val_loss:.3f}, macro_f1={val_stats.f1_macro:.3f}, '
              f'min_f1={val_stats.f1_min:.3f} - {t1 - t0:.2f} s')

        if isinstance(loss_fn, AdaptiveFocalLoss):
            # Top-3 / bot-3 alpha summary so the operator can eyeball whether
            # the loss is reweighting toward the struggling classes each epoch.
            alpha_np = loss_fn.alpha.detach().cpu().numpy()
            order = alpha_np.argsort()
            ranked_alpha = [(class_ls[i], alpha_np[i]) for i in order]
            print_head_tail('alpha', ranked_alpha, 3, bot_first=True)

        _log_epoch_tb(
            writer=writer,
            epoch=epoch,
            epoch_stats=epoch_stats,
            val_stats=val_stats,
            train_per_class_f1=train_per_class_f1,
            aux_factor=aux_factor,
            scheduler=scheduler,
            loss_fn=loss_fn,
            class_ls=class_ls,
        )

        curr_macro, curr_min = val_stats.f1_macro.item(), val_stats.f1_min.item()

        # Early stop + snapshot best weights (piggybacks on new-best detection)
        early_stop_count += 1
        if curr_macro > best_macro:
            second_macro, second_macro_epoch = best_macro, best_macro_epoch
            best_macro, best_macro_epoch = curr_macro, epoch
            # state_dict() = snapshot of all model weights as a dict
            # deepcopy because state_dict returns references that would change as training continues
            best_state = deepcopy(model.state_dict())
            # Snapshot the per-class val F1 at this same best-macro epoch so the
            # recorded breakdown matches the saved checkpoint.
            best_val_f1_per_class = val_stats.f1_per_class.detach().cpu().numpy()
            best_val_present = val_stats.present.detach().cpu().numpy()
            best_val_accuracy = val_stats.accuracy
            best_val_top2 = val_stats.top2_accuracy
            best_macro_epoch_snap = epoch
            print(f'Picked! => Best value {curr_macro:.3f}')
            # Compact per-class snapshot on new-best epochs: top-5 and bot-5
            # of present classes, one line each. Full per-class breakdown
            # lands in the test-time log at the end of each serial.
            present_idx = val_stats.present.nonzero(as_tuple=True)[0].tolist()
            scored = sorted(
                [(class_ls[i], val_stats.f1_per_class[i].item()) for i in present_idx],
                key=lambda t: t[1],
            )
            print_head_tail('val', scored, 5, bot_first=False)
            early_stop_count = 0
        elif curr_macro > second_macro:
            second_macro, second_macro_epoch = curr_macro, epoch

        if curr_min > best_min:
            second_min, second_min_epoch = best_min, best_min_epoch
            best_min, best_min_epoch = curr_min, epoch
        elif curr_min > second_min:
            second_min, second_min_epoch = curr_min, epoch

        # Strict <: a later epoch that merely ties the best val loss doesn't
        # replace it, so the earliest epoch reaching the minimum is kept.
        if val_stats.val_loss < best_val_loss:
            best_val_loss, best_val_loss_epoch = val_stats.val_loss, epoch

        if early_stop_count == hyp.early_stop_n_epochs:
            print(f'Early stop with best value {best_macro:.3f}')
            break

    # Save best checkpoint and restore it into the model. Done before TB
    # hparam logging so a logging failure doesn't lose the trained weights.
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, str(save_path))
    model.load_state_dict(best_state)

    _write_hparams_summary(
        writer,
        best_macro, best_macro_epoch, second_macro, second_macro_epoch,
        best_min, best_min_epoch, second_min, second_min_epoch,
        best_val_loss, best_val_loss_epoch, epoch,
        hyp,
    )

    # Val metrics at the best-macro epoch (the checkpoint that gets saved):
    # macro/min/accuracy/top-2 + the present-class per-class F1, for the serial
    # manifest (extra.val_at_best_macro_epoch). macro/min are the mean/min of the
    # snapshot per-class, so they stay exactly consistent with the breakdown.
    # None if no epoch ever beat the macro=0.0 init (degenerate run).
    if best_val_f1_per_class is not None:
        per_class = {
            class_ls[i]: float(best_val_f1_per_class[i])
            for i in range(len(class_ls))
            if best_val_present[i]
        }
        f1s = list(per_class.values())
        # Python-float sum/len here on purpose, not macro_min_over_present:
        # these are float64 values headed for the YAML manifest, and routing
        # them through a float32 tensor .mean() would drift the recorded
        # numbers. present-filtered above, same zero-support rule as the helper.
        val_at_best = {
            'epoch': best_macro_epoch_snap,
            'macro_f1': sum(f1s) / len(f1s),
            'min_f1': min(f1s),
            'accuracy': best_val_accuracy,
            'top2_accuracy': best_val_top2,
            'per_class_f1': per_class,
        }
    else:
        val_at_best = None
    return model, val_at_best


# ==========================================================================
# Task: orchestrates data loading, model creation, training, and evaluation
# ==========================================================================


class Task:
    def __init__(self, taxonomy: Taxonomy, hyp: Hyp, n_joints=COCO_N_JOINTS,
                 weight_dir: Path = Path('weight')) -> None:
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
        self.n_joints = n_joints
        # Run config, threaded explicitly from here down (loaders, model build,
        # train_network); no helper reads the module-global default.
        self.hyp = hyp
        # pose_style lives here (not on prepare_dataloaders) so get_network_architecture
        # can build without a prior loader step; kills the old call-order trap.
        self.pose_style = hyp.pose_style
        # Head dim and class names come straight off the taxonomy now: labels.npy
        # lands in [0, taxonomy.n_classes) at collation time, so there's no
        # runtime active/full remap and no data-derived head sizing.
        self.taxonomy = taxonomy
        # Where to save/load weights for this run. Caller should pass a
        # per-invocation subdir (e.g. weight/run_YYYYMMDD_HHMMSS) so fresh
        # runs never collide with older weights — see __main__ setup.
        self.weight_dir = weight_dir

    def prepare_dataloaders(self, root_dir: Path):
        self.train_loader, \
        self.val_loader, \
        self.test_loader \
            = prepare_npy_collated_loaders(
                root_dir=root_dir,
                pose_style=self.pose_style,
                batch_size=self.hyp.batch_size,
                use_cuda=self.use_cuda,
                num_workers=(0, 0, 0),
                train_partial=self.hyp.train_partial
            )

        self._assert_label_coverage()

    def _assert_label_coverage(self) -> None:
        """Contract guard replacing the old runtime active-class adapter.

        Labels.npy is meant to land in ``[0, taxonomy.n_classes)`` at collation
        time, so the head is the full taxonomy. Two invariants, both fail loud:

        - no split may carry a label index outside the head (a corrupt or
          stale-vintage collated set).
        - train must cover every class in the taxonomy. A class the head can
          emit but train never teaches would carry a label-smoothed ghost
          gradient every step; better to refuse the run.

        Together these imply val/test can't hold a class absent from train,
        so there is no separate check for that. Reads labels
        post-``train_partial`` slicing, so a too-aggressive partial that
        starves a class is caught here too.
        """
        expected = set(range(self.taxonomy.n_classes))
        train_present = {int(x) for x in np.unique(self.train_loader.dataset.labels)}
        val_present = {int(x) for x in np.unique(self.val_loader.dataset.labels)}
        test_present = {int(x) for x in np.unique(self.test_loader.dataset.labels)}

        splits = (('train', train_present), ('val', val_present), ('test', test_present))
        oob_descriptions = []
        for split_name, present in splits:
            oob = sorted(present - expected)
            if oob:
                # taxonomy.classes can't name these; they sit past the head.
                named = [f'<oob:{i}>' for i in oob]
                oob_descriptions.append(f'{split_name}: {oob} ({named})')
        if oob_descriptions:
            raise ValueError(
                f'label indices outside the taxonomy {self.taxonomy.name!r} '
                f'head [0, {self.taxonomy.n_classes}): '
                f'{"; ".join(oob_descriptions)}. The collated labels.npy is '
                f'likely corrupt or a stale vintage; failing at startup rather '
                f'than as a CUDA IndexError inside the loss.'
            )

        missing_in_train = expected - train_present
        if missing_in_train:
            named = [self.taxonomy.classes[i] for i in sorted(missing_in_train)]
            raise ValueError(
                f'taxonomy {self.taxonomy.name!r} has {len(expected)} classes '
                f'but train covers only {len(train_present)}. Missing class '
                f'indices: {sorted(missing_in_train)} ({named}). Either lift '
                f'train_partial (currently {self.hyp.train_partial}) or use a '
                f'taxonomy whose head matches what train can teach.'
            )

    def get_network_architecture(self, model_name='BST_X'):
        """Create the model at the taxonomy head dim and ground its inputs.

        Output dim is ``taxonomy.n_classes`` directly; labels on disk are
        already in that index space (no runtime remap), and
        ``_assert_label_coverage`` has confirmed train teaches the whole head.
        """
        self.net, self.n_bones = build_bst_x_network(
            model_name,
            n_joints=self.n_joints,
            pose_style=self.pose_style,
            n_classes=self.taxonomy.n_classes,
            seq_len=self.hyp.seq_len,
            device=self.device,
        )
        self.model_name = model_name

    def seek_network_weights(self, model_info='', serial_no=1, tb_dir: Path | None = None):
        """Load existing weights if found, otherwise train from scratch. Weight filenames encode the
        full experiment config, e.g.: 'bst_x_JnB_bone_between_2_hits_with_max_limits_seq_100_bst_24_2.pt'

        :return: ``(weight_existed, val_at_best)``.
        ``weight_existed`` is True when a checkpoint was loaded (no training ran),
        False when freshly trained.
        ``val_at_best`` is the per-class val F1 snapshot from ``train_network``
        (None on the load path or a degenerate run).
        """
        parts = [self.pose_style]
        if model_info:
            parts.append(model_info)
        parts.append(self.taxonomy.name)
        if serial_no != 1:
            parts.append(str(serial_no))
        config_tag = '_'.join(parts)  # pose style, data config, taxonomy, serial

        weight_stem = f'{self.model_name.lower()}_{config_tag}'
        self.display_name = f'{self.model_name}_{config_tag}'  # printed name; model_name stays the arch key
        weight_path = self.weight_dir / f'{weight_stem}.pt'
        self.weight_path = weight_path
        if weight_path.exists():
            self.net.load_state_dict(
                torch.load(str(weight_path), map_location=self.device, weights_only=True)
            )
            return True, None  # weight already existed; no fresh val snapshot
        else:
            train_t0 = time.time()
            self.net, val_at_best = train_network(
                model=self.net,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                device=self.device,
                save_path=weight_path,
                n_bones=self.n_bones,
                n_classes=self.taxonomy.n_classes,
                class_ls=list(self.taxonomy.classes),
                taxonomy=self.taxonomy,
                hyp=self.hyp,
                tb_dir=tb_dir,
            )
            t = timedelta(seconds=int(time.time() - train_t0))
            print(f'Total training time: {t}')
            return False, val_at_best  # newly trained

    def test(self, dump: dict, show_details=False, show_confusion_matrix=False) -> dict:
        """Derive test top-1 metrics from a precomputed dump.

        ``dump`` is one split's output from ``dump_topk_predictions``: top-1 reads
        straight off ``y_pred_top1`` (argmax), no second forward pass through the
        test loader.
        """
        pred = torch.from_numpy(dump['y_pred_top1'])
        gt = torch.from_numpy(dump['y_true'])
        print(f'Test (num_strokes: {len(pred)}) =>')

        # torcheval on purpose: the headline manifest metric keeps an
        # implementation independent of the count-based helper train/val share.
        f1_score_each = multiclass_f1_score(
            pred, gt, num_classes=self.taxonomy.n_classes, average=None
        )

        # Present = classes with test ground truth. torcheval's bincount here
        # stays independent of the count helper (see the note above).
        present = torch.bincount(gt, minlength=self.taxonomy.n_classes) > 0
        present_idx = present.nonzero(as_tuple=True)[0].tolist()
        class_ls = list(self.taxonomy.classes)

        show_f1_results(
            model_name=self.display_name,
            f1_score_each=f1_score_each[present_idx] if present_idx else f1_score_each,
            class_ls=pad_class_labels(
                [class_ls[i] for i in present_idx] if present_idx else class_ls
            ),
            show_details=show_details
        )

        acc = torch.sum(pred == gt).item() / len(pred)
        print('Accuracy:', f'{acc:.3f}')

        if show_confusion_matrix:
            plot_confusion_matrix(
                y_true=gt,
                y_pred=pred,
                need_pre_argmax=False,
                model_name=self.display_name,
                font_size=6,
                save=False
            )

        macro_f1, min_f1 = macro_min_over_present(f1_score_each, present)
        # per-class stays list-indexed by present_idx: empty present_idx yields
        # {} (matching the all-absent macro/min of 0.0 from the helper).
        per_class_f1 = {
            class_ls[i]: float(f1_score_each[i].item()) for i in present_idx
        }

        return {
            'macro_f1':     float(macro_f1.item()),
            'min_f1':       float(min_f1.item()),
            'accuracy':     float(acc),
            'num_strokes':  int(len(pred)),
            'per_class_f1': per_class_f1,
        }

    def test_topk_acc(self, dump: dict, k=2) -> dict:
        """Derive top-k accuracy from a precomputed dump's raw logits.

        Re-derives top-k from a fresh ``torch.topk(logits, k=k)`` rather than
        slicing the stored ``topk_idx[:, :k]`` (the dump runs at k=5; slicing
        breaks ties differently from a k=2 topk on rank-boundary rows). Real
        trained logits are tie-free, so this matches a 3-pass top-k on actual data.
        """
        assert k > 1, 'k should be > 1'
        logits = torch.from_numpy(dump['logits'])
        gt = torch.from_numpy(dump['y_true'])
        pred = torch.topk(logits, k=k, dim=1).indices
        gt_in_topk = torch.any(pred == gt.unsqueeze(1), dim=1)  # one bool per sample: ground truth among the top-k
        # sum/len keeps exact float64; .float().mean() would round through float32 and drift the recorded accuracy
        acc = gt_in_topk.sum().item() / len(gt)
        print(f'Top{k} Accuracy: {acc:.3f}')
        return {f'top{k}_accuracy': float(acc)}

    def dump_predictions(
        self, run_dir: Path, serial_no: int, k: int = 5,
    ) -> dict[str, dict]:
        """Dump per-split prediction npz (raw logits + top-k + ground truth).

        The per-stroke-logits payload that motivated the refactor: lets the FE
        show per-clip confidence and any consumer fit post-hoc temperature
        scaling without re-running inference. One npz per split per serial under
        ``run_dir/predictions/``. Non-best serials are pruned manually after the
        runner finishes (no auto-deletion).

        Each split is re-read through a fresh ``shuffle=False`` loader, and the
        npz carries its own ``clip_stems`` column row-aligned with ``logits`` /
        ``y_true``. The stems come from the in-memory dataset, so they track the
        rows the model actually saw -- after the zero-length-clip drop and any
        train_partial reorder -- NOT the raw on-disk ``clip_stems.npy``. The
        FE-shape JSON converter joins row -> stem inside the npz, no external
        sidecar and no re-deriving the collation filters.

        Returns the per-split dump dicts so the caller can derive test metrics
        from the same forward pass.

        :param run_dir: experiments/<run_id>/ for this run.
        :param serial_no: serial whose weights are currently loaded in self.net.
        :param k: top-k width recorded per row.
        :return: ``{split_name: dump_dict}`` for train / val / test.
        """
        out_dir = run_dir / 'predictions'
        out_dir.mkdir(parents=True, exist_ok=True)
        sources = (
            ('train', self.train_loader),
            ('val',   self.val_loader),
            ('test',  self.test_loader),
        )
        dumps: dict[str, dict] = {}
        for split_name, source in sources:
            dataset = source.dataset
            ordered = DataLoader(
                dataset, batch_size=source.batch_size,
                shuffle=False, num_workers=0, pin_memory=False,
            )
            dump = dump_topk_predictions(self.net, ordered, self.device, k=k)
            write_prediction_npz(
                out_dir / f'{split_name}_serial_{serial_no}.npz',
                dump, dataset, self.taxonomy, run_dir.name, serial_no,
            )
            dumps[split_name] = dump
        return dumps


# ==========================================================================
# Main: train and test on ShuttleSet
# ==========================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser; parse_args stays in main so the parser is testable
    in isolation. The per-serial sharing contract the flags serve is enforced
    (and documented) in resolve_hyp.
    """
    parser = argparse.ArgumentParser(
        description='BST training entry point. CLI flags exist mainly for the '
                    'hparam_sweep wrapper; running with no flags trains a full '
                    '5-serial run from the module-level Hyp defaults.',
    )
    parser.add_argument(
        '--serial-no', type=int, default=None,
        help='Run only this serial (1-indexed) and exit. Used by hparam_sweep to '
             'pause between serials for kill checks. Requires --log-path and '
             '--run-id when serial-no > 1.',
    )
    parser.add_argument(
        '--run-id', type=str, default=None,
        help='Resume into an existing experiments/<run_id>/ dir. Required when '
             '--serial-no > 1; optional otherwise (a fresh run_<timestamp> is '
             'minted if absent).',
    )
    parser.add_argument(
        '--log-path', type=str, default=None,
        help='Pin the test log file path. Required when --serial-no > 1 so all '
             'serials append to the same log file. Without it, each invocation '
             'creates a fresh test_logs/test_<timestamp>.log.',
    )
    parser.add_argument('--p-flip', type=float, default=None)
    parser.add_argument('--p-jitter', type=float, default=None)
    parser.add_argument('--cap-y', type=float, default=None)
    parser.add_argument('--cap-x', type=float, default=None)
    parser.add_argument('--eps', type=float, default=None)
    # Cell selectors for collation_runner.py (and any manual override). All
    # optional; absent ones fall back to the module-level Hyp defaults.
    # --taxonomy / --split-column / --collation-id together pick the on-disk
    # collation to read; --ablation-id is the nullable training-time tag (augs
    # / loss / wiring on a fixed collation), manifest-only, never in the path.
    parser.add_argument('--taxonomy', default=None)
    parser.add_argument('--split-column', default=None)
    parser.add_argument('--collation-id', default=None)
    parser.add_argument('--ablation-id', default=None)
    # Swept AdamW weight decay (the WD-sweep dimension), overriding the Hyp
    # default. Absent leaves the module default (0.01). Applies to the decay
    # param group only; the no-decay group stays at 0.0 regardless.
    parser.add_argument('--weight-decay', type=float, default=None)
    # Testing-only n_epochs override (e.g. --serial-no 1 short-run bit-exacts
    # that don't want the full 80-epoch default). Not piped through
    # hparam_sweep; for production sweeps, edit Hyp.n_epochs directly.
    parser.add_argument('--n-epochs', type=int, default=None)
    return parser


def resolve_hyp(args: argparse.Namespace) -> Hyp:
    """Validate the CLI args and layer any overrides onto the module-level Hyp.

    Returns a fresh Hyp built from the module-global defaults. Raises on a bad serial-no
    contract or a partial augmentation override, both before any filesystem write.
    """
    # Per-serial invocation contract: pass all three sharing-flags together so every
    # serial lands in one run dir with one continuous log file. The runner drives serial
    # count per cell (5 default, 10 for headline cells), so there's no fixed upper bound
    # here beyond "1-indexed".
    if args.serial_no is not None:
        if args.serial_no < 1:
            raise ValueError(
                f'--serial-no must be >= 1, got {args.serial_no!r}.'
            )
        if args.serial_no > 1 and (not args.log_path or not args.run_id):
            raise ValueError(
                '--serial-no > 1 requires both --log-path and --run-id so '
                'subsequent serials append to the same log and share the run dir.'
            )

    # Augmentation CLI overrides are all-or-nothing. Wrapper passes the full cell-config
    # dict (base + overrides resolved); manual invocations leave them all None and use
    # the module-level Hyp defaults.
    aug_overrides = [args.p_flip, args.p_jitter, args.cap_y, args.cap_x, args.eps]
    provided = [x is not None for x in aug_overrides]
    if any(provided) and not all(provided):
        raise ValueError(
            'Augmentation CLI overrides must be all-or-nothing. Pass either '
            'all five (--p-flip --p-jitter --cap-y --cap-x --eps) or none.'
        )
    resolved = hyp  # module-level production defaults; the global is never rebound
    if all(provided):
        resolved = resolved._replace(augmentation=frozendict({
            'p_flip':   args.p_flip,
            'p_jitter': args.p_jitter,
            'cap_y':    args.cap_y,
            'cap_x':    args.cap_x,
            'eps':      args.eps,
        }))

    # Cell selectors: override the Hyp defaults when the runner (or a manual
    # invocation) passes them. Each is independent and nullable.
    cell_overrides = {}
    if args.taxonomy:
        cell_overrides['taxonomy'] = args.taxonomy
    if args.split_column:
        cell_overrides['split_column'] = args.split_column
    if args.collation_id:
        cell_overrides['collation_id'] = args.collation_id
    if args.ablation_id:
        cell_overrides['ablation_id'] = args.ablation_id
    if args.weight_decay is not None:
        cell_overrides['weight_decay'] = args.weight_decay
    if args.n_epochs is not None:
        cell_overrides['n_epochs'] = args.n_epochs
    if cell_overrides:
        resolved = resolved._replace(**cell_overrides)
    return resolved


class RunPaths(NamedTuple):
    run_dir: Path
    run_id: str
    log_path: Path
    weight_dir: Path
    collated_root: Path
    model_info: str


def resolve_run_paths(
    args: argparse.Namespace, hyp: Hyp, taxonomy: Taxonomy,
) -> RunPaths:
    """Resolve the run paths and register the run.

    Side effect: creates the test_logs dir and writes the run manifest via track_run, so
    the manifest exists before any serial appends to it.
    """
    # Collated dir naming via shared helper (mirrored on the prepare_train writer side);
    # see ``pipeline.config.derive_npy_collated_dir_basename``.
    if hyp.seq_len not in (30, 100):
        raise NotImplementedError(f'Unsupported hyp.seq_len={hyp.seq_len!r}; expected 30 or 100.')
    npy_collated_dir = derive_npy_collated_dir_basename(
        seq_len=hyp.seq_len,
        split_column=hyp.split_column,
        collation_id=hyp.collation_id,
    )

    # Weights filename suffix. Independent of the collated-dir name; encodes config
    # knobs that change per run (seq_len-derived window tag, train_partial). Empty
    # string is a valid value (seq_len=30, full data).
    model_info_parts: list[str] = []
    if hyp.seq_len == 100:
        model_info_parts.append(f'{CLIP_WINDOW}_seq_100')
    if not 0 < hyp.train_partial <= 1:
        raise ValueError(f'hyp.train_partial must be in (0, 1], got {hyp.train_partial}')
    if hyp.train_partial != 1:
        model_info_parts.append(f"train_partial_{str(hyp.train_partial).replace('0.', '0p', 1)}")
    model_info = '_'.join(model_info_parts)

    # ----------------------------------------------------------------------
    # Per-run experiment folder (tracked via run_tracker).
    # Every run mints experiments/bst_x/shuttleset/run_<timestamp>/ with:
    #   manifest.yaml          (hyperparams + config.classes, git SHA, per-serial metrics)
    #   weights/<save_name>.pt (best checkpoint per serial)
    #   tb/serial_N/           (TB event files per serial)
    #   predictions/<split>_serial_N.npz (per-stroke logits + top-k dump)
    # The runner passes a fixed --run-id across a cell's serials so they share one run
    # dir + log: serial 1 creates the manifest, later serials append via track_serial.
    # Weights are per-serial, so re-running a serial with --run-id finds its .pt and
    # skips training.
    # ----------------------------------------------------------------------
    timestamp = f'{datetime.now():%Y%m%d_%H%M%S}'
    run_id = args.run_id or f'run_{timestamp}'

    # Test output is auto-teed to a timestamped log file so metrics are never lost to a
    # dropped terminal. Training stdout stays on terminal only; TB captures it. One log
    # file per script invocation, all serials inside. Uses the fresh invocation
    # timestamp (not run_id) so resumed re-tests don't overwrite the original run's log
    # file.
    #
    # Anchor experiments/ and test_logs/ to the repo-root experiments/bst_x/shuttleset/
    # so write paths don't depend on cwd. Lets `python -m bst_x_train` land outputs in
    # the canonical run-artefact tree regardless of where it was invoked from.
    script_dir = Path(__file__).resolve().parent
    experiments_dir = script_dir.parent.parent / 'experiments' / 'bst_x' / 'shuttleset'
    log_dir = experiments_dir / 'test_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_path) if args.log_path else log_dir / f'test_{timestamp}.log'

    extra = compute_data_provenance(
        clips_csv_path=Path(hyp.clips_csv),
        collation_id=hyp.collation_id,
        npy_collated_dir=npy_collated_dir,
    )
    # config.classes lands the resolved class list next to the Hyp dump, mirroring
    # BRIC's manifest schema; the FE registry reads it without importing any taxonomy
    # module. track_run treats the dict as a Mapping and stores it verbatim
    # (config.collation_id / config.ablation_id ride along).
    config_payload = dict(hyp._asdict())
    # The two frozendict fields must land as plain dict: yaml.safe_dump's
    # representers are exact-type and reject dict subclasses (frozendict included).
    config_payload['augmentation'] = dict(hyp.augmentation)
    if hyp.adaptive_focal is not None:
        config_payload['adaptive_focal'] = dict(hyp.adaptive_focal)
    config_payload['classes'] = list(taxonomy.classes)
    run_dir, run_id = track_run(
        config=config_payload, run_id=run_id, log_path=log_path, extra=extra,
        experiments_dir=experiments_dir,
    )
    weight_dir = run_dir / 'weights'

    # Collated dir, resolved the same way the collator wrote it, via the shared
    # root helper (BST_X_COLLATED_DATA_ROOT, e.g. /scratch/comp320a on bourbaki,
    # else the in-repo preparing_data/ convention). taxonomy.name is the resolved
    # canonical name, matching the writer's parent dir. Without the env var the
    # reader looks in-repo while the writer wrote to /scratch, so keep them in sync.
    collated_root = (
        resolve_collated_data_root() / f'ShuttleSet_data_{taxonomy.name}' / npy_collated_dir
    )
    return RunPaths(run_dir, run_id, log_path, weight_dir, collated_root, model_info)


def run_serial(
    serial_no: int, taxonomy: Taxonomy, hyp: Hyp, weight_dir: Path,
    collated_root: Path, run_dir: Path, model_info: str, tee: Tee,
) -> None:
    """Run one serial end to end: build the model, load-or-train its weights,
    dump per-stroke predictions, test through the tee, and register the serial's
    metrics in the manifest."""
    print(f'Running serial {serial_no} ...')
    task = Task(
        n_joints=COCO_N_JOINTS, taxonomy=taxonomy, hyp=hyp,
        weight_dir=weight_dir,
    )
    task.prepare_dataloaders(root_dir=collated_root)

    task.get_network_architecture(model_name='BST_X')

    tb_dir = run_dir / 'tb' / f'serial_{serial_no}'
    _weight_exists, val_at_best = task.seek_network_weights(
        model_info=model_info, serial_no=serial_no, tb_dir=tb_dir,
    )

    # Per-stroke logits dump (all splits) for the FE / calibration. Runs
    # every serial; non-best are pruned manually after the runner finishes.
    # Returns the per-split dumps so test_metrics/topk_metrics can derive
    # off the same forward pass.
    dumps = task.dump_predictions(run_dir=run_dir, serial_no=serial_no, k=5)

    with redirect_stdout(tee):
        print(f'\n=== Serial {serial_no} ({task.display_name}) ===')
        test_metrics = task.test(
            dump=dumps['test'],
            show_details=True, show_confusion_matrix=False,
        )
        topk_metrics = task.test_topk_acc(dump=dumps['test'], k=2)

    # Writes the manifest entry, and if aim is installed (it isn't on
    # the HPC train venv, so usually a no-op) mirrors this serial into
    # Aim as a fresh run each call (aim 3.29 can't reopen a stable
    # hash). Re-running a serial adds another Aim run rather than
    # overwriting; the clean, idempotent rebuild is aim_backfill.py --wipe.
    track_serial(
        run_dir=run_dir,
        serial_no=serial_no,
        weights_path=task.weight_path,
        tb_dir=tb_dir,
        metrics={**test_metrics, **topk_metrics},
        extra=({'val_at_best_macro_epoch': val_at_best}
               if val_at_best else None),
    )

    print('Serial', serial_no, 'done.')


def main() -> None:
    # Load .env so BST_X_COLLATED_DATA_ROOT (and any BST_* paths) resolve the
    # same way the collator does; shell exports still win. No-op without .env.
    load_repo_dotenv()

    parser = build_arg_parser()
    args = parser.parse_args()
    hyp = resolve_hyp(args)  # main-local shadow of the module global

    # Resolve the taxonomy; its canonical name drives the on-disk dir +
    # weight-file naming, matching what the collator wrote.
    taxonomy = taxonomy_lookup(hyp.taxonomy)

    run_paths = resolve_run_paths(args, hyp, taxonomy)

    # Per-serial invocation: run only the requested serial. Otherwise loop the
    # manual default of 5. Log open mode flips to append for serial-no > 1 so
    # later per-serial invocations don't clobber the earlier blocks.
    if args.serial_no:
        serial_range = range(args.serial_no, args.serial_no + 1)
        log_open_mode = 'a' if args.serial_no > 1 else 'w'
    else:
        serial_range = range(1, 6)
        log_open_mode = 'w'

    with open(run_paths.log_path, log_open_mode) as log_f:
        tee = Tee(sys.stdout, log_f)
        _print_taxonomy_block(taxonomy, tee)
        for serial_no in serial_range:
            run_serial(
                serial_no, taxonomy, hyp, run_paths.weight_dir, run_paths.collated_root,
                run_paths.run_dir, run_paths.model_info, tee,
            )

    print(f'\nTest log saved to: {run_paths.log_path}')
    print(f'Run manifest:    {run_paths.run_dir / "manifest.yaml"}')


if __name__ == '__main__':
    main()
