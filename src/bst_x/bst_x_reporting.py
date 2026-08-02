"""Training-loop reporting split out of bst_x_train.py.

The per-epoch TensorBoard scalar logger, the end-of-run HParams summary, the
per-run taxonomy banner, and a small ranked-class printer shared by the loop's
alpha and val-F1 snapshots. Everything here is a pure sink (a SummaryWriter or
stdout); the writer is built in train_network and threaded in, so nothing in
this module owns run state.
"""

from contextlib import redirect_stdout
from typing import TYPE_CHECKING

import torch

from classifier_shared.taxonomy import Taxonomy
from loss.adaptive_focal import AdaptiveFocalLoss

if TYPE_CHECKING:
    # Type-only: annotations reference these, but importing them at runtime
    # would cycle with bst_x_train (which imports this module).
    from bst_x_train import EpochStats, Hyp, ValStats


def safe_rate(numerator: int, denominator: int) -> float:
    """Fraction guarded against a zero denominator (returns 0.0)."""
    return numerator / denominator if denominator > 0 else 0.0


def print_head_tail(metric: str, ranked: list[tuple[str, float]], k: int, *, bot_first: bool) -> None:
    """Print the k highest- and k lowest-scoring classes, one line each.

    ``ranked`` is a list of (name, score) pairs already sorted ascending by
    score; the caller owns the sort so each site keeps its own tie-break
    (numpy argsort for alpha, stable sorted for val F1). ``bot_first`` prints
    the bot line before the top line, matching the alpha summary's order; the
    val summary prints top first. Scores render at two decimals.
    """
    top_line = f'  {metric} top{k}: ' + ' '.join(
        f'{name}={score:.2f}' for name, score in reversed(ranked[-k:])
    )
    bot_line = f'  {metric} bot{k}: ' + ' '.join(
        f'{name}={score:.2f}' for name, score in ranked[:k]
    )
    if bot_first:
        print(bot_line)
        print(top_line)
    else:
        print(top_line)
        print(bot_line)


def _log_epoch_tb(
    writer,
    epoch,
    epoch_stats: 'EpochStats',
    val_stats: 'ValStats',
    train_per_class_f1,
    aux_factor,
    scheduler,
    loss_fn,
    class_ls,
):
    """Per-epoch TensorBoard scalars: train/val loss, macro/min F1 (train + val),
    the aux schedule factor + LR, the two jitter-rate diagnostics, and the
    per-class train/val F1 (and per-class alpha under adaptive focal).
    """
    writer.add_scalar('Loss/Train', epoch_stats.train_loss, epoch)
    writer.add_scalar('Loss/Val', val_stats.val_loss, epoch)
    writer.add_scalar('F1/Val_macro', val_stats.f1_macro, epoch)
    writer.add_scalar('F1/Val_min', val_stats.f1_min, epoch)
    # Train F1 macro/min summaries mirror the val pair above, so the
    # train-vs-val gap reads off two scalars per epoch instead of needing
    # to re-aggregate the per-class arrays. .mean()/.min() over the
    # length-n_classes tensor of active-class F1s, .item() unwraps to float.
    writer.add_scalar('F1_train/macro', train_per_class_f1.mean().item(), epoch)
    writer.add_scalar('F1_train/min', train_per_class_f1.min().item(), epoch)
    writer.add_scalar('Schedule/aux_factor', aux_factor, epoch)
    # Cosine LR per epoch. Deterministic from the schedule, but logging it
    # saves the reconstruction and overlays cleanly with the per-class F1 /
    # alpha arcs. get_last_lr()[0] = LR after this epoch's final step.
    writer.add_scalar('Schedule/learning_rate', scheduler.get_last_lr()[0], epoch)
    # Jitter effective rate: fraction of clips that rolled yes AND had at
    # least one non-degenerate axis. Watching this scalar shows whether the
    # case-1 (fully-degenerate envelope) skip rate is eating into the
    # nominal p_jitter target. See augmentation_framework.md.
    jitter_effective_rate = safe_rate(epoch_stats.jitter_n_effective, epoch_stats.jitter_n_total)
    writer.add_scalar('Aug/jitter_effective_rate', jitter_effective_rate, epoch)
    # Shuttle OOB rate: fraction of clips where the effective shift
    # pushed a previously-real shuttle frame off-screen, triggering the
    # (0, 0) sentinel. Diagnostic for the cap_x trade-off the doc flags
    # around edge-of-frame shuttle classes (cross_court_net_shot, rush
    # trajectories). High rate = cap_x is replacing a meaningful fraction
    # of real shuttle observations with the off-screen sentinel.
    shuttle_oob_rate = safe_rate(epoch_stats.jitter_n_oob, epoch_stats.jitter_n_total)
    writer.add_scalar('Aug/shuttle_oob_rate', shuttle_oob_rate, epoch)
    for i, c in enumerate(class_ls):
        writer.add_scalar(f'F1_train/{c}', train_per_class_f1[i].item(), epoch)
        # Val per-class F1 only for classes present in val this epoch; an
        # absent class scores F1=0 by construction and would read as a real
        # regression on the TB curve.
        if val_stats.present[i]:
            writer.add_scalar(f'F1_val/{c}', val_stats.f1_per_class[i].item(), epoch)
        if isinstance(loss_fn, AdaptiveFocalLoss):
            writer.add_scalar(f'Alpha/{c}', loss_fn.alpha[i].item(), epoch)


def _write_hparams_summary(
    writer,
    best_macro, best_macro_epoch, second_macro, second_macro_epoch,
    best_min, best_min_epoch, second_min, second_min_epoch,
    best_val_loss, best_val_loss_epoch, stopped_epoch,
    hyp: 'Hyp',
):
    """HParams summary: one row per run, sortable in TB's HParams tab.
    stopped_epoch - best_macro_epoch == early_stop_n_epochs confirms clean early-stop.
    Coerce non-scalar values (dicts, None, etc.) to strings; TB's add_hparams
    only accepts int / float / str / bool / Tensor. Closes the writer when done.
    """
    hparam_dict = {}
    for key, value in hyp._asdict().items():
        is_tb_scalar = isinstance(value, (int, float, str, bool)) or torch.is_tensor(value)
        hparam_dict[key] = value if is_tb_scalar else str(value)

    writer.add_hparams(
        hparam_dict=hparam_dict,
        metric_dict={
            'best/macro_f1':        best_macro,
            'best/macro_f1_epoch':  best_macro_epoch,
            'best/macro_f1_2nd':    second_macro,
            'best/macro_f1_2nd_ep': second_macro_epoch,
            'best/min_f1':          best_min,
            'best/min_f1_epoch':    best_min_epoch,
            'best/min_f1_2nd':      second_min,
            'best/min_f1_2nd_ep':   second_min_epoch,
            'best/val_loss':        best_val_loss,
            'best/val_loss_epoch':  best_val_loss_epoch,
            'stopped_epoch':        stopped_epoch,
        },
        # run_name='.' stops TB nesting a timestamped subdir per add_hparams call.
        run_name='.',
        global_step=stopped_epoch,
    )
    writer.close()


def _print_taxonomy_block(taxonomy: Taxonomy, tee) -> None:
    """Loud one-time taxonomy summary at run start, captured by the tee'd log.

    Resolved class list lives in the manifest's ``config.classes``; train/val/test
    coverage invariants are enforced by ``Task._assert_label_coverage``.
    """
    with redirect_stdout(tee):
        print(f'[taxonomy] {taxonomy.name}: {taxonomy.n_classes} classes, '
              f'has_sides={taxonomy.has_sides}, has_unknown={taxonomy.has_unknown}')
        print(f'[taxonomy] classes: {list(taxonomy.classes)}')
