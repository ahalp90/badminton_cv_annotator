"""MonoTrack's numerical augmentation and window sampling recipe.

Ported from sibling monotrack/modified-tracknet/train-hitnet.ipynb, cells 2,
8–9 and 11. Keep the numerical operations and RNG draw order in sync with
that recipe. Framework imports live in train_hitnet.py.
"""
from __future__ import annotations

import math
import random
from collections.abc import Iterator
from typing import Any

import numpy as np

num_consec = 12  # notebook name; the augmentation helpers read it as a global
left_window = 6
FEATURES_PER_FRAME = 78
BATCH_SIZE = 64
BALANCE_SEED = 0  # shared by every run, so all 12 runs see the same windows

# ---- Notebook cell 2, unchanged apart from formatting -------------------------------------------
eps = 1e-6


def reflect(x: np.ndarray) -> np.ndarray:
    x = np.array(x)
    idx = np.abs(x) < eps
    for i in range(0, x.shape[1], 2):
        x[:, i] = -x[:, i]
    x[idx] = 0.
    return x


def scale_data(x: np.ndarray) -> np.ndarray:
    x = np.array(x)

    def scale_by_col(x: np.ndarray, cols: list[int]) -> np.ndarray:
        x_ = np.array(x[:, cols])
        idx = np.abs(x_) < eps
        m, M = np.min(x_[~idx]), np.max(x_[~idx])
        x_[~idx] = (x_[~idx] - m) / (M - m) + 1
        x[:, cols] = x_
        return x

    even_cols = [2 * i for i in range(x.shape[1] // 2)]
    odd_cols = [2 * i + 1 for i in range(x.shape[1] // 2)]
    x = scale_by_col(x, even_cols)
    x = scale_by_col(x, odd_cols)
    return x


def identity(x: np.ndarray) -> np.ndarray:
    return x


def drop_consecutive(x: np.ndarray, rep_value: float = 0.) -> np.ndarray:
    x = np.array(x)
    for i in range(x.shape[0]):
        j = random.randint(0, num_consec - 1)
        x[i][max(0, 78 * (j - 2)):min(78 * (j + 2), 78 * num_consec)] = rep_value
    return x


def corrupt_consecutive(x: np.ndarray, rep_value: float = 0.) -> np.ndarray:
    x = np.array(x)
    for i in range(x.shape[0]):
        j = random.randint(0, num_consec - 1)
        l, r = max(0, 78 * (j - 2)), min(78 * (j + 2), 78 * num_consec)
        x[i][l:r] = np.random.rand(1, r - l)
    return x


def drop_data(x: np.ndarray, rep_value: float = 0, keep_prob: float = 0.95) -> np.ndarray:
    x = np.array(x)
    indices = np.random.choice(np.arange(x.size), replace=False, size=int(x.size * (1 - keep_prob)))
    x[np.unravel_index(indices, x.shape)] = rep_value
    return x


def corrupt_data(x: np.ndarray, keep_prob: float = 0.95) -> np.ndarray:
    x = np.array(x)
    idx = np.abs(x) < eps
    indices = np.random.choice(np.arange(x.size), replace=False, size=int(x.size * (1 - keep_prob)))
    shape = x[np.unravel_index(indices, x.shape)].shape
    low, hi = max(np.min(x[:, 0::2]), np.min(x[:, 1::2])), min(np.max(x[:, 0::2]), np.max(x[:, 1::2]))
    target = np.random.rand(*shape) * (hi - low) + low
    x[np.unravel_index(indices, x.shape)] = target
    x[idx] = 0.
    return x


def jiggle_and_rotate(x: np.ndarray) -> np.ndarray:
    x = np.array(x.reshape((x.shape[0], x.shape[1] // 2, 2)))
    idx = np.abs(x) < eps
    angle = (np.random.rand() - 0.5) * math.pi / 180 * 30
    rotate = np.array([[math.cos(angle), -math.sin(angle)],
                       [math.sin(angle), math.cos(angle)]])
    a, b = np.random.rand() * 0.1, np.random.rand() * 0.1
    shear = np.array([[1 + a * b, a],
                      [b, 1]])
    x = x @ shear @ rotate
    x[idx] = 0.
    x = x.reshape((x.shape[0], x.shape[1] * 2))
    return x


def drop_random_and_jiggle(x: np.ndarray) -> np.ndarray:
    return drop_data(jiggle_and_rotate(x), 0, 0.95)


def corrupt_random_and_jiggle(x: np.ndarray) -> np.ndarray:
    return corrupt_data(jiggle_and_rotate(x), 0.95)


def drop_consecutive_and_jiggle(x: np.ndarray) -> np.ndarray:
    return drop_consecutive(jiggle_and_rotate(x))


def corrupt_consecutive_and_jiggle(x: np.ndarray) -> np.ndarray:
    return corrupt_consecutive(jiggle_and_rotate(x))


# ---- Notebook cell 11 ----------------------------------------------------------------------------
choices = [identity, drop_consecutive, drop_data, corrupt_consecutive, corrupt_data, jiggle_and_rotate,
           drop_random_and_jiggle, corrupt_random_and_jiggle, drop_consecutive_and_jiggle,
           corrupt_consecutive_and_jiggle]
probs = [2, 1, 1, 1, 1, 5, 1, 1, 9, 9]
probs = [x / sum(probs) for x in probs]


def augment(x: np.ndarray) -> np.ndarray:
    aug = np.random.choice(choices, p=probs)
    if random.random() < 0.5:
        x = reflect(x)
    return scale_data(aug(x))


def fetch_data(x_train: np.ndarray, y_train: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    while True:
        idx = np.random.permutation(x_train.shape[0])
        x_train, y_train = x_train[idx], y_train[idx]
        for i in range(0, x_train.shape[0], BATCH_SIZE):
            yield augment(x_train[i:i + BATCH_SIZE]), y_train[i:i + BATCH_SIZE]



def role_windows(frames: np.ndarray, labels: np.ndarray, rallies: list[dict[str, Any]], role: str) -> tuple[np.ndarray, np.ndarray]:
    """Every 12-frame window of every usable rally in one split role, before balancing.

    A window's label is its hit class over frames 7-12 (left_window = 6), as in
    the notebook's y_list[left_window:] maximum.
    """
    starts, targets = [], []
    for rally in rallies:
        if "excluded" in rally or rally["role"] != role:
            continue
        window_count = rally["frames"] - num_consec + 1
        first = rally["offset"] + np.arange(window_count)
        label_frames = labels[first[:, None] + np.arange(left_window, num_consec)]
        starts.append(first)
        targets.append(label_frames.max(axis=1))
    starts, targets = np.concatenate(starts), np.concatenate(targets)
    return starts, targets


def balanced(starts: np.ndarray, targets: np.ndarray, frames: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Notebook cells 8-9: keep every hit window and a matching share of no-hit windows."""
    p = (targets > 0).sum() / (targets == 0).sum()
    keep = (np.random.default_rng(seed).random(len(targets)) < p) | (targets > 0)
    chosen = starts[keep]
    windows = frames[chosen[:, None] + np.arange(num_consec)].reshape(len(chosen), -1)
    return windows, targets[keep].astype(np.float32)


