"""Seeded contact/background batches with coverage before repetition."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler


def shuffled_cycles(indices: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a fixed stream, shuffling again only after every centre is consumed."""
    pieces = []
    remaining = count
    while remaining:
        permutation = rng.permutation(indices)
        take = min(remaining, len(permutation))
        pieces.append(permutation[:take])
        remaining -= take
    return np.concatenate(pieces)


class ContactBatchSampler(Sampler[list[int]]):
    """Precompute the complete run so recovery skips batches without redrawing them."""

    def __init__(self, labels: np.ndarray, batch_size: int, contact_fraction: float, steps: int, seed: int) -> None:
        contact_count = round(batch_size * contact_fraction)
        if not 0 < contact_count < batch_size or contact_count != batch_size * contact_fraction:
            raise ValueError("contact_fraction must give a whole, non-zero count of both classes per batch")
        contact_indices = np.flatnonzero(labels > 0)
        background_indices = np.flatnonzero(labels == 0)
        if not len(contact_indices) or not len(background_indices):
            raise ValueError("Training needs eligible contact and background centres")
        contact_seed, background_seed, order_seed = np.random.SeedSequence(seed).spawn(3)
        contacts = shuffled_cycles(contact_indices, steps * contact_count, np.random.default_rng(contact_seed))
        backgrounds = shuffled_cycles(
            background_indices, steps * (batch_size - contact_count), np.random.default_rng(background_seed),
        )
        batches = np.concatenate((contacts.reshape(steps, contact_count),
                                  backgrounds.reshape(steps, batch_size - contact_count)), axis=1)
        order = np.random.default_rng(order_seed).random(batches.shape).argsort(axis=1)
        self.batches = np.take_along_axis(batches, order, axis=1)
        self.start_step = 0

    def __iter__(self) -> Iterator[list[int]]:
        for batch in self.batches[self.start_step:]:
            yield batch.tolist()

    def __len__(self) -> int:
        return len(self.batches) - self.start_step
