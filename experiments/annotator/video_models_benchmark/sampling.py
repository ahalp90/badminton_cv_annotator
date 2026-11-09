"""Seeded contact/background batches with coverage before repetition."""
from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import numpy as np
from torch.utils.data import Sampler


class EvaluationSampler(Sampler[int]):
    """Evaluate absolute video frames at a fixed stride, keeping the full dataset."""

    def __init__(self, frames: np.ndarray, stride: int, offset: int = 0) -> None:
        if stride < 1 or not 0 <= offset < stride:
            raise ValueError("Stride must be positive and offset must be between zero and stride minus one")
        self.indices = np.flatnonzero(frames % stride == offset)
        self.stride, self.offset = stride, offset

    def __iter__(self) -> Iterator[int]:
        yield from self.indices.tolist()

    def __len__(self) -> int:
        return len(self.indices)


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


def shuffled_centres(indices: np.ndarray, rng: np.random.Generator) -> Iterator[int]:
    while True:
        yield from rng.permutation(indices)


class ContactBatchSampler(Sampler[list[int]]):
    """Precompute fixed runs or continue the same seeded stream without an end."""

    def __init__(
        self, labels: np.ndarray, batch_size: int, contact_fraction: float, steps: int, seed: int,
        *, continuous: bool = False,
    ) -> None:
        contact_count = round(batch_size * contact_fraction)
        if not 0 < contact_count < batch_size or contact_count != batch_size * contact_fraction:
            raise ValueError("contact_fraction must give a whole, non-zero count of both classes per batch")
        contact_indices = np.flatnonzero(labels > 0)
        background_indices = np.flatnonzero(labels == 0)
        if not len(contact_indices) or not len(background_indices):
            raise ValueError("Training needs eligible contact and background centres")
        self.start_step = 0
        self.batch_size, self.contact_count = batch_size, contact_count
        self.contact_indices, self.background_indices = contact_indices, background_indices
        self.seed = seed
        self.batches: np.ndarray | None = None
        if continuous:
            return
        contact_seed, background_seed, order_seed = np.random.SeedSequence(seed).spawn(3)
        contacts = shuffled_cycles(contact_indices, steps * contact_count, np.random.default_rng(contact_seed))
        backgrounds = shuffled_cycles(
            background_indices, steps * (batch_size - contact_count), np.random.default_rng(background_seed),
        )
        batches = np.concatenate((contacts.reshape(steps, contact_count),
                                  backgrounds.reshape(steps, batch_size - contact_count)), axis=1)
        order = np.random.default_rng(order_seed).random(batches.shape).argsort(axis=1)
        self.batches = np.take_along_axis(batches, order, axis=1)

    def __iter__(self) -> Iterator[list[int]]:
        if self.batches is None:
            yield from islice(self.continuous_batches(), self.start_step, None)
        else:
            for batch in self.batches[self.start_step:]:
                yield batch.tolist()

    def continuous_batches(self) -> Iterator[list[int]]:
        contact_seed, background_seed, order_seed = np.random.SeedSequence(self.seed).spawn(3)
        contacts = shuffled_centres(self.contact_indices, np.random.default_rng(contact_seed))
        backgrounds = shuffled_centres(self.background_indices, np.random.default_rng(background_seed))
        order_rng = np.random.default_rng(order_seed)
        while True:
            batch = list(islice(contacts, self.contact_count))
            batch.extend(islice(backgrounds, self.batch_size - self.contact_count))
            order = order_rng.random(self.batch_size).argsort()
            yield [int(batch[index]) for index in order]

    def __len__(self) -> int:
        if self.batches is None:
            raise TypeError("Continuous training has no fixed batch count")
        return len(self.batches) - self.start_step
