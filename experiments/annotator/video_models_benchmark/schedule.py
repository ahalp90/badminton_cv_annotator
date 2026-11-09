"""Continuation schedule and validation patience for the fixed-budget pilot."""
from __future__ import annotations

import math

VALIDATION_INTERVAL = 5_000
HALF_CYCLE_STEPS = 10_000
PATIENCE = 3


def continuation_rate(step: int, start_step: int, peak: float, *, half_cycle_steps: int | None = None) -> float:
    half_cycle = HALF_CYCLE_STEPS if half_cycle_steps is None else half_cycle_steps
    return peak * 0.5 * (1 - math.cos(math.pi * (step - start_step) / half_cycle))


def checks_without_improvement(history: list[dict]) -> int:
    best_loss, unsuccessful = float("inf"), 0
    for row in history:
        loss = row["validation"]["loss"]
        if loss < best_loss:
            best_loss, unsuccessful = loss, 0
        else:
            unsuccessful += 1
    return unsuccessful
