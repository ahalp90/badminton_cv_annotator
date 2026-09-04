from __future__ import annotations

import numpy as np
import pytest

from scripts.degradation_temperature_report import (
    CANDIDATE_TEMPERATURES,
    build_report,
    format_report,
    percentiles_of,
    temperature_report,
)


def test_percentiles_of_known_values() -> None:
    percentiles = percentiles_of(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert percentiles[50.0] == pytest.approx(3.0)
    assert percentiles[5.0] == pytest.approx(1.2)
    assert percentiles[95.0] == pytest.approx(4.8)


def test_temperature_report_saturation_count() -> None:
    # tanh(10 / 0.5) and tanh(10 / 1) both round to 1.0 well past the 0.99
    # threshold; tanh(10 / 8) = tanh(1.25) ~= 0.85 does not saturate.
    slopes = np.array([10.0, 10.0, -10.0, 0.0])

    tight = temperature_report(slopes, temperature=0.5)
    assert tight["saturated_count"] == 3
    assert tight["saturated_fraction"] == pytest.approx(0.75)

    loose = temperature_report(slopes, temperature=8.0)
    assert loose["saturated_count"] == 0


def test_build_report_covers_every_candidate_temperature() -> None:
    slopes = np.array([-4.0, -1.0, 0.0, 1.0, 4.0])
    report = build_report(slopes)

    assert report["n_slopes"] == 5
    assert [entry["temperature"] for entry in report["by_temperature"]] == list(
        CANDIDATE_TEMPERATURES
    )
    # A gentler temperature compresses less: fewer or equal saturated values
    # than a sharper one, for the same slope population.
    saturated_by_temperature = {
        entry["temperature"]: entry["saturated_count"] for entry in report["by_temperature"]
    }
    assert saturated_by_temperature[0.5] >= saturated_by_temperature[8.0]


def test_build_report_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="no slope values"):
        build_report(np.array([]))


def test_format_report_mentions_every_temperature_and_the_slope_count() -> None:
    text = format_report(build_report(np.array([1.0, 2.0, 3.0])))
    assert "3 raw slope values" in text
    for temperature in CANDIDATE_TEMPERATURES:
        assert f"Temperature {temperature:g}" in text
