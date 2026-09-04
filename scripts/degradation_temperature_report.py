"""Evidence for the degradation tanh temperature (issue #138).

``dataset_builder.degradation`` stores the raw least-squares slope alongside
a tanh-compressed ``slope_tanh`` computed at one fixed temperature,
``DEGRADATION_TEMPERATURE = 2.0``. Because the raw slope survives in every
``player_trends`` row, this script does not need to refit anything: given an
exported dataset directory, it reads ``player_trends.csv.gz`` and, for each
of a handful of candidate temperatures, reports what ``slope_tanh`` would
have looked like at that temperature. That turns "pick a magic number like 2"
into a choice backed by a distribution instead of a guess.

Usage::

    uv run python scripts/degradation_temperature_report.py \\
        --dataset-dir /scratch/<user>/dataset-v1/shuttleset
"""

from __future__ import annotations

from collections.abc import Sequence
import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dataset_builder.schema_v1 import PLAYER_TRENDS, read_table  # noqa: E402 (after sys.path insert)


CANDIDATE_TEMPERATURES: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
PERCENTILES: tuple[float, ...] = (5.0, 25.0, 50.0, 75.0, 95.0)
# tanh(x) > 0.99 once x > ~2.65: a value this close to +/-1 has lost most of
# the information that separated it from an even steeper slope.
SATURATION_THRESHOLD = 0.99


def percentiles_of(values: np.ndarray) -> dict[float, float]:
    """Return ``values``' percentiles at each point in ``PERCENTILES``."""
    return {p: float(np.percentile(values, p)) for p in PERCENTILES}


def temperature_report(slopes: np.ndarray, temperature: float) -> dict[str, object]:
    """Return the slope_tanh percentiles and saturation count at one temperature."""
    tanh_values = np.tanh(slopes / temperature)
    saturated = int(np.count_nonzero(np.abs(tanh_values) > SATURATION_THRESHOLD))
    return {
        "temperature": temperature,
        "percentiles": percentiles_of(tanh_values),
        "saturated_count": saturated,
        "saturated_fraction": saturated / len(tanh_values),
    }


def build_report(
    slopes: np.ndarray, temperatures: Sequence[float] = CANDIDATE_TEMPERATURES
) -> dict[str, object]:
    """Return raw-slope percentiles plus a ``temperature_report`` per candidate.

    :param slopes: raw ``player_trends.slope`` values, one per fitted trend.
    :param temperatures: candidate tanh temperatures to compare.
    """
    if len(slopes) == 0:
        raise ValueError("no slope values to report on")
    return {
        "n_slopes": len(slopes),
        "raw_slope_percentiles": percentiles_of(slopes),
        "by_temperature": [temperature_report(slopes, t) for t in temperatures],
    }


def format_report(report: dict[str, object]) -> str:
    """Render ``build_report``'s output as the same text the CLI prints."""
    lines = [f"{report['n_slopes']} raw slope values", "", "Raw slope percentiles:"]
    for p, value in report["raw_slope_percentiles"].items():
        lines.append(f"  p{p:g}: {value:.4f}")
    lines.append("")
    for entry in report["by_temperature"]:
        lines.append(f"Temperature {entry['temperature']:g}: slope_tanh percentiles")
        for p, value in entry["percentiles"].items():
            lines.append(f"  p{p:g}: {value:.4f}")
        lines.append(
            f"  saturated (|slope_tanh| > {SATURATION_THRESHOLD}): "
            f"{entry['saturated_count']} of {report['n_slopes']} "
            f"({entry['saturated_fraction'] * 100:.2f}%)"
        )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", required=True, type=Path,
        help="export-v1 or export-v1-shuttleset22 output directory containing player_trends.csv.gz",
    )
    args = parser.parse_args()

    trends = read_table(args.dataset_dir, PLAYER_TRENDS)
    slopes = trends["slope"].to_numpy(dtype=float)
    print(format_report(build_report(slopes)))


if __name__ == "__main__":
    main()
