"""Summarise local_built_courts.py's per-view records.

Prints the rebuild check, then three measures of the courts each pair builds before full scoring:
the horizon's distance (the bound on near-straight-down courts), the players' implied width, and how
deep the courts that matter rank in their pair by line-guess average, with build order beside it.

Usage: summarise_built_courts.py OUT_DIR UNFILTERED_OUT_DIR
  UNFILTERED_OUT_DIR: local_built_courts.py's output with --without-upright-filter
"""

import json
import sys
from pathlib import Path

import numpy as np

ROLES = ("chosen", "top five", "shortlist")
K_VALUES = (256, 512, 1024, 2048, 4096, 8192)
# As in local_built_courts.py: 20 bins per tenfold step, from 1 cm to 100 m.
WIDTH_BINS_M = np.geomspace(0.01, 100.0, 81)
FIXED_WIDTH_BOUNDS_M = (0.2, 3.0)


def deepest(courts: list[dict], role: str, field: str) -> int | None:
    """The deepest court in the role, each court counted at its shallowest copy."""
    shallowest: dict[str, int] = {}
    for court in courts:
        for court_id in court["roles"].get(role, []):
            shallowest[court_id] = min(shallowest.get(court_id, court[field]), court[field])
    return max(shallowest.values()) if shallowest else None


def read_records(folder: str) -> dict[str, dict]:
    return {path.stem: json.loads(path.read_text()) for path in sorted(Path(folder).glob("*.json"))}


def built(record: dict) -> int:
    return sum(pair["usable"] for pair in record["pairs"])


def pair_total(record: dict, field: str) -> int:
    return sum(pair[field] for pair in record["pairs"])


def courts_outside(records: list[dict], low: float, high: float) -> int:
    """Built courts whose player width is certainly outside [low, high] metres, at the histogram's resolution."""
    counts = np.sum([pair["width_counts"] for record in records for pair in record["pairs"]], axis=0)
    return int(counts[WIDTH_BINS_M[1:] <= low].sum() + counts[WIDTH_BINS_M[:-1] >= high].sum())


def role_widths(record: dict, roles: tuple[str, ...]) -> list[float]:
    return [court["player_width_m"] for court in record["courts"]
            if set(court["roles"]) & set(roles) and not np.isnan(court["player_width_m"])]


def main() -> None:
    filtered, unfiltered = read_records(sys.argv[1]), read_records(sys.argv[2])
    if sorted(filtered) != sorted(unfiltered):
        raise ValueError("the two runs cover different views")
    records = list(filtered.values())

    checked = [court for record in records for court in record["courts"] if "average_gap" in court]
    print(f"rebuild check over {len(checked)} shortlist courts in {len(records)} views: "
          f"largest corner gap {max(court['relative_corner_gap'] for court in checked):.1e} of the corner's "
          f"distance from the image origin, largest average gap {max(court['average_gap'] for court in checked):.1e}, "
          f"missing courts {sum(len(record['missing']) for record in records)}")

    print("\nhorizon distance from the image centre, in image widths. With lenses up to 90 degrees wide, a camera")
    print("within 20 degrees of straight down needs at least 1.37, and within 10 degrees at least 2.84")
    print("view\tchosen\tshortlists' largest\twith the filter: built, >= 1.37, >= 2.84, no horizon"
          "\twithout the filter: built, >= 1.37, >= 2.84, no horizon")
    fields = ("horizon_within_20_deg_bound", "horizon_within_10_deg_bound", "no_horizon")
    totals = np.zeros(8, dtype=int)
    for view_id, record in filtered.items():
        chosen = [court["horizon_widths"] for court in record["courts"] if "chosen" in court["roles"]]
        shortlist = [court["horizon_widths"] for court in record["courts"] if "shortlist" in court["roles"]]
        counts = [built(record), *(pair_total(record, field) for field in fields),
                  built(unfiltered[view_id]), *(pair_total(unfiltered[view_id], field) for field in fields)]
        totals += counts
        print(f"{view_id}\t{min(chosen):.2f}\t" if chosen else f"{view_id}\t-\t", end="")
        print(f"{max(shortlist):.2f}\t" if shortlist else "-\t", end="")
        print(", ".join(map(str, counts[:4])) + "\t" + ", ".join(map(str, counts[4:])))
    print("all\t\t\t" + ", ".join(map(str, totals[:4])) + "\t" + ", ".join(map(str, totals[4:])))

    low, high = FIXED_WIDTH_BOUNDS_M
    print("\nplayers' implied width, metres (median over the people a court puts on or within 1 m of it)")
    print(f"view\tchosen\tshortlists: min, median, max\tbuilt\tno one near\toutside {low}-{high} m: built, shortlisted")
    for record in records:
        chosen = role_widths(record, ("chosen",))
        shortlist = role_widths(record, ("shortlist",))
        shortlist_outside = sum(not low <= width <= high for width in shortlist)
        print(f"{record['view_id']}\t" + (f"{np.median(chosen):.2f}" if chosen else "-") + "\t"
              + (", ".join(f"{value:.2f}" for value in np.percentile(shortlist, [0, 50, 100])) if shortlist else "-")
              + f"\t{built(record)}\t{pair_total(record, 'no_player')}"
              + f"\t{courts_outside([record], low, high)}, {shortlist_outside}")

    total_built = sum(built(record) for record in records)
    total_unfiltered = sum(built(record) for record in unfiltered.values())
    print(f"\nbuilt courts certainly outside each width range (with the filter, of {total_built};"
          f" without it, of {total_unfiltered})")
    ranges = {"fixed": FIXED_WIDTH_BOUNDS_M}
    for label, roles in (("every shortlisted court", ("shortlist",)), ("chosen and top five", ("chosen", "top five"))):
        widths = [width for record in records for width in role_widths(record, roles)]
        ranges[f"widest range keeping {label}"] = (min(widths), max(widths))
    for label, (range_low, range_high) in ranges.items():
        inside = courts_outside(records, range_low, range_high)
        before = courts_outside(list(unfiltered.values()), range_low, range_high)
        print(f"{label}, {range_low:.2f}-{range_high:.2f} m: {inside} ({inside / total_built:.1%}); "
              f"without the filter {before} ({before / total_unfiltered:.1%})")

    print("\ndeepest rank in its pair, by line-guess average (by build order), of the courts in each role")
    print("view\t" + "\t".join(ROLES))
    depths = {(role, field): [] for role in ROLES for field in ("average_rank", "build_position")}
    for record in records:
        cells = []
        for role in ROLES:
            for field in ("average_rank", "build_position"):
                depth = deepest(record["courts"], role, field)
                if depth is not None:
                    depths[(role, field)].append(depth)
            average_depth, build_depth = (deepest(record["courts"], role, field)
                                          for field in ("average_rank", "build_position"))
            cells.append("-" if average_depth is None else f"{average_depth} ({build_depth})")
        print(f"{record['view_id']}\t" + "\t".join(cells))

    print("\nviews whose courts in each role all fall within the top K by average (by build order),"
          " out of views with such courts")
    print("K\t" + "\t".join(ROLES))
    for k in K_VALUES:
        cells = []
        for role in ROLES:
            by_average, by_build = (sum(depth <= k for depth in depths[(role, field)])
                                    for field in ("average_rank", "build_position"))
            cells.append(f"{by_average} ({by_build}) of {len(depths[(role, 'average_rank')])}")
        print(f"{k}\t" + "\t".join(cells))
    print("deepest\t" + "\t".join(f"{max(depths[(role, 'average_rank')])} ({max(depths[(role, 'build_position')])})"
                                  for role in ROLES))


if __name__ == "__main__":
    main()
