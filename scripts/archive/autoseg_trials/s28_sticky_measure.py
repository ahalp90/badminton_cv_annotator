"""Measure sticky-anchor contact cells at label level, +/-10 frames."""
from __future__ import annotations

import csv
from pathlib import Path

import m_miss_junk_census as census
import s28_sticky_pin as pin

HERE = Path(__file__).resolve().parent
OUTPUT_PATH = HERE / 's28_sticky_measure_outputs' / 'sticky_cells.csv'


def _video_cell(cfg, chain, label: str, radius: int, multiple: float) -> dict[str, object]:
    master = pin.harness.pd.read_csv(pin.harness.retest.SHOTS_MASTER)
    gt_rallies = pin.harness.retest.load_gt_rallies(master, cfg.vid)
    gt_frames = [frame for rally in gt_rallies for frame in rally.stroke_frames]
    candidates = [
        contact.contact_frame for contact in chain.filtered_contacts
        if contact.wrist_near is not False and contact.suppressed is not True
    ]
    matches = census._global_matches(gt_frames, candidates)
    return {
        'cell': label,
        'video': cfg.name,
        'impulse_multiple': multiple,
        'radius': radius,
        'matches': len(matches),
        'gt': len(gt_frames),
        'candidates': len(candidates),
        'recall': len(matches) / len(gt_frames) if gt_frames else 0.0,
        'precision': len(matches) / len(candidates) if candidates else 0.0,
    }


def _write_rows(rows: list[dict[str, object]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        'cell', 'video', 'impulse_multiple', 'radius', 'matches', 'gt',
        'candidates', 'recall', 'precision',
    )
    with OUTPUT_PATH.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                column: f'{row[column]:.6f}' if column in ('recall', 'precision')
                else row[column]
                for column in columns
            })


def main() -> None:
    stage8 = pin.stage8
    original_multiple = stage8.CONTACT_IMPULSE_MULTIPLE
    if original_multiple != 4.0:
        raise AssertionError(f'unexpected starting CONTACT_IMPULSE_MULTIPLE={original_multiple}')

    rows: list[dict[str, object]] = []
    # m4 pins are asserted before the measurement-only m2 patch.
    m4_runs = {
        9: pin.run_radius(9),
        7: pin.run_radius(7),
    }
    for radius, runs in m4_runs.items():
        for cfg in (pin.harness.retest.PILOT, pin.harness.retest.VID15):
            rows.append(_video_cell(cfg, runs[cfg.name][0], f'm4/r{radius}', radius, 4.0))

    try:
        stage8.CONTACT_IMPULSE_MULTIPLE = 2.0
        for radius in (7, 9):
            for cfg in (pin.harness.retest.PILOT, pin.harness.retest.VID15):
                chain, _digest = pin.run_video(cfg, radius, score_output=False)
                rows.append(_video_cell(cfg, chain, f'm2/r{radius}', radius, 2.0))
    finally:
        stage8.CONTACT_IMPULSE_MULTIPLE = original_multiple

    if stage8.CONTACT_IMPULSE_MULTIPLE != 4.0:
        raise AssertionError('CONTACT_IMPULSE_MULTIPLE was not restored to 4.0')

    rows.sort(key=lambda row: (str(row['cell']), str(row['video'])))
    _write_rows(rows)
    for row in rows:
        print(
            f"{row['cell']} {row['video']}: matches={row['matches']} GT={row['gt']} "
            f"candidates={row['candidates']} recall={row['recall']:.6f} "
            f"precision={row['precision']:.6f}"
        )
    print(f'wrote {OUTPUT_PATH.relative_to(HERE)}')


if __name__ == '__main__':
    main()
