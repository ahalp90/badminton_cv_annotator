"""Convert ShuttleSet player A/B labels to camera-relative Top/Bottom.

Start from the match.csv downcourt assignment. Swap ends for set 2 and
again after the first player reaches 11 points in set 3. ``collect_shots``
applies these changes while reading and filtering the set CSVs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from classifier_shared.taxonomy import ZH_TO_EN

# Columns we need from each set CSV.
_SHOT_COLS = ['rally', 'ball_round', 'frame_num',
              'roundscore_A', 'roundscore_B', 'player', 'type']


def collect_shots(
    set_info_dir: Path,
    v_info: pd.Series,
    stroke_types_zh: list[str],
) -> pd.DataFrame:
    """Read a video's labelled shots across sets and assign camera-relative players.

    :param set_info_dir: Directory containing the match folders.
    :param v_info: match.csv row with video folder name and downcourt assignment.
    :param stroke_types_zh: Chinese stroke names to retain.
    :return: Rows with set, rally, ball_round, frame_num, roundscore_A/B,
        player (Top/Bottom) and type (English).
    """
    folder_path = set_info_dir / v_info['video']
    first_A_is_top = bool(v_info['downcourt'])
    collected = []

    # Sets 1 and 2.
    for set_i in range(1, 3):
        csv_path = folder_path / f'set{set_i}.csv'
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)[_SHOT_COLS]
        df = df[df['type'].isin(stroke_types_zh)]
        df.insert(0, 'set', np.full(len(df), set_i, dtype=int))
        df = map_players(df, first_A_is_top, set_i)
        collected.append(df)

    # Set 3 (if it exists): split at the 11-point court switch.
    csv_path = folder_path / 'set3.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)[_SHOT_COLS]
        df.insert(0, 'set', np.full(len(df), 3, dtype=int))

        # Find the switch before filtering, while all score rows are present.
        i_split = find_set3_switch_rally(df)
        df_before = map_players(df.iloc[:i_split], first_A_is_top, 1)
        df_after = map_players(df.iloc[i_split:], first_A_is_top, 2)

        df_before = df_before[df_before['type'].isin(stroke_types_zh)]
        df_after = df_after[df_after['type'].isin(stroke_types_zh)]
        collected.extend([df_before, df_after])

    if not collected:
        return pd.DataFrame(columns=['set'] + _SHOT_COLS)

    result = pd.concat(collected).reset_index(drop=True)
    result['type'] = result['type'].map(ZH_TO_EN)
    return result


def map_players(df: pd.DataFrame, first_A_is_top: bool, set_num: int) -> pd.DataFrame:
    """Copy the rows and replace A/B player labels with Top/Bottom.

    Use the initial court assignment for set 1; reverse it for set 2.
    For set 3, split at the 11-point switch and call with set_num 1 before it
    and 2 after it.
    """
    df = df.copy()
    if first_A_is_top ^ (set_num == 2):
        df['player'] = np.where(df['player'] == 'A', 'Top', 'Bottom')
    else:
        df['player'] = np.where(df['player'] == 'B', 'Top', 'Bottom')
    return df


def find_set3_switch_rally(df: pd.DataFrame) -> int:
    """Row position where players change ends in set 3.

    Find the first rally where either recorded score reaches 11. Return the
    first row after that rally, or len(df) if nobody reaches 11. Use the full
    score rows before filtering shots.
    """
    i_A = df['roundscore_A'].searchsorted(11, side='left')
    i_B = df['roundscore_B'].searchsorted(11, side='left')
    i = min(i_A, i_B)

    # The recorded set may end before anyone reaches 11.
    if i >= len(df):
        return len(df)

    switch_rally = df.iloc[i]['rally']
    return df['rally'].searchsorted(switch_rally, side='right')
