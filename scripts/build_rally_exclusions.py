"""List ShuttleSet rallies with flaws or contacts lost from the shots master.

The original set annotations are the source of the rally population and shot order.
Each distinct reason gets one row, including for rallies absent from the master.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ORIGINALS = REPO_ROOT / 'data' / 'shuttleset' / 'set'
DEFAULT_MASTER = REPO_ROOT / 'training' / 'data' / 'shuttleset' / 'annotations' / 'shots_master.csv'
DEFAULT_OUTPUT = REPO_ROOT / 'data' / 'shuttleset' / 'rally_exclusions.csv.gz'
KEY_COLUMNS = ['vid', 'set', 'rally']
OUTPUT_COLUMNS = [*KEY_COLUMNS, 'reason']


def load_original_contacts(originals: Path) -> pd.DataFrame:
    """Read all original contacts before any master filtering."""
    matches = pd.read_csv(originals / 'match.csv', usecols=['id', 'video'])
    if matches['id'].duplicated().any():
        raise ValueError('match.csv contains duplicate video IDs')

    parts = []
    for match in matches.itertuples(index=False):
        match_dir = originals / match.video
        set_paths = sorted(match_dir.glob('set[0-9]*.csv'))
        if not set_paths:
            raise FileNotFoundError(f'no set annotations for vid={match.id}: {match_dir}')
        for set_path in set_paths:
            set_number = int(set_path.stem.removeprefix('set'))
            contacts = pd.read_csv(set_path, usecols=['rally', 'ball_round', 'player', 'flaw'])
            contacts.insert(0, 'set', set_number)
            contacts.insert(0, 'vid', int(match.id))
            parts.append(contacts)

    originals_df = pd.concat(parts, ignore_index=True)
    originals_df[['vid', 'set', 'rally']] = originals_df[KEY_COLUMNS].astype(int)
    return originals_df


def build_exclusions(originals: Path, master_path: Path) -> pd.DataFrame:
    """Return one row per distinct rally exclusion reason."""
    contacts = load_original_contacts(originals)
    master = pd.read_csv(master_path, usecols=['vid', 'set_id', 'rally'])
    master['set'] = master['set_id'].str.removeprefix('set').astype(int)
    master[KEY_COLUMNS] = master[KEY_COLUMNS].astype(int)

    original_counts = contacts.groupby(KEY_COLUMNS, sort=False).size().rename('original_rows')
    master_counts = master.groupby(KEY_COLUMNS).size().rename('master_rows')
    counts = original_counts.to_frame().join(master_counts).fillna({'master_rows': 0})
    missing_rows = counts.index[counts['original_rows'] > counts['master_rows']]

    flaw_marked = contacts['flaw'].notna() & contacts['flaw'].astype(str).str.strip().ne('')
    flawed = contacts.loc[flaw_marked, KEY_COLUMNS].drop_duplicates()

    # A/B identify players throughout a rally, even when their court sides swap later.
    ordered = contacts.sort_values([*KEY_COLUMNS, 'ball_round'], kind='stable')
    previous = ordered.groupby(KEY_COLUMNS, sort=False)['player'].shift()
    same_player = ordered['player'].isin(['A', 'B']) & ordered['player'].eq(previous)
    repeated = ordered.loc[same_player, KEY_COLUMNS].drop_duplicates()

    parts = [flawed.assign(reason='flaw')]
    parts.append(missing_rows.to_frame(index=False).assign(reason='missing_master_rows'))
    parts.append(repeated.assign(reason='consecutive_same_player'))
    result = pd.concat(parts, ignore_index=True)[OUTPUT_COLUMNS]
    return result.drop_duplicates().sort_values(OUTPUT_COLUMNS, kind='stable').reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--originals', type=Path, default=DEFAULT_ORIGINALS)
    parser.add_argument('--master', type=Path, default=DEFAULT_MASTER)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    exclusions = build_exclusions(arguments.originals, arguments.master)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    exclusions.to_csv(arguments.output, index=False, compression={'method': 'gzip', 'mtime': 0})
    print(f'Wrote {len(exclusions)} reasons for {exclusions[KEY_COLUMNS].drop_duplicates().shape[0]} rallies '
          f'to {arguments.output}')


if __name__ == '__main__':
    main()
