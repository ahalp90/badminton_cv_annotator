"""Replay fixed court sharing over one saved fast-robust video extract, without new court searches.

Each originally detected scene gets back the court its own search chose, before the
source run's sharing replaced it. Its middle-frame context is rebuilt from the source
video, fresh DeepLSD lines and the saved poses, as the source run's detect() built it.
Every restored court joins one VideoPool in scene order. A `no_court` scene whose
middle frame's hash is near a group's reference then gets its context rebuilt the
same way and joins as a receiver. One apply() shares courts within each camera view,
receivers included. No search runs, so no scene changes its own court.

Run with PYTHONPATH at the frozen sharing checkout. The source extract is only read.

The output keeps the source's metadata and every scene row in order. Other rows are
copied unchanged. view_groups is rebuilt for the whole video. The source run's
timings move to source_extraction_seconds. Each row's seconds and stage_seconds also
stay those of the source run. The replay block holds this run's timings and its
checks against the source.

Nothing is written when a court scene's context cannot be rebuilt, a court scene
cannot join a view group or a group fails to apply. The same holds when the rebuilt
groups, pooled fits or scenes' own scores differ from the source's. A receiver that
fails is logged, recorded on its row and counted; it keeps no court.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import logging
import os
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

# Match the source run's thread limits before NumPy initialises its libraries.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'

import numpy as np

from court_detector import feet, view_pool
from court_detector.detect import PreparedView, SceneCourts, Switches, json_round_trip
from court_detector.run_video import (
    CourtTools,
    frame_view,
    load_court_tools,
    read_json,
    write_json,
)
from court_detector.scene_sources import SceneInfo
from court_detector.video_inputs import PoseArrays, VideoFrames

SOURCE_TIMING_FIELDS = ('tools_seconds', 'setup_seconds', 'scene_seconds', 'pose_prerun_seconds',
                        'processing_seconds', 'total_seconds')
# The scene's own outcome, which view_pool.replace_court and receive_court keep under a
# scene_ prefix when they share a court
RESTORED_FIELDS = ('corners_native_px', 'chosen_key', 'reused_from', 'status', 'no_court_reason')

logger = logging.getLogger('replay_court_sharing')


def command_option(command: list[str], option: str) -> str:
    """The value after an option in the source run's detector command."""
    return command[command.index(option) + 1]


def restored_row(row: dict[str, Any]) -> dict[str, Any]:
    """A copy of a source row with the scene's own court back in place and no old pool record."""
    restored = copy.deepcopy(row)
    for field in RESTORED_FIELDS:
        own_field = 'scene_' + field
        if own_field in restored:
            restored[field] = restored.pop(own_field)
    restored.pop('view_pool', None)
    return restored


def middle_frame(row: dict[str, Any], frames: VideoFrames) -> np.ndarray:
    """The scene's middle frame, decoded with its whole foot window in order, as scene_courts decoded it.

    The feet check then reads the window from the cache.
    """
    anchor = row['frame_index']
    window = feet.window_frames(anchor, frames.fps, row['start_frame'], row['end_frame'])
    return frames.read(window)[window.index(anchor)]


def prepared_view(row: dict[str, Any], frame: np.ndarray, frames: VideoFrames, people: PoseArrays, tools: CourtTools,
                  switches: Switches) -> PreparedView:
    """A scene's middle-frame inputs and context, built as the source run's detect() built them."""
    scene = SceneInfo(row['start_frame'], row['end_frame'])
    view = frame_view(row['view_id'], frame, row['frame_index'], scene, tools.lines, people)
    feet_window = feet.window_feet(view, people, frames, switches.enforce_scene_consistency,
                                   not switches.require_people)
    return tools.detector.prepare(view, feet_window.all_feet_px)


def check_groups(pool: view_pool.VideoPool, source_groups: list[dict[str, Any]]) -> None:
    """Fail unless the rebuilt view groups hold the source's scenes in the source's order.

    The source run added the same courts in the same order. Grouping reads only each
    middle frame, its context and its own court, so a faithful rebuild repeats every
    group. Every fast-robust member donates, so the members also fix the donors.
    """
    rebuilt = [[member.row['view_id'] for member in group.members] for group in pool.groups]
    saved = [group['member_view_ids'] for group in source_groups]
    if rebuilt != saved:
        first = next(index for index, pair in enumerate(itertools.zip_longest(rebuilt, saved)) if pair[0] != pair[1])
        raise RuntimeError(f'rebuilt view groups differ from the source from group {first}: '
                           f'{len(rebuilt)} rebuilt, {len(saved)} saved')


def check_source_scores(pool: view_pool.VideoPool, source: dict[str, Any]) -> dict[str, Any]:
    """Fail unless each rebuilt scene's own court scores exactly as the source saved it.

    The source computed these scores from the same frame, lines and poses with the same
    measurement code, and JSON floats read back exactly. Any difference therefore means
    a rebuilt context differs from the one the source searched. The source saved scene
    scores only in groups that chose a shared court. Other scenes are counted by their
    source group's reason.
    """
    saved_scores = {row['view_id']: row['view_pool']['scores']['scene']
                    for row in source['scenes'] if 'view_pool' in row}
    group_reasons = {view_id: str(group['reason'])
                     for group in source['view_groups'] for view_id in group['member_view_ids']}
    matched, differed, unsaved = 0, [], Counter()
    for group in pool.groups:
        for member in group.members:
            view_id = member.row['view_id']
            if view_id not in saved_scores:
                unsaved[group_reasons[view_id]] += 1
                continue
            rebuilt = json_round_trip(member.middle_score)
            if rebuilt == saved_scores[view_id]:
                matched += 1
            else:
                differed.append((view_id, saved_scores[view_id].get('combined_score'), rebuilt.get('combined_score')))
    if differed:
        raise RuntimeError(f'{len(differed)} of {matched + len(differed)} saved scene scores differ from the rebuilt '
                           f'ones; first (view, saved, rebuilt) combined scores: {differed[:5]}')
    return {'matched_exactly': matched, 'without_saved_score': dict(unsaved)}


def check_fits(summaries: list[dict[str, Any]], source_groups: list[dict[str, Any]]) -> int:
    """Fail unless each group the source fitted gets the source's donor markings and pooled fit.

    Both come before the choice between courts, the only step 84bbba1e changed.

    :return: The number of fits checked.
    """
    checked = 0
    for summary, saved in zip(summaries, source_groups, strict=True):
        if 'fit' not in saved:
            continue  # too few donor scenes to pool
        rebuilt = json_round_trip({'markings': summary['markings'], 'fit': summary['fit']})
        if rebuilt != {'markings': saved['markings'], 'fit': saved['fit']}:
            raise RuntimeError(f"{summary['reference_view_id']}: rebuilt pooled fit differs from the source's")
        checked += 1
    return checked


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, required=True, help='source fast-robust run_video .json.gz extract')
    parser.add_argument('--config', type=Path, required=True,
                        help="the source run's run_config.json.gz, with its cohort and detector command")
    parser.add_argument('--output', type=Path, required=True, help='new .json.gz for the replayed video')
    args = parser.parse_args()
    # This also stops the replay overwriting its source.
    if args.output.exists():
        parser.error(f'{args.output} already exists')
    return args


def add_receivers(rows: list[dict[str, Any]], pool: view_pool.VideoPool, frames: VideoFrames, people: PoseArrays,
                  tools: CourtTools, switches: Switches) -> int:
    """Add each courtless scene whose middle frame may match a view group to the pool, as a receiver.

    Every group exists by now. A frame whose hash is near no group's reference cannot be
    placed, so it skips its lines and context.

    :return: The number of receivers added.
    """
    added = 0
    for row in rows:
        frame = middle_frame(row, frames)
        if not pool.shortlisted(view_pool.image_hash(frame)):
            continue
        pool.add_receiver(row, prepared_view(row, frame, frames, people, tools, switches))
        added += 1
        logger.info('%s: rebuilt receiver %d', row['view_id'], added)
    return added


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    args = parse_arguments()
    started = perf_counter()
    source = read_json(args.input)
    config = read_json(args.config)
    video_id = source['video_id']
    if source['court_mode'] != view_pool.CourtMode.FAST_ROBUST or source['reuse_courts'] or source['saved_lines']:
        raise ValueError(f'{video_id}: only fast-robust extracts with live lines and no reuse can be replayed')
    # A video outside the source run's cohort has no entry, so it stops here.
    entry = {item['id']: item for item in config['cohort']}[video_id]
    video = Path(entry['video'])
    if video.name != source['video']:
        raise ValueError(f"{video_id}: cohort video {video} is not the source video {source['video']}")

    # Every other switch the context and VideoPool read keeps the source run's default.
    # The line templates' device matters only to the search, so it stays on the CPU.
    switches = Switches(require_people=source['require_people'])
    command = config['detector_command']
    tools = load_court_tools(switches, deeplsd_source=Path(command_option(command, '--deeplsd-source')),
                             deeplsd_weights=Path(command_option(command, '--deeplsd-weights')),
                             device=command_option(command, '--device'))
    rows = [restored_row(row) for row in source['scenes']]
    court_rows = [row for row in rows if row['status'] == 'court']
    # Errors and scenes too short for the foot window never reached a fit, so they cannot receive.
    courtless_rows = [row for row in rows if row['status'] == 'no_court']
    pool = view_pool.VideoPool(tools.detector.live, switches, view_pool.CourtMode.FAST_ROBUST)
    setup_started = perf_counter()
    prepare_seconds = add_seconds = 0.0
    with VideoFrames(video) as frames:
        metadata = (frames.fps, frames.frame_count, list(frames.size))
        if metadata != (source['fps'], source['frame_count'], source['native_size']):
            raise ValueError(f'{video_id}: {video} has fps, frame count and size {metadata}, unlike the source')
        people = PoseArrays.from_directory(Path(entry['people']))
        if people.frame_count < frames.frame_count:
            raise ValueError(f'{video_id}: saved poses do not cover the source video')
        setup_seconds = perf_counter() - setup_started
        for number, row in enumerate(court_rows, start=1):
            prepare_started = perf_counter()
            prepared = prepared_view(row, middle_frame(row, frames), frames, people, tools, switches)
            # In fast-robust mode a scene's court is its middle frame's own fit, so it fills
            # both of SceneCourts' court fields, as detect() does.
            corners = np.asarray(row['corners_native_px'], dtype=float)
            scene = SceneCourts(prepared.context, prepared.native_frame, corners, corners)
            add_started = perf_counter()
            pool.add(row, scene)
            prepare_seconds += add_started - prepare_started
            add_seconds += perf_counter() - add_started
            # VideoPool.add logs a failed join and records it on the row instead of raising.
            if 'view_pool' in row:
                raise RuntimeError(f"{row['view_id']}: could not join a view group: {row['view_pool']['error']}")
            logger.info('%s: rebuilt court scene %d/%d', row['view_id'], number, len(court_rows))

        check_groups(pool, source['view_groups'])
        source_scores = check_source_scores(pool, source)
        logger.info('%s: %d view groups and %d saved scene scores match the source; adding receivers', video_id,
                    len(pool.groups), source_scores['matched_exactly'])
        receivers_started = perf_counter()
        receivers_added = add_receivers(courtless_rows, pool, frames, people, tools, switches)
        receivers_seconds = perf_counter() - receivers_started

    apply_started = perf_counter()
    summaries = pool.apply()
    apply_seconds = perf_counter() - apply_started
    # VideoPool.apply logs a failed group and keeps its scene courts instead of raising.
    failed = [summary['reference_view_id'] for summary in summaries if 'error' in summary]
    if failed:
        raise RuntimeError(f'{video_id}: view groups failed to apply: {failed}')
    fits_checked = check_fits(summaries, source['view_groups'])

    changed_choices = sum((summary['chosen_court'], summary['chosen_view_id']) != (saved['chosen_court'],
                                                                                    saved['chosen_view_id'])
                          for summary, saved in zip(summaries, source['view_groups'], strict=True))
    source_corners = {row['view_id']: row['corners_native_px'] for row in source['scenes']}
    changed_courts = sum(row['corners_native_px'] != source_corners[row['view_id']] for row in court_rows)
    received = [row['view_id'] for row in courtless_rows if row['status'] == 'court']
    failed_receivers = [row['view_id'] for row in courtless_rows if 'error' in row.get('view_pool', {})]
    result = {key: value for key, value in source.items() if key not in SOURCE_TIMING_FIELDS}
    result.update(scenes=rows, view_groups=summaries,
                  source_extraction_seconds={field: source[field] for field in SOURCE_TIMING_FIELDS})
    result['replay'] = {
        'source_file': args.input.name, 'source_commit': config['commit'],
        'detector_source': str(Path(view_pool.__file__).resolve().parent), 'court_scenes': len(court_rows),
        'changed_courts': changed_courts, 'source_scores': source_scores,
        'view_groups': {'count': len(summaries), 'fits_matched': fits_checked, 'changed_choices': changed_choices},
        'receivers': {'courtless_scenes': len(courtless_rows), 'contexts_rebuilt': receivers_added,
                      'placed': sum(len(summary['receiver_view_ids']) for summary in summaries),
                      'received': len(received), 'failed': failed_receivers},
        'seconds': {'tools': tools.load_seconds, 'setup': setup_seconds, 'prepare': prepare_seconds,
                    'pool_add': add_seconds, 'receivers': receivers_seconds, 'pool_apply': apply_seconds,
                    'total': perf_counter() - started},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Write beside the output, then rename, so the output appears only once complete.
    partial = args.output.with_name(args.output.name + '.partial')
    write_json(partial, result)
    partial.replace(args.output)
    logger.info('%s: %d of %d courts changed and %d of %d courtless scenes took a court; wrote %s', video_id,
                changed_courts, len(court_rows), len(received), len(courtless_rows), args.output)


if __name__ == '__main__':
    main()
