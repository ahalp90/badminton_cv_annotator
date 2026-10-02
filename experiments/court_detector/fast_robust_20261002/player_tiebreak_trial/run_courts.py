"""Run the paired score-first court trial over its cohort, with resume and a queryable status.

This is the release run's run_courts.py with one change: each video runs
trial_run_video.py (beside this file) in place of `court_detector.run_video`. That
script runs production run_video and adds the trial arm from the same candidate pass.

Each video runs as its own `--manifest` batch of one. A crash therefore cannot leave
shared models or workers in a bad state for the next video. A result moves into
`videos/<id>.json.gz` only after its child exits 0, so that directory holds complete,
successful results and nothing else.

Run root layout:
  cohort.json.gz            the videos: id, video, people, plus dataset notes
  run_config.json.gz        commit, detector command and input paths of the first launch
  lanes/lane-<k>.json.gz    one runner's pid, host, state and current attempt
  lanes/lane-<k>.log        that runner's log: one line per video start and finish
  attempts/<id>/<n>-<time>/ run_video --output-dir of try n: run.log, summary.json.gz, exit_code
  videos/<id>.json.gz       the complete baseline result of each successful video
  trial_videos/<id>.json.gz that video's trial arm, in the same schema
  choices/<id>.jsonl.gz     both picks and the candidate table of each scene that reached the choice

Subcommands: `check` (input preflight), `ready` (handover check), `run`, `status`.
Standard library only, except `check`, which reads pose array headers with NumPy.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import logging
import lzma
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

RESULT_SCHEMA = 'court-detector-video/1'
TRIAL_ENTRY = Path(__file__).resolve().with_name('trial_run_video.py')
# The trial's extra outputs, moved beside videos/ when their video succeeds.
TRIAL_OUTPUTS = {'trial_videos': '.json.gz', 'choices': '.jsonl.gz'}
# The settings every result must report. The detector command below produces them.
INTENDED_SETTINGS = {'court_mode': 'fast-robust', 'require_people': True, 'reuse_courts': False,
                     'template_device': 'cuda', 'saved_people': True, 'saved_lines': False}
# No --reuse-courts or --full-score-limit: fresh, exhaustive middle-frame fits.
DETECTOR_OPTIONS = ['--pyscenedetect', '--court-mode', 'fast-robust', '--require-people',
                    '--device', 'cuda', '--template-device', 'cuda']
RUN_VIDEO_KEYS = ('id', 'video', 'people')  # run_video rejects any other manifest key
POSE_FILES = ('pose_bboxes.npy.xz', 'pose_kps.npy.xz', 'pose_ndet.npy.xz')  # what PoseArrays reads
# Three failed videos in a row suggest a broken GPU, disk or environment, not bad videos.
MAX_CONSECUTIVE_FAILURES = 3
SCENE_LINE = re.compile(r': scene (\d+)/(\d+) (\w+)$')
LOG_TAIL_BYTES = 2_000_000  # a whole-match log is about 5 MB; its last scene line is near the end

logger = logging.getLogger('run_courts')


def read_json(path: Path) -> Any:
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    """Write through a temporary file, so a reader never sees half a file."""
    temporary = path.with_name(path.name + '.tmp')
    with gzip.open(temporary, 'wt') as stream:
        json.dump(value, stream, indent=1)
    os.replace(temporary, path)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def seconds_since(stamp: str) -> float:
    return (datetime.now().astimezone() - datetime.fromisoformat(stamp)).total_seconds()


def result_problem(path: Path, video_id: str) -> str | None:
    """None when `path` is a complete result of the intended run; otherwise why it is not."""
    try:
        result = read_json(path)
    except (OSError, EOFError, ValueError) as error:  # missing, truncated or not JSON
        return f'unreadable: {error!r}'
    if result.get('schema') != RESULT_SCHEMA or result.get('video_id') != video_id:
        return f'schema {result.get("schema")!r} or video_id {result.get("video_id")!r} differs'
    differing = {key: result.get(key) for key, value in INTENDED_SETTINGS.items() if result.get(key) != value}
    if differing:
        return f'settings differ: {differing}'
    if any(row['status'] == 'detection_failed' for row in result['scenes']):
        return 'one or more scenes failed detection'
    if any(row.get('view_pool', {}).get('error') for row in result['scenes']):
        return 'one or more scenes failed to join a view group'
    if any(group.get('error') for group in result.get('view_groups', [])):
        return 'one or more view groups failed'
    return None


def checkout_commit(checkout: Path) -> str:
    """The checkout's commit. Uncommitted tracked changes would run code no commit names."""
    def git(*arguments: str) -> str:
        return subprocess.run(['git', '-C', str(checkout), *arguments], capture_output=True, text=True,
                              check=True).stdout.strip()

    if git('status', '--porcelain', '--untracked-files=no'):
        raise SystemExit(f'{checkout} has uncommitted tracked changes; commit or restore them first')
    return git('rev-parse', 'HEAD')


def detector_command(args: argparse.Namespace, manifest: Path, output_dir: Path) -> list[str]:
    return [str(args.python), '-u', str(TRIAL_ENTRY), '--manifest', str(manifest),
            '--output-dir', str(output_dir), *DETECTOR_OPTIONS, '--workers', str(args.workers),
            '--deeplsd-source', str(args.deeplsd),
            '--deeplsd-weights', str(args.deeplsd / 'weights' / 'deeplsd_md.tar')]


def lane_alive(lane: dict[str, Any]) -> bool | None:
    """Whether a lane's runner still runs; None when it ran on another host."""
    if lane['host'] != socket.gethostname():
        return None
    try:
        command = Path(f'/proc/{lane["pid"]}/cmdline').read_bytes()
    except OSError:
        return False
    # A reused pid belongs to some other program.
    return b'run_courts.py' in command


def read_lanes(root: Path) -> list[dict[str, Any]]:
    return [read_json(path) for path in sorted((root / 'lanes').glob('lane-*.json.gz'))]


def attempt_outcome(attempt: Path) -> dict[str, Any]:
    """One try's outcome from its own files: exit code, and the error when it failed."""
    outcome: dict[str, Any] = {'attempt': str(attempt), 'exit_code': None, 'error': None}
    if not (attempt / 'exit_code').exists():
        return outcome  # still running, or the runner died before the child finished
    outcome['exit_code'] = int((attempt / 'exit_code').read_text())
    if outcome['exit_code'] == 0 and (attempt / 'result_problem').exists():
        outcome['error'] = (attempt / 'result_problem').read_text()
    elif outcome['exit_code'] != 0:
        try:
            videos = read_json(attempt / 'results' / 'summary.json.gz')['videos']
        except (OSError, EOFError, ValueError):  # no summary, or one cut short when its writer died
            videos = []
        if videos and videos[0].get('error'):
            outcome['error'] = videos[0]['error']
        else:
            lines = log_tail(attempt / 'run.log').strip().splitlines()
            outcome['error'] = lines[-1] if lines else 'no output'
    return outcome


def log_tail(path: Path) -> str:
    with open(path, 'rb') as stream:
        stream.seek(max(0, path.stat().st_size - LOG_TAIL_BYTES))
        return stream.read().decode(errors='replace')


def write_lane(args: argparse.Namespace, lane: dict[str, Any], **changes: Any) -> None:
    lane.update(changes, updated_at=now())
    write_json(args.root / 'lanes' / f'lane-{args.lane}.json.gz', lane)


def run_attempt(entry: dict[str, Any], args: argparse.Namespace, lane: dict[str, Any]) -> bool:
    """Run one video in its own child process; move its result into videos/ when it succeeds."""
    video_id = entry['id']
    tries = args.root / 'attempts' / video_id
    # Numbered first, so name order is attempt order even for two tries in one second.
    attempt = tries / f'{len(list(tries.glob("*"))) + 1:02d}-{time.strftime("%Y%m%dT%H%M%S")}'
    attempt.mkdir(parents=True)
    manifest = attempt / 'manifest.json.gz'
    write_json(manifest, [{key: entry[key] for key in RUN_VIDEO_KEYS}])
    write_lane(args, lane, current_id=video_id, current_attempt=str(attempt), current_started_at=now())
    logger.info('START %s %s', video_id, attempt)
    started = time.monotonic()
    write_json(attempt / 'execution.json.gz', {'command': detector_command(args, manifest, attempt / 'results'),
                                             'cpus': sorted(os.sched_getaffinity(0)), 'lane': args.lane,
                                             'lanes': args.lanes, 'workers': args.workers})
    # The detector package resolves from the checkout, as in the dataset builder's child.
    environment = {**os.environ, 'PYTHONPATH': '.:src'}
    with open(attempt / 'run.log', 'wb') as log:
        child = subprocess.Popen(detector_command(args, manifest, attempt / 'results'), cwd=args.checkout,
                                 env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            exit_code = child.wait()
        finally:
            if child.poll() is None:  # the runner is stopping; take the child with it
                child.terminate()
                child.wait()
    (attempt / 'exit_code').write_text(f'{exit_code}\n')
    seconds = time.monotonic() - started
    if exit_code == 0:
        result = attempt / 'results' / 'videos' / f'{video_id}.json.gz'
        problem = result_problem(result, video_id)
        if problem is None:
            trial_result = attempt / 'results' / 'trial_videos' / f'{video_id}.json.gz'
            problem = result_problem(trial_result, video_id)
        if problem is None and not (attempt / 'results' / 'choices' / f'{video_id}.jsonl.gz').exists():
            problem = 'missing candidate measurements'
        if problem is None:
            # Before the baseline result, whose presence in videos/ marks the video complete.
            for directory, suffix in TRIAL_OUTPUTS.items():
                trial_output = attempt / 'results' / directory / f'{video_id}{suffix}'
                os.replace(trial_output, args.root / directory / trial_output.name)
            os.replace(result, args.root / 'videos' / f'{video_id}.json.gz')
            logger.info('DONE %s in %.0f s', video_id, seconds)
            return True
        (attempt / 'result_problem').write_text(problem)
    logger.error('FAILED %s after %.0f s: %s', video_id, seconds, attempt_outcome(attempt)['error'])
    return False


def ensure_config(args: argparse.Namespace, cohort: list[dict[str, Any]]) -> None:
    """Keep the original provenance; allow worker-count changes when resuming."""
    config = {'commit': checkout_commit(args.checkout),
              'detector_command': detector_command(args, Path('MANIFEST'), Path('OUTPUT_DIR')),
              'cohort': [{key: entry[key] for key in RUN_VIDEO_KEYS} for entry in cohort]}
    path = args.root / 'run_config.json.gz'
    if not path.exists():
        write_json(path, {**config, 'first_launch': now()})
        return
    recorded = read_json(path)
    # Workers change scheduling; retain the original command in the provenance file.
    def comparable_command(command: list[str]) -> list[str]:
        command = command.copy()
        command[command.index('--workers') + 1] = 'WORKERS'
        return command

    differing = []
    for key, current in config.items():
        previous = recorded[key]
        if key == 'detector_command':
            current, previous = comparable_command(current), comparable_command(previous)
        if current != previous:
            differing.append(key)
    if differing:
        raise SystemExit(f'{path} differs in {differing}. Earlier results would not match this launch; '
                         'use a new run root, or restore the recorded commit and options.')


def validate_results(root: Path) -> None:
    """Refuse incompatible completed outputs before starting any detector."""
    invalid = {}
    for path in sorted((root / 'videos').glob('*.json.gz')):
        problem = result_problem(path, path.name.removesuffix('.json.gz'))
        if problem is not None:
            invalid[path.name] = problem
    if invalid:
        raise SystemExit(f'videos/ holds results this run did not produce: {invalid}')


def ready(args: argparse.Namespace) -> int:
    """Check the handover without launching children or changing existing records."""
    for lane in read_lanes(args.root):
        if lane_alive(lane) is not False:
            raise SystemExit(f'lane {lane["lane"]} (pid {lane["pid"]}) is still running; stop it first')
    # Signalling only the runner PID can leave its detector alive.
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            command = (process / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if str(TRIAL_ENTRY).encode() in command and any(str(args.root).encode() in arg for arg in command):
            raise SystemExit(f'detector pid {process.name} is still running; stop the old process group first')
    ensure_config(args, read_json(args.cohort))
    validate_results(args.root)
    count = len(list((args.root / 'videos').glob('*.json.gz')))
    print(f'ready: preserving {count} completed videos; {args.lanes} lanes, {args.workers} workers each')
    return 0


def run(args: argparse.Namespace) -> int:
    """Claim unfinished videos from the shared cohort; skip completed or busy videos."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    cohort = read_json(args.cohort)
    ensure_config(args, cohort)
    for directory in ('lanes', 'attempts', 'videos', 'video_locks', *TRIAL_OUTPUTS):
        (args.root / directory).mkdir(exist_ok=True)
    # Held until this process exits, so a second launch of the same lane cannot start.
    lock = open(args.root / 'lanes' / f'lane-{args.lane}.lock', 'w')  # noqa: SIM115
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f'lane {args.lane} is already running; see `status`') from None
    # Static lanes do not claim video locks, so they cannot overlap shared-queue lanes.
    for lane in read_lanes(args.root):
        incompatible = lane['lanes'] != args.lanes or lane.get('scheduling') != 'shared'
        if incompatible and lane_alive(lane) is not False:
            raise SystemExit(f'lane {lane["lane"]} of {lane["lanes"]} (pid {lane["pid"]}) is still running; '
                             'stop it before changing lane count or scheduling')
    validate_results(args.root)
    pending = [entry for entry in cohort if not (args.root / 'videos' / f'{entry["id"]}.json.gz').exists()]
    lane = {'lane': args.lane, 'lanes': args.lanes, 'pid': os.getpid(), 'host': socket.gethostname(),
            'started_at': now(), 'lane_videos': len(cohort), 'pending_at_start': len(pending),
            'scheduling': 'shared',
            'workers': args.workers, 'cpus': sorted(os.sched_getaffinity(0))}
    write_lane(args, lane, state='running', current_id=None, current_attempt=None)
    logger.info('lane %d of %d: shared queue, %d pending', args.lane, args.lanes, len(pending))
    # SIGTERM from `kill -- -PGID` unwinds like Ctrl-C, so the lane records that it stopped.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    failures_in_a_row = 0
    failed_videos = 0
    try:
        for entry in pending:
            with open(args.root / 'video_locks' / f'{entry["id"]}.lock', 'a') as video_lock:
                try:
                    fcntl.flock(video_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                # Another lane may have completed this video after our initial scan.
                if (args.root / 'videos' / f'{entry["id"]}.json.gz').exists():
                    continue
                success = run_attempt(entry, args, lane)
            failures_in_a_row = 0 if success else failures_in_a_row + 1
            failed_videos += not success
            if failures_in_a_row == MAX_CONSECUTIVE_FAILURES:
                lane['state'] = 'stopped_after_failures'
                logger.error('stopping: %d videos failed in a row', failures_in_a_row)
                break
        else:
            lane['state'] = 'finished'
    except (KeyboardInterrupt, SystemExit):
        lane['state'] = 'stopped'
        logger.warning('stopped by a signal; the current video reruns on resume')
        raise
    finally:
        write_lane(args, lane, current_id=None, current_attempt=None, finished_at=now())
    return 1 if failed_videos else 0


def current_progress(attempt: Path) -> str:
    """The running video's phase from its log: setup and cuts, scene i/n, or pooling."""
    if not (attempt / 'run.log').exists():
        return 'starting'
    scene_progress = None
    pooling = False
    # Pooling output can push the final scene count beyond the usual log tail.
    with (attempt / 'run.log').open() as stream:
        for line in stream:
            if (match := SCENE_LINE.search(line.rstrip())) is not None:
                scene_progress = match.groups()
            elif 'pooling courts across scenes' in line:
                pooling = True
    if scene_progress is not None:
        completed, total, status = scene_progress
        if pooling:
            return f'scenes {completed}/{total} — pooling'
        return f'scene {completed}/{total} ({status})'
    return 'pooling' if pooling else 'setup or scene detection'


def scene_eta_minutes(phase: str, elapsed_seconds: float) -> float | None:
    """Extrapolate the remaining scene time; pooling has no measured progress rate."""
    match = re.match(r'^scene (\d+)/(\d+) ', phase)
    if match is None:
        return None
    completed, total = map(int, match.groups())
    return round(elapsed_seconds / completed * (total - completed) / 60, 1)


def status_report(root: Path, cohort: list[dict[str, Any]]) -> dict[str, Any]:
    """Every video's state from the run root's own files, plus lane liveness."""
    lanes = read_lanes(root)
    running = {}
    for lane in lanes:
        lane['alive'] = lane_alive(lane)
        if lane['alive'] and lane['current_id'] is not None:
            attempt = Path(lane['current_attempt'])
            elapsed_seconds = seconds_since(lane['current_started_at'])
            phase = current_progress(attempt)
            running[lane['current_id']] = {'lane': lane['lane'], 'phase': phase,
                                           'minutes': round(elapsed_seconds / 60, 1),
                                           'scene_eta_minutes': scene_eta_minutes(phase, elapsed_seconds)}
    videos, scene_statuses, pool_choices = {}, Counter(), Counter()
    video_seconds = []
    for entry in cohort:
        video_id = entry['id']
        attempt_dirs = sorted((root / 'attempts' / video_id).glob('*'))
        outcomes = [attempt_outcome(attempt) for attempt in attempt_dirs]
        record: dict[str, Any] = {'dataset': entry['dataset'], 'attempts': len(outcomes),
                                  'failed_attempts': [outcome for outcome in outcomes
                                                      if outcome['exit_code'] not in (None, 0) or outcome['error']]}
        result_path = root / 'videos' / f'{video_id}.json.gz'
        if result_path.exists():
            problem = result_problem(result_path, video_id)
            record['state'] = 'complete' if problem is None else 'invalid_result'
            if problem is None:
                result = read_json(result_path)
                record.update(output=str(result_path), scenes=len(result['scenes']),
                              seconds=round(result['total_seconds']),
                              trial_arm=(root / 'trial_videos' / f'{video_id}.json.gz').exists())
                video_seconds.append(result['total_seconds'])
                scene_statuses.update(row['status'] for row in result['scenes'])
                pool_choices.update(str(group['chosen_court']) for group in result.get('view_groups', []))
        elif video_id in running:
            record.update(state='running', **running[video_id])
        elif not outcomes:
            record['state'] = 'pending'
        elif outcomes[-1]['exit_code'] is None:
            record['state'] = 'interrupted'  # its runner died or was stopped mid-video; it reruns on resume
        else:
            record.update(state='failed', error=outcomes[-1]['error'])
        videos[video_id] = record
    states = Counter(record['state'] for record in videos.values())
    by_dataset = Counter(record['dataset'] for record in videos.values() if record['state'] == 'complete')
    config_path = root / 'run_config.json.gz'
    config = read_json(config_path) if config_path.exists() else {}
    left = len(cohort) - states['complete']
    live_lanes = sum(lane['alive'] is True for lane in lanes)
    eta_hours = None
    if video_seconds and live_lanes:
        eta_hours = round(sum(video_seconds) / len(video_seconds) * left / live_lanes / 3600, 1)
    return {'root': str(root), 'commit': config.get('commit'), 'first_launch': config.get('first_launch'),
            'hours_since_first_launch': round(seconds_since(config['first_launch']) / 3600, 1) if config else None,
            'videos_total': len(cohort), 'states': dict(states), 'complete_by_dataset': dict(by_dataset),
            'scenes_in_complete_videos': sum(scene_statuses.values()), 'scene_statuses': dict(scene_statuses),
            'view_group_choices': dict(pool_choices), 'rough_eta_hours': eta_hours,
            'lanes': lanes, 'videos': videos}


def print_status(report: dict[str, Any]) -> None:
    states = report['states']
    print(f'{report["root"]}  commit {report["commit"]}  first launch {report["first_launch"]} '
          f'({report["hours_since_first_launch"]} h ago)')
    print(f'videos: {states.get("complete", 0)}/{report["videos_total"]} complete {report["complete_by_dataset"]}; '
          f'states {states}')
    for lane in report['lanes']:
        alive = {True: 'alive', False: 'NOT RUNNING', None: 'other host'}[lane['alive']]
        print(f'lane {lane["lane"]}/{lane["lanes"]} pid {lane["pid"]} {alive}, state {lane["state"]}, '
              f'started {lane["started_at"]}')
    for video_id, record in report['videos'].items():
        if record['state'] == 'running':
            eta = record['scene_eta_minutes']
            eta_text = '' if eta is None else f', scene ETA ~{eta} min + pooling'
            print(f'running: {video_id} lane {record["lane"]}, {record["phase"]}, {record["minutes"]} min{eta_text}')
        elif record['state'] in ('failed', 'interrupted', 'invalid_result'):
            print(f'{record["state"]}: {video_id} ({record["attempts"]} attempts) {record.get("error") or ""}')
        elif record['state'] == 'complete' and record['failed_attempts']:
            print(f'complete after {len(record["failed_attempts"])} failed attempt(s): {video_id}')
    print(f'results: {report["root"]}/videos/<id>.json.gz; lane logs: {report["root"]}/lanes/')
    complete = [record for record in report['videos'].values() if record['state'] == 'complete']
    print(f'trial arm saved for {sum(record["trial_arm"] for record in complete)} of {len(complete)} complete videos')
    print(f'scenes in complete videos: {report["scenes_in_complete_videos"]} {report["scene_statuses"]}')
    print(f'view group choices: {report["view_group_choices"]}; rough ETA {report["rough_eta_hours"]} h '
          '(mean seconds per complete video x videos left / live lanes)')


def npy_xz_shape(path: Path) -> tuple[int, ...]:
    """A compressed .npy array's shape from its header, without decompressing the data."""
    import numpy as np

    with lzma.open(path) as stream:
        version = np.lib.format.read_magic(stream)
        read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        return read_header(stream)[0]


def check(args: argparse.Namespace) -> int:
    """Read-only preflight: every video and pose file, the checkout, the models and disk space."""
    cohort = read_json(args.cohort)
    problems = []
    slots = Counter()
    for entry in cohort:
        video, people = Path(entry['video']), Path(entry['people'])
        if not os.access(video, os.R_OK):
            problems.append(f'{entry["id"]}: video not readable: {video}')
        unreadable = [name for name in POSE_FILES if not os.access(people / name, os.R_OK)]
        if unreadable:
            problems.append(f'{entry["id"]}: pose files not readable in {people}: {unreadable}')
            continue
        pose_frames, pose_slots, _ = npy_xz_shape(people / 'pose_bboxes.npy.xz')
        slots[f'{entry["dataset"]}: {pose_slots} people per frame'] += 1
        # run_video rejects poses shorter than the video; ShuttleSet's tracked count finds that now.
        if entry['frame_count'] is not None and pose_frames < entry['frame_count']:
            problems.append(f'{entry["id"]}: poses cover {pose_frames} of {entry["frame_count"]} frames')
    print(f'cohort: {len(cohort)} videos {dict(Counter(entry["dataset"] for entry in cohort))}; {dict(slots)}')
    print(f'checkout {args.checkout}: commit {checkout_commit(args.checkout)}, no uncommitted tracked changes')
    weights = args.deeplsd / 'weights' / 'deeplsd_md.tar'
    for path in (args.python, args.deeplsd, weights):
        if not path.exists():
            problems.append(f'missing: {path}')
    existing = args.root / 'videos'
    print(f'run root {args.root}: {len(list(existing.glob("*.json.gz"))) if existing.exists() else 0} results; '
          f'{shutil.disk_usage(args.root).free / 1e9:.0f} GB free')
    for problem in problems:
        print(f'PROBLEM {problem}')
    print('check passed' if not problems else f'check failed: {len(problems)} problems')
    return 0 if not problems else 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('check', 'ready', 'run', 'status'))
    parser.add_argument('--root', type=Path, required=True, help='the run root, which holds cohort.json.gz')
    parser.add_argument('--cohort', type=Path, help='default: ROOT/cohort.json.gz')
    parser.add_argument('--checkout', type=Path, help='project checkout at the run commit (check, run)')
    parser.add_argument('--python', type=Path, help='court environment interpreter (check, run)')
    parser.add_argument('--deeplsd', type=Path, help='DeepLSD checkout holding weights/deeplsd_md.tar (check, run)')
    parser.add_argument('--lane', type=int, default=0, help='this worker lane identifier')
    parser.add_argument('--lanes', type=int, default=1)
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=8)
    parser.add_argument('--json', action='store_true', help='status: print the full report as JSON')
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.cohort = args.cohort or args.root / 'cohort.json.gz'
    if args.command != 'status' and None in (args.checkout, args.python, args.deeplsd):
        parser.error(f'{args.command} needs --checkout, --python and --deeplsd')
    if not 0 <= args.lane < args.lanes:
        parser.error('--lane must be in 0..LANES-1')
    return args


def main() -> int:
    args = parse_arguments()
    if args.command == 'check':
        return check(args)
    if args.command == 'run':
        return run(args)
    if args.command == 'ready':
        return ready(args)
    report = status_report(args.root, read_json(args.cohort))
    if args.json:
        print(json.dumps(report, indent=1))
    else:
        print_status(report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
