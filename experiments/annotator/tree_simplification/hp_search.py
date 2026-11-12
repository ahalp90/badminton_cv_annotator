"""Run the joint contact/selector grid, validation repeats and frozen test."""

from __future__ import annotations

import argparse
import multiprocessing
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))

# isort: split

from annotator.training.contact import ContactFitConfig
from experiments.annotator import development
from experiments.annotator.heuristic_only_tune.data import (
    SearchVideo,
    load_original_videos,
)
from experiments.annotator.heuristic_only_tune.run import (
    load_paths,
    read_json,
    write_json,
)
from experiments.annotator.tree_simplification.hp_search_workers import (
    Candidate,
    cache_contact_scores,
    check_deadline,
    contact_directory,
    fit_contact,
    fit_selector,
    initialise_cpu_worker,
    load_pools,
    pool_directory,
    prepare_pool,
    score_fold,
)
from experiments.annotator.tree_simplification.input_stages import _training_records

SEEDS = (0, 100, 200, 300, 400)
BASELINE = Candidate()
SCORING_RULE = 'monotrack-discovery-contact-endpoints-v1'


def grid(selector_rounds: tuple[int, ...] = (200, 400)) -> list[Candidate]:
    candidates = [Candidate(*values) for values in product(
        (12, 36), (31, 63), (180, 360), (0.75, 0.8, 0.85, 0.9), (15, 31, 63), selector_rounds,
    )]
    return sorted(candidates, key=lambda candidate: (candidate.changes, candidate.label))


def ranking(row: dict[str, Any], objective: str) -> tuple:
    scores, tighter = row['totals']['10'], row['totals']['5']
    if objective == 'rallies':
        measures = (scores['fully_correct_rallies'], scores['attributed_f1'], tighter['fully_correct_rallies'])
    else:
        measures = (scores['attributed_f1'], scores['fully_correct_rallies'], tighter['attributed_f1'])
    candidate = Candidate(**row['candidate'])
    return (*(-value for value in measures), candidate.changes, candidate.label)


def shortlist(rows: list[dict[str, Any]]) -> list[Candidate]:
    selected = {BASELINE.label: BASELINE}
    for objective in ('rallies', 'f1'):
        for row in sorted(rows, key=lambda record: ranking(record, objective))[:2]:
            candidate = Candidate(**row['candidate'])
            selected[candidate.label] = candidate
    return list(selected.values())


class Progress:
    def __init__(self, root: Path, started: float) -> None:
        self.path = root / 'progress.json.gz'
        self.record: dict[str, Any] = {'status': 'running', 'started_at': started, 'completed': 0, 'total': 0}

    def update(self, **changes: Any) -> None:
        self.record.update(changes, updated_at=time.time())
        write_json(self.path, self.record)

    def drain(self, messages: Queue) -> None:
        while True:
            try:
                self.update(message=messages.get_nowait())
            except Empty:
                return


class CircularQueue:
    """Reserve bounded slots before preparing a family; release after scoring."""

    def __init__(self, capacity: int) -> None:
        self.slots: list[Any] = [None] * capacity
        self.free: Queue[int] = Queue(capacity)
        self.ready: Queue[int] = Queue(capacity)
        for slot in range(capacity):
            self.free.put(slot)

    def reserve(self, deadline: float, stopped: threading.Event) -> int:
        while not stopped.is_set():
            check_deadline(deadline)
            try:
                return self.free.get(timeout=1)
            except Empty:
                continue
        raise TimeoutError('Search preparation stopped')

    def publish(self, slot: int, value: Any) -> None:
        self.slots[slot] = value
        self.ready.put(slot)

    def release(self, slot: int) -> None:
        self.slots[slot] = None
        self.free.put(slot)


def bounded_jobs(executor: Executor, function: Any, arguments: list[tuple], limit: int, deadline: float) -> None:
    pending: set[Future] = set()
    remaining = iter(arguments)
    while True:
        check_deadline(deadline)
        while len(pending) < limit:
            try:
                values = next(remaining)
            except StopIteration:
                break
            pending.add(executor.submit(function, *values))
        if not pending:
            return
        done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
        for future in done:
            future.result()


def prepare_inputs(root: Path, paths: Path, windows: list[int], workers: int, progress: Progress, deadline: float) -> None:
    progress.update(stage='preparation', completed=0, total=len(windows), active=[], message='Preparing saved evidence')
    (root / 'candidates').mkdir(parents=True, exist_ok=True)
    for count, window in enumerate(windows, 1):
        check_deadline(deadline)
        candidate = root / 'candidates' / f'w{window}.json.gz'
        write_json(candidate, {
            'schema': 'annotator-training-candidates/1', 'stage': 'fixed-preprocessing',
            'candidates': [{'label': f'w{window}', 'overrides_base30': {} if window == 12 else {
                'body_unit_half_window': window,
            }, 'strategies': {}}],
        })
        subprocess.run([
            sys.executable, '-m', 'experiments.annotator.tree_simplification.prepare',
            '--paths', str(paths), '--output', str(root / 'prepared' / f'w{window}'),
            '--candidate', str(candidate), '--workers', str(workers),
        ], check=True, timeout=max(1, deadline - time.time()))
        progress.update(completed=count, message=f'Prepared body window {window}')


class Search:
    def __init__(
        self, root: Path, videos: tuple[SearchVideo, ...], cpu: ProcessPoolExecutor, gpu: ThreadPoolExecutor,
        workers: int, gpu_fits: int, slots: int, device: str, deadline: float, progress: Progress,
    ) -> None:
        self.root, self.videos = root, videos
        self.development = development.development_for(videos)
        self.by_id = {video.identity: video for video in videos}
        self.training_ids = self.development.id_order(self.development.videos(development.TRAINING_GROUPS))
        self.validation_ids = self.development.id_order(self.development.videos((development.VALIDATION_GROUP,)))
        self.cpu, self.gpu = cpu, gpu
        self.workers, self.gpu_fits, self.slot_count = workers, gpu_fits, slots
        self.device, self.deadline, self.progress = device, deadline, progress
        self.messages: Queue[str] = Queue()
        self.contact_records: dict[int, dict] = {}
        self.best: dict[str, Any] | None = None

    def contact_fits(self, candidate: Candidate, seed: int, include_full: bool) -> None:
        prepared = self.root / 'prepared' / f'w{candidate.body_window}'
        directory = contact_directory(self.root, candidate, seed)
        names = development.contact_fit_videos(self.development)
        names.pop('final_40')
        if not include_full:
            names.pop('v_32')
        missing = {name: order for name, order in names.items() if not (
            (directory / f'{name}.joblib').exists() and (directory / f'{name}.json.gz').exists()
        )}
        if not missing:
            return
        if candidate.body_window not in self.contact_records:
            inputs = read_json(prepared / 'preparation.json.gz')['inputs']
            self.contact_records[candidate.body_window] = _training_records(prepared, inputs, self.development)
        records = self.contact_records[candidate.body_window]
        config = replace(ContactFitConfig(), max_leaf_nodes=candidate.contact_leaves, max_iter=candidate.contact_rounds,
                         random_seed=20260824 + seed, device=self.device)
        jobs = []
        for name, sampling_order in missing.items():
            jobs.append((records, sampling_order, self.development.id_order(sampling_order), config,
                         directory / f'{name}.joblib', self.deadline))
        self.messages.put(f'Fitting {candidate.contact_label}, seed {seed}: {len(jobs)} contact models')
        bounded_jobs(self.gpu, fit_contact, jobs, self.gpu_fits, self.deadline)
        self.messages.put(f'Contact models ready: {candidate.contact_label}, seed {seed}')

    def prepare_family(self, candidate: Candidate, seed: int, validation: bool) -> None:
        self.contact_fits(candidate, seed, include_full=validation)
        identities = (*self.training_ids, *self.validation_ids) if validation else self.training_ids
        prepared = self.root / 'prepared' / f'w{candidate.body_window}' / 'videos'
        contacts = contact_directory(self.root, candidate, seed)
        jobs = []
        for identity in identities:
            group = self.development.group_by_video[identity]
            name = 'v_32' if group == development.VALIDATION_GROUP else f'held_{group}'
            destination = contacts / 'scores' / f'{identity}.joblib'
            if not destination.exists():
                jobs.append((prepared / f'{identity}.joblib', contacts / f'{name}.joblib', destination, self.deadline))
        bounded_jobs(self.cpu, cache_contact_scores, jobs, self.workers * 2, self.deadline)
        jobs = []
        for identity in identities:
            split = self.by_id[identity].split
            destination = pool_directory(self.root, candidate, seed, split) / f'{identity}.joblib'
            if not destination.exists() or not destination.with_suffix('.json.gz').exists():
                jobs.append((prepared / f'{identity}.joblib', contacts / 'scores' / f'{identity}.joblib',
                             candidate.cutoff, destination, self.deadline))
        self.messages.put(f'Preparing proposals: {candidate.family_label}, seed {seed}')
        bounded_jobs(self.cpu, prepare_pool, jobs, self.workers * 2, self.deadline)

    def result_path(self, candidate: Candidate, seed: int, validation: bool) -> Path:
        if validation:
            return self.root / 'validation' / candidate.label / f'seed-{seed}' / 'result.json.gz'
        return self.root / 'results' / 'screen' / f'{candidate.label}.json.gz'

    def producer(
        self, families: list[tuple[list[Candidate], int]], queue: CircularQueue, stopped: threading.Event,
        validation: bool,
    ) -> None:
        for candidates, seed in families:
            slot = queue.reserve(self.deadline, stopped)
            try:
                self.prepare_family(candidates[0], seed, validation)
                queue.publish(slot, (candidates, seed))
            except BaseException:
                queue.release(slot)
                raise

    def run(self, candidates: list[Candidate], seeds: tuple[int, ...], validation: bool = False) -> list[dict[str, Any]]:
        rows = []
        families: dict[tuple[str, int], list[Candidate]] = {}
        for seed in seeds:
            for candidate in candidates:
                path = self.result_path(candidate, seed, validation)
                if path.exists():
                    rows.append(read_json(path))
                else:
                    families.setdefault((candidate.family_label, seed), []).append(candidate)
        self.best = min(rows, key=lambda row: ranking(row, 'rallies')) if rows else None
        planned = [(selected, seed) for (_family, seed), selected in families.items()]
        fold_count = 1 if validation else 4
        self.progress.update(stage='validation' if validation else 'screen', completed=len(rows),
                             total=len(candidates) * len(seeds), fits_completed=len(rows) * fold_count,
                             fits_total=len(candidates) * len(seeds) * fold_count, active=[])
        queue, stopped = CircularQueue(self.slot_count), threading.Event()
        with ThreadPoolExecutor(max_workers=1) as preparation:
            producer = preparation.submit(self.producer, planned, queue, stopped, validation)
            try:
                for _family in planned:
                    while True:
                        check_deadline(self.deadline)
                        self.progress.drain(self.messages)
                        try:
                            slot = queue.ready.get(timeout=1)
                            break
                        except Empty:
                            if producer.done():
                                producer.result()
                    selected, seed = queue.slots[slot]
                    self.progress.update(active=[f'{selected[0].family_label}, seed {seed}'])
                    rows.extend(self.run_family(selected, seed, validation))
                    queue.release(slot)
                producer.result()
            finally:
                stopped.set()
        return rows

    def run_family(self, candidates: list[Candidate], seed: int, validation: bool) -> list[dict[str, Any]]:
        directory = pool_directory(self.root, candidates[0], seed, 'train')
        pools = load_pools(directory, self.training_ids)
        jobs = deque()
        for candidate in candidates:
            held_groups = ('V',) if validation else development.TRAINING_GROUPS
            for held in held_groups:
                fitting = tuple(group for group in development.TRAINING_GROUPS if group != held)
                videos = tuple(video for video in self.videos if self.development.group_by_video[video.identity] == held)
                jobs.append((candidate, held, fitting, videos))
        fitting: dict[Future, tuple] = {}
        scoring: dict[Future, tuple] = {}
        per_candidate: dict[str, list[dict]] = {candidate.label: [] for candidate in candidates}
        results = []
        while jobs or fitting or scoring:
            check_deadline(self.deadline)
            self.progress.drain(self.messages)
            while jobs and len(fitting) < self.gpu_fits and len(scoring) < self.workers * 2:
                job = jobs.popleft()
                candidate, _held, groups, _videos = job
                future = self.gpu.submit(fit_selector, pools, candidate, seed, groups,
                                         self.root, self.device, self.deadline)
                fitting[future] = job
            done, _pending = wait((*fitting, *scoring), timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                if future in fitting:
                    candidate, held, _groups, videos = fitting.pop(future)
                    model = future.result()
                    self.progress.update(fits_completed=self.progress.record['fits_completed'] + 1,
                                         message=f'Fitted {candidate.label}, seed {seed}, held {held}')
                    split = 'validation' if validation else 'train'
                    output = self.result_path(candidate, seed, validation).parent / candidate.label / f'seed-{seed}' / held
                    scored = self.cpu.submit(score_fold, model, pool_directory(self.root, candidate, seed, split),
                                             videos, output, self.deadline)
                    scoring[scored] = (candidate, held)
                else:
                    candidate, _held = scoring.pop(future)
                    per_candidate[candidate.label].append(future.result())
                    expected = 1 if validation else 4
                    if len(per_candidate[candidate.label]) == expected:
                        result = self.save_result(candidate, seed, per_candidate[candidate.label], validation)
                        results.append(result)
        return results

    def save_result(self, candidate: Candidate, seed: int, folds: list[dict], validation: bool) -> dict[str, Any]:
        per_video = {}
        for fold in folds:
            per_video.update(fold['per_video'])
        split = 'validation' if validation else 'train'
        directory = pool_directory(self.root, candidate, seed, split)
        before = {identity: read_json(directory / f'{identity}.json.gz')['before'] for identity in per_video}
        result = {'candidate': asdict(candidate), 'label': candidate.label, 'seed': seed,
                  'per_video': per_video, 'totals': development.pooled_scores(per_video),
                  'before': development.pooled_scores(before)}
        path = self.result_path(candidate, seed, validation)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, result)
        if self.best is None or ranking(result, 'rallies') < ranking(self.best, 'rallies'):
            self.best = result
        self.progress.update(completed=self.progress.record['completed'] + 1,
                             message=f'Scored {candidate.label}, seed {seed}',
                             best_label=self.best['label'],
                             best_rallies=self.best['totals']['10']['fully_correct_rallies'],
                             best_player_f1=self.best['totals']['10']['attributed_f1'])
        return result


def validation_means(candidates: list[Candidate], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    means = []
    for candidate in candidates:
        selected = [record for record in records if record['label'] == candidate.label]
        totals = {}
        for tolerance in ('5', '10'):
            totals[tolerance] = {
                name: sum(record['totals'][tolerance][name] for record in selected) / len(SEEDS)
                for name in selected[0]['totals'][tolerance]
            }
        means.append({'candidate': asdict(candidate), 'label': candidate.label, 'totals': totals, 'seeds': selected})
    return means


def run(args: argparse.Namespace, progress: Progress, deadline: float) -> None:
    candidates = grid(tuple(args.selector_rounds))
    if args.smoke:
        candidates = [BASELINE, replace(BASELINE, cutoff=0.75)]
    paths = load_paths(args.paths)
    videos = load_original_videos(paths['shots_master'])
    prepare_inputs(args.output, args.paths, sorted({candidate.body_window for candidate in candidates}),
                   args.workers, progress, deadline)
    with threadpool_limits(limits=2), ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'), initializer=initialise_cpu_worker,
    ) as cpu, ThreadPoolExecutor(max_workers=args.gpu_fits) as gpu:
        search = Search(args.output, videos, cpu, gpu, args.workers, args.gpu_fits, args.slots,
                        args.device, deadline, progress)
        screened = search.run(candidates, (0,))
        if args.smoke:
            progress.update(status='complete', stage='smoke', active=[], message='Two-cutoff smoke completed')
            return
        write_json(args.output / 'screen-summary.json.gz', screened)
        shortlist_path = args.output / 'shortlist.json.gz'
        if shortlist_path.exists():
            selected = [Candidate(**row) for row in read_json(shortlist_path)]
        else:
            selected = shortlist(screened)
            write_json(shortlist_path, [asdict(candidate) for candidate in selected])
        validation = search.run(selected, SEEDS, validation=True)
        means = validation_means(selected, validation)
        write_json(args.output / 'validation-summary.json.gz', means)
        winners = {BASELINE.label: BASELINE}
        for objective in ('rallies', 'f1'):
            winner = min(means, key=lambda row: ranking(row, objective))
            winners[winner['label']] = Candidate(**winner['candidate'])
        write_json(args.output / 'selection.json.gz', {
            'candidates': [asdict(candidate) for candidate in winners.values()], 'validation': means,
        })
    progress.update(status='complete', active=[], message='Selector search and validation repeats completed')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=24)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--gpu-fits', type=int, choices=(1, 2), default=2)
    parser.add_argument('--slots', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--selector-rounds', nargs='+', type=int, choices=(200, 400), default=[200, 400])
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--smoke', action='store_true', help='Screen baseline at .90 and .75; reuse outputs in the full run')
    args = parser.parse_args()
    if args.hours <= 0 or args.workers < 1 or 400 not in args.selector_rounds:
        parser.error('hours/workers must be positive and selector rounds must include baseline 400')
    args.selector_rounds = sorted(set(args.selector_rounds))
    args.output, args.paths = args.output.resolve(), args.paths.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {'paths': {name: str(path) for name, path in load_paths(args.paths).items()},
                'candidates': [asdict(candidate) for candidate in grid(tuple(args.selector_rounds))],
                'device': args.device, 'hours': args.hours, 'scoring_rule': SCORING_RULE}
    started = time.time()
    if not args.smoke:
        record_path = args.output / 'run.json.gz'
        if record_path.exists():
            saved = read_json(record_path)
            if saved['settings'] != settings:
                raise ValueError('Run settings changed; choose a fresh output directory')
            started = saved['started_at']
        else:
            write_json(record_path, {'settings': settings, 'started_at': started})
    progress = Progress(args.output, started)
    try:
        run(args, progress, started + args.hours * 3600)
    except (TimeoutError, subprocess.TimeoutExpired) as error:
        progress.update(status='time_limit', message=str(error), active=[])
        raise SystemExit(124) from error
    except BaseException as error:
        progress.update(status='failed', message=f'{type(error).__name__}: {error}', active=[])
        raise


if __name__ == '__main__':
    main()
