"""Persistent review decisions, kept separate from court corner coordinates."""
from __future__ import annotations

import csv
import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path

# Labelled spans of the video. End frames are inclusive; an open span has end None.
PERIOD_LABELS = {'play': 'Play', 'jitter': 'Jitter', 'transition': 'Transition'}


@dataclass
class ReviewItem:
    frame: int
    reason: str
    status: str = 'pending'
    change_frame: int | None = None
    annotation_frame: int | None = None


class ReviewQueue:
    """Keep the selected candidate independent of the frame being inspected."""

    def __init__(self, path: Path, video: str, items: list[ReviewItem]) -> None:
        self.path = path
        self.video = video
        self.items = items
        self.markers: list[int] = []
        self.standalone_markers: list[int] = []
        self.periods: list[dict] = []
        self.notes = ''
        self.index = 0
        if path.exists():
            with gzip.open(path, 'rt') as source:
                saved = json.load(source)
            if saved['video'] != video or saved['version'] != 1:
                raise ValueError(f'Incompatible review file: {path}')
            previous = {item['frame']: item for item in saved['items']}
            for item in self.items:
                if item.frame in previous:
                    stored = previous[item.frame]
                    item.status = stored['status']
                    item.change_frame = stored['change_frame']
                    item.annotation_frame = stored['annotation_frame']
            # Preserve decisions for optional candidates when their filter changes.
            self.hidden = [item for frame, item in previous.items() if frame not in {entry.frame for entry in items}]
            self.markers = saved['camera_changes']
            associated = {item['change_frame'] for item in saved['items']}
            self.standalone_markers = saved.get('standalone_changes',
                                                [frame for frame in self.markers if frame not in associated])
            self.periods = saved.get('periods', [])
            self.notes = saved.get('notes', '')
        else:
            self.hidden = []
        self.index = next((index for index, item in enumerate(items) if self.needs_review(item)), 0)

    @staticmethod
    def needs_review(item: ReviewItem) -> bool:
        return item.status in ('pending', 'uncertain') or (item.status == 'changed' and item.annotation_frame is None)

    @property
    def current(self) -> ReviewItem:
        return self.items[self.index]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        with gzip.open(temporary, 'wt') as output:
            json.dump({'version': 1, 'video': self.video,
                       'items': [asdict(item) for item in self.items] + self.hidden,
                       'camera_changes': sorted(self.markers),
                       'standalone_changes': sorted(self.standalone_markers),
                       'periods': sorted(self.periods, key=lambda period: (period['start'], period['kind'])),
                       'notes': self.notes}, output, indent=2)
        temporary.replace(self.path)

    def move(self, direction: int, pending_only: bool = False) -> int:
        for offset in range(1, len(self.items) + 1):
            index = (self.index + direction * offset) % len(self.items)
            if not pending_only or self.needs_review(self.items[index]):
                self.index = index
                break
        return self.current.frame

    def detach_change(self) -> None:
        old_frame = self.current.change_frame
        self.current.change_frame = None
        retained = any(item.change_frame == old_frame for item in self.items)
        retained = retained or any(item['change_frame'] == old_frame for item in self.hidden)
        if old_frame in self.markers and old_frame not in self.standalone_markers and not retained:
            self.markers.remove(old_frame)

    def decide(self, status: str) -> None:
        self.detach_change()
        self.current.status = status
        self.save()

    def mark_change(self, frame: int, associate: bool = True) -> None:
        if associate:
            self.detach_change()
            if self.current.annotation_frame is not None and self.current.annotation_frame < frame:
                self.current.annotation_frame = None
            self.current.change_frame = frame
            self.current.status = 'changed'
        elif frame not in self.standalone_markers:
            self.standalone_markers.append(frame)
        if frame not in self.markers:
            self.markers.append(frame)
        self.save()

    def next_change(self, frame: int) -> int | None:
        ordered = sorted(self.markers)
        return next((marker for marker in ordered if marker > frame), ordered[0] if ordered else None)

    def remove_change(self, frame: int) -> bool:
        if frame not in self.markers:
            return False
        self.markers.remove(frame)
        if frame in self.standalone_markers:
            self.standalone_markers.remove(frame)
        for item in self.items:
            if item.change_frame == frame:
                item.change_frame = None
                item.status = 'pending'
        for item in self.hidden:
            if item['change_frame'] == frame:
                item['change_frame'] = None
                item['status'] = 'pending'
        self.save()
        return True

    def open_period(self, kind: str) -> dict | None:
        return next((period for period in self.periods if period['kind'] == kind and period['end'] is None), None)

    def overlapping_period(self, kind: str, start: int, end: int) -> dict | None:
        """A closed period of this kind sharing any frame with start..end, inclusive."""
        return next((period for period in self.periods if period['kind'] == kind and period['end'] is not None
                     and period['start'] <= end and start <= period['end']), None)

    def start_period(self, kind: str, frame: int) -> None:
        if self.open_period(kind) is not None:
            raise ValueError(f'{PERIOD_LABELS[kind]} already has an open period')
        if (clash := self.overlapping_period(kind, frame, frame)) is not None:
            raise ValueError(f'Frame {frame} is already inside the {PERIOD_LABELS[kind].lower()} period '
                             f'{clash["start"]}–{clash["end"]}')
        self.periods.append({'kind': kind, 'start': frame, 'end': None})
        self.save()

    def end_period(self, kind: str, frame: int) -> None:
        period = self.open_period(kind)
        if period is None:
            raise ValueError(f'{PERIOD_LABELS[kind]} has no open period')
        if frame < period['start']:
            raise ValueError(f'{PERIOD_LABELS[kind]} cannot end before its start at frame {period["start"]}')
        if (clash := self.overlapping_period(kind, period['start'], frame)) is not None:
            raise ValueError(f'Ending here would swallow the {PERIOD_LABELS[kind].lower()} period '
                             f'{clash["start"]}–{clash["end"]}')
        period['end'] = frame
        self.save()

    def add_shake(self, frame: int) -> None:
        """Record a momentary jolt as a one-frame jitter period."""
        self.start_period('jitter', frame)
        self.end_period('jitter', frame)

    def remove_period(self, period: dict) -> None:
        self.periods.remove(period)
        self.save()

    def set_notes(self, notes: str) -> None:
        self.notes = notes
        self.save()

    def annotated(self, frame: int) -> bool:
        if self.current.change_frame is not None and frame < self.current.change_frame:
            return False
        self.current.annotation_frame = frame
        if self.current.status in ('pending', 'uncertain'):
            self.current.status = 'annotated'
        self.save()
        return True


def load_candidates(path: Path | None, video: str, fps: float, total: int,
                    include_uncertain: bool = False) -> list[ReviewItem]:
    """Use nominal FPS to make editable starting points, merging flags within 3 s."""
    items = [ReviewItem(0, 'Initial court')]
    if path is None:
        return items
    with gzip.open(path, 'rt') as source:
        rows = list(csv.DictReader(source))
    rows.sort(key=lambda row: float(row['end_seconds']))
    last_seconds = -10.0
    for row in rows:
        if row['video'] != video:
            raise ValueError(f'Candidate video {row["video"]!r} does not match {video!r}')
        if row['flag'] == 'uncertain_transition' and not include_uncertain:
            continue
        seconds = float(row['end_seconds'])
        if seconds - last_seconds <= 3:
            continue
        frame = min(total - 1, max(0, round(max(0, seconds - 0.5) * fps)))
        if frame != items[-1].frame:
            items.append(ReviewItem(frame, row['flag'].replace('_', ' ')))
        last_seconds = seconds
    return items
