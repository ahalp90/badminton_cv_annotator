"""Plain-text outcome reports for the rally-segmentation batch CLI."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class VideoOutcome:
    """The final batch outcome for one input shuttle-track file."""

    video_id: str
    status: str
    rallies: int = 0
    contacts: int = 0
    reason: str | None = None


def _count_word(count: int, singular: str, plural: str) -> str:
    return singular if count == 1 else plural


def _format_reason(reason: str) -> str:
    return re.sub(r'\s+', ' ', reason).strip()


def format_batch_report(
    outcomes: Sequence[VideoOutcome], *, all_excluded: bool = False,
) -> str:
    """Format the final report for outcomes in input-path order.

    :param outcomes: One outcome for each input shuttle-track file.
    :param all_excluded: Whether the doubles filter excluded every input video.
    :return: Report text without a trailing newline.
    """
    video_count = len(outcomes)
    processed_count = sum(outcome.status == 'processed' for outcome in outcomes)
    excluded_count = sum(outcome.status == 'excluded' for outcome in outcomes)
    skipped_count = sum(outcome.status == 'skipped' for outcome in outcomes)
    rally_count = sum(outcome.rallies for outcome in outcomes if outcome.status == 'processed')
    contact_count = sum(outcome.contacts for outcome in outcomes if outcome.status == 'processed')

    if video_count == 0:
        headline = 'batch completed: no input videos found'
    elif all_excluded:
        video_word = _count_word(video_count, 'video', 'videos')
        headline = f'batch failed: all {video_count} {video_word} excluded'
    elif processed_count == 0:
        video_word = _count_word(video_count, 'video', 'videos')
        headline = f'batch failed: 0 of {video_count} {video_word} processed'
    else:
        video_word = _count_word(video_count, 'video', 'videos')
        headline = f'batch completed: {processed_count} of {video_count} {video_word} processed'

    outcome_lines = []
    for outcome in outcomes:
        if outcome.status == 'processed':
            rally_word = _count_word(outcome.rallies, 'rally', 'rallies')
            contact_word = _count_word(outcome.contacts, 'contact', 'contacts')
            outcome_lines.append(
                f'- {outcome.video_id}: processed; '
                f'{outcome.rallies} {rally_word}; {outcome.contacts} {contact_word}'
            )
        else:
            reason = _format_reason(outcome.reason or '')
            outcome_lines.append(f'- {outcome.video_id}: {outcome.status}; {reason}')
    if not outcome_lines:
        outcome_lines.append('- none')

    return '\n\n'.join([
        headline,
        '\n'.join([
            'Counts',
            f'- videos processed: {processed_count}/{video_count}',
            f'- videos excluded before processing: {excluded_count}/{video_count}',
            f'- videos skipped: {skipped_count}/{video_count}',
            f'- rallies written: {rally_count}',
            f'- contacts written: {contact_count}',
        ]),
        '\n'.join(['Video outcomes', *outcome_lines]),
        '\n'.join([
            'Rally exclusions',
            '- not recorded by this CLI; rally-level reasons exist only in the run_video chain',
        ]),
    ])


def derive_batch_report_path(rally_spans_path: Path) -> Path:
    """Return the report path beside a configured rally-spans CSV path."""
    return rally_spans_path.with_name(f'{rally_spans_path.stem}_batch_report.txt')


def publish_batch_report(
    outcomes: Sequence[VideoOutcome], rally_spans_path: Path, *, all_excluded: bool = False,
) -> Path:
    """Write and print one report, returning the saved report path."""
    report_text = format_batch_report(outcomes, all_excluded=all_excluded)
    report_path = derive_batch_report_path(rally_spans_path)
    report_path.write_text(report_text + '\n', encoding='utf-8')
    print(report_text)
    return report_path
