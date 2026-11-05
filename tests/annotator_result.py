"""Small annotation results for callers that do not inspect outcome calculations."""

from annotator.run_video import AnnotatorResult
from annotator.sequence.contacts import ContactSequence, FinishedRally
from annotator.types import ContactCandidate


def empty_annotation(spans: list[tuple[int, int]], contacts: list[ContactCandidate]) -> AnnotatorResult:
    rallies = tuple(FinishedRally(ContactSequence(index, start, end, ()), None)
                    for index, (start, end) in enumerate(spans))
    return AnnotatorResult(rallies, {}, {}, {}, (), contacts, ())
