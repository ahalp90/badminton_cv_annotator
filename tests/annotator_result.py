"""Small annotation results for callers that do not inspect outcome calculations."""

from annotator.annotate_video import AnnotatorResult
from annotator.types import ContactCandidate, ContactSequence, FinishedRally


def empty_annotation(spans: list[tuple[int, int]], contacts: list[ContactCandidate]) -> AnnotatorResult:
    rallies = tuple(FinishedRally(ContactSequence(index, start, end, ()), None)
                    for index, (start, end) in enumerate(spans))
    return AnnotatorResult(rallies, {}, {}, {}, (), contacts, ())
