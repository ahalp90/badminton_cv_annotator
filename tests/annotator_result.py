"""Small annotation results for callers that do not inspect outcome calculations."""

from annotator.run_video import AnnotatorResult
from annotator.types import ContactCandidate


def empty_annotation(spans: list[tuple[int, int]], contacts: list[ContactCandidate]) -> AnnotatorResult:
    return AnnotatorResult(
        spans=spans, contacts=contacts, filtered_contacts=[], filtered_by_rally={},
        striker_halves=[], n_strokes_list=[], next_servers=[], fitted_first_all=[],
        verdict_rows={}, landings={}, geometric_verdict_rows={}, hit_height_by_frame={},
        hit_height_failures=[],
    )
