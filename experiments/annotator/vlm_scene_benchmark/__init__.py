"""Recording-only contracts and scoring for the Issue 38 VLM benchmark."""

from .contracts import BenchmarkRunRecord, read_run_record, write_run_record
from .scoring import score_run_record, write_score_summary

__all__ = [
    "BenchmarkRunRecord",
    "read_run_record",
    "score_run_record",
    "write_run_record",
    "write_score_summary",
]
