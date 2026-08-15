"""ShuttleSet22 source and whole-video extraction support."""

from shuttleset22.manifest import (
    EXPECTED_FPS,
    EXPECTED_MATCH_IDS,
    AnnotationCorpus,
    AnnotationMatch,
    ResolvedSource,
    SourceContext,
    SourceEntry,
    SourceKind,
    SourceManifest,
    annotation_corpus_sha256,
    load_source_context,
    resolve_sources,
)

__all__ = [
    "EXPECTED_FPS",
    "EXPECTED_MATCH_IDS",
    "AnnotationCorpus",
    "AnnotationMatch",
    "ResolvedSource",
    "SourceContext",
    "SourceEntry",
    "SourceKind",
    "SourceManifest",
    "annotation_corpus_sha256",
    "load_source_context",
    "resolve_sources",
]
