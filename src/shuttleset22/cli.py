"""Acquire and resume whole-video ShuttleSet22 vision extraction."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING

from dataset_builder.vision import save_json_gz
from shuttleset22.acquisition import (
    acquire_source,
    source_receipt_path,
    validate_source_receipt,
    write_source_receipt,
)
from shuttleset22.manifest import (
    EXPECTED_MATCH_IDS,
    ResolvedSource,
    SourceContext,
    SourceEntry,
    SourceKind,
    load_source_context,
    resolve_sources,
)

if TYPE_CHECKING:
    from shuttleset22.extraction import ExtractionSettings


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "shuttleset22" / "sources.toml"
ACQUISITION_REPORT = "acquisition_report.json.gz"
EXTRACTION_REPORT = "extraction_report.json.gz"
CORPUS_REPORT_SCHEMA = "shuttleset22-corpus-report/1"

AcquireSource = Callable[..., ResolvedSource]
ExtractSource = Callable[..., object]


@dataclass(frozen=True)
class WorkflowResult:
    """One completed corpus pass, including partial terminal outcomes."""

    report: Path
    statuses: tuple[Mapping[str, object], ...]

    @property
    def failures(self) -> tuple[Mapping[str, object], ...]:
        return tuple(status for status in self.statuses if status["outcome"] == "failed")

    def counts(self) -> dict[str, int]:
        values: dict[str, int] = {}
        for status in self.statuses:
            outcome = str(status["outcome"])
            values[outcome] = values.get(outcome, 0) + 1
        return values


def run_acquisition(
    context: SourceContext,
    *,
    source_root: Path,
    overlap_root: Path,
    video_ids: Sequence[int] | None = None,
    yt_dlp: Sequence[str] = ("yt-dlp",),
    acquire: AcquireSource = acquire_source,
) -> WorkflowResult:
    """Acquire or validate selected sources and persist source receipts."""
    root = Path(source_root)
    root.mkdir(parents=True, exist_ok=True)
    selected = _select_entries(context, video_ids)
    report = root / ACQUISITION_REPORT
    statuses: list[Mapping[str, object]] = [_pending_status(entry) for entry in selected]
    _write_report(report, context, "acquisition", selected, statuses)
    digests: dict[str, int] = {}
    for index, entry in enumerate(selected):
        if entry.kind is SourceKind.UNRESOLVED:
            status = _unavailable_status(entry)
        else:
            destination_existed = (
                entry.kind is SourceKind.DOWNLOAD
                and (root / entry.canonical_filename).is_file()
            )
            try:
                if entry.kind is SourceKind.DOWNLOAD:
                    source = acquire(
                        context,
                        entry.match_id,
                        source_root=root,
                        overlap_root=overlap_root,
                        yt_dlp=yt_dlp,
                    )
                else:
                    source = resolve_sources(
                        context,
                        source_root=root,
                        overlap_root=overlap_root,
                        video_ids=[entry.match_id],
                    )[0]
                _require_unique_source(source, digests)
                expected_receipt = source_receipt_path(root, source)
                if destination_existed or expected_receipt.exists() or expected_receipt.is_symlink():
                    receipt = validate_source_receipt(context, source, receipt_root=root)
                else:
                    receipt = write_source_receipt(context, source, receipt_root=root)
                status = _source_status(
                    source,
                    outcome="reused" if destination_existed or entry.kind is not SourceKind.DOWNLOAD else "processed",
                    receipt=receipt,
                    path_root=root,
                )
            except Exception as error:
                status = _failed_status(entry, error)
        statuses[index] = status
        _write_report(report, context, "acquisition", selected, statuses)
    return WorkflowResult(report, tuple(statuses))


def run_extraction(
    context: SourceContext,
    *,
    source_root: Path,
    overlap_root: Path,
    output_root: Path,
    settings: ExtractionSettings,
    video_ids: Sequence[int] | None = None,
    extract: ExtractSource | None = None,
) -> WorkflowResult:
    """Run or validate selected source extractions and persist progress."""
    if extract is None:
        from shuttleset22.extraction import extract_source

        extract = extract_source
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    selected = _select_entries(context, video_ids)
    report = root / EXTRACTION_REPORT
    statuses: list[Mapping[str, object]] = [_pending_status(entry) for entry in selected]
    _write_report(
        report,
        context,
        "extraction",
        selected,
        statuses,
        settings=settings.configuration(),
        source_commit=settings.source_commit,
    )
    digests: dict[str, int] = {}
    for index, entry in enumerate(selected):
        if entry.kind is SourceKind.UNRESOLVED:
            status = _unavailable_status(entry)
        else:
            try:
                source = resolve_sources(
                    context,
                    source_root=source_root,
                    overlap_root=overlap_root,
                    video_ids=[entry.match_id],
                )[0]
                _require_unique_source(source, digests)
                validate_source_receipt(context, source, receipt_root=source_root)
                result = extract(
                    source,
                    output_root=root,
                    settings=settings,
                )
                reused = _boolean_attribute(result, "reused")
                receipt = _path_attribute(result, "receipt")
                output_dir = _path_attribute(result, "output_dir")
                artifacts = _mapping_attribute(result, "artifacts")
                status = {
                    **_source_status(
                        source,
                        outcome="reused" if reused else "processed",
                        receipt=receipt,
                        path_root=root,
                    ),
                    "output_dir": _relative_report_path(root, output_dir),
                    "artifacts": {
                        str(name): _relative_report_path(root, Path(path))
                        for name, path in sorted(artifacts.items())
                    },
                }
            except Exception as error:
                status = _failed_status(entry, error)
        statuses[index] = status
        if _three_consecutive_identical_failures(statuses[: index + 1]):
            reason = "Stopped after three consecutive matches failed for the same reason."
            for pending_index in range(index + 1, len(selected)):
                statuses[pending_index] = _pending_status(
                    selected[pending_index],
                    reason=reason,
                )
            _write_report(
                report,
                context,
                "extraction",
                selected,
                statuses,
                settings=settings.configuration(),
                source_commit=settings.source_commit,
            )
            break
        _write_report(
            report,
            context,
            "extraction",
            selected,
            statuses,
            settings=settings.configuration(),
            source_commit=settings.source_commit,
        )
    return WorkflowResult(report, tuple(statuses))


def _select_entries(
    context: SourceContext,
    video_ids: Sequence[int] | None,
) -> tuple[SourceEntry, ...]:
    requested = tuple(EXPECTED_MATCH_IDS if video_ids is None else video_ids)
    if len(requested) != len(set(requested)):
        raise ValueError(f"requested ShuttleSet22 IDs contain duplicates: {requested}")
    entries = context.manifest.by_id()
    unknown = sorted(set(requested).difference(entries))
    if unknown:
        raise ValueError(f"requested ShuttleSet22 IDs are unknown: {unknown}")
    return tuple(entries[match_id] for match_id in requested)


def _require_unique_source(source: ResolvedSource, digests: dict[str, int]) -> None:
    duplicate_id = digests.get(source.integrity.md5)
    if duplicate_id is not None:
        raise ValueError(
            f"source content is duplicated between matches {duplicate_id} "
            f"and {source.entry.match_id}"
        )
    digests[source.integrity.md5] = source.entry.match_id


def _source_status(
    source: ResolvedSource,
    *,
    outcome: str,
    receipt: Path,
    path_root: Path,
) -> dict[str, object]:
    source_metadata = source.metadata.to_dict()
    source_metadata["source_path"] = source.metadata.source_path.name
    source_integrity = source.integrity.to_dict()
    source_integrity["path"] = source.metadata.source_path.name
    return {
        "id": source.entry.match_id,
        "video": source.entry.video,
        "source_kind": source.entry.kind.value,
        "outcome": outcome,
        "reason": None,
        "source_metadata": source_metadata,
        "source_integrity": source_integrity,
        "receipt": _relative_report_path(path_root, receipt),
    }


def _unavailable_status(entry: SourceEntry) -> dict[str, object]:
    return {
        "id": entry.match_id,
        "video": entry.video,
        "source_kind": entry.kind.value,
        "outcome": "unavailable",
        "reason": entry.unresolved_reason,
    }


def _pending_status(
    entry: SourceEntry,
    *,
    reason: str = "Not attempted yet.",
) -> dict[str, object]:
    return {
        "id": entry.match_id,
        "video": entry.video,
        "source_kind": entry.kind.value,
        "outcome": "pending",
        "reason": reason,
    }


def _failed_status(entry: SourceEntry, error: Exception) -> dict[str, object]:
    return {
        "id": entry.match_id,
        "video": entry.video,
        "source_kind": entry.kind.value,
        "outcome": "failed",
        "reason": f"{type(error).__name__}: {error}",
    }


def _three_consecutive_identical_failures(
    statuses: Sequence[Mapping[str, object]],
) -> bool:
    if len(statuses) < 3:
        return False
    recent = statuses[-3:]
    return all(status["outcome"] == "failed" for status in recent) and len(
        {status["reason"] for status in recent}
    ) == 1


def _write_report(
    path: Path,
    context: SourceContext,
    phase: str,
    selected: Sequence[SourceEntry],
    statuses: Sequence[Mapping[str, object]],
    *,
    settings: Mapping[str, object] | None = None,
    source_commit: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "schema": CORPUS_REPORT_SCHEMA,
        "phase": phase,
        "source_manifest": {
            "schema": context.manifest.schema,
            "upstream_commit": context.manifest.upstream_commit,
            "dataset_path": context.manifest.dataset_path,
            "annotation_sha256": context.manifest.annotation_sha256,
            "expected_fps": str(context.manifest.expected_fps),
        },
        "selected_ids": [entry.match_id for entry in selected],
        "statuses": list(statuses),
    }
    if settings is not None:
        payload["settings"] = dict(settings)
    if source_commit is not None:
        payload["source_commit"] = source_commit
    save_json_gz(path, payload)


def _clean_source_commit(repo_root: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0:
        raise RuntimeError(f"could not inspect tracked source state: {status.stderr.strip()}")
    if status.stdout.strip():
        raise RuntimeError("tracked files differ from HEAD; refusing to record a false source commit")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if revision.returncode != 0 or not revision.stdout.strip():
        raise RuntimeError(f"could not resolve source commit: {revision.stderr.strip()}")
    return revision.stdout.strip()


def _relative_report_path(root: Path, path: Path) -> str:
    resolved_root = Path(root).resolve(strict=True)
    resolved_path = Path(path).resolve(strict=True)
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"reported path {resolved_path} is outside report root {resolved_root}") from exc


def _add_context_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--shuttleset-match-csv", type=Path, required=True)
    parser.add_argument("--ids", type=int, nargs="+")


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--overlap-root", type=Path, required=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-sources", help="Validate source identities.")
    _add_context_arguments(validate)

    download = subparsers.add_parser("download", help="Acquire or validate source videos.")
    _add_context_arguments(download)
    _add_source_arguments(download)
    download.add_argument("--yt-dlp", default="yt-dlp")

    extraction = subparsers.add_parser("extract", help="Run or resume GPU extraction.")
    _add_context_arguments(extraction)
    _add_source_arguments(extraction)
    extraction.add_argument("--output-root", type=Path, required=True)
    extraction.add_argument(
        "--tracknet-dir",
        type=Path,
        default=REPO_ROOT / "src" / "shared" / "tracknetv3",
    )
    extraction.add_argument(
        "--tracknet-model",
        type=Path,
        default=REPO_ROOT / "src" / "shared" / "tracknetv3" / "ckpts" / "TrackNet_best.pt",
    )
    extraction.add_argument(
        "--inpaint-model",
        type=Path,
        default=REPO_ROOT / "src" / "shared" / "tracknetv3" / "ckpts" / "InpaintNet_best.pt",
    )
    active_python = Path(sys.prefix) / "bin" / "python"
    extraction.add_argument("--tracknet-python", type=Path, default=active_python)
    extraction.add_argument("--pose-interpreter", type=Path, default=active_python)
    extraction.add_argument("--tracknet-stride", type=int, choices=(1, 8), default=8)
    extraction.add_argument("--tracknet-workers", type=int, default=1)
    extraction.add_argument("--tracknet-batch-size", type=int, default=32)
    extraction.add_argument(
        "--tracknet-large-video",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    extraction.add_argument("--pose-shards", type=int, default=8)
    extraction.add_argument("--pose-n-max", type=int, default=16)
    extraction.add_argument("--pose-device", choices=("cpu", "cuda"), default="cuda")
    extraction.add_argument("--pose-decode-mode", choices=("seek", "scan"), default="seek")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ShuttleSet22 command-line workflow."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        context = load_source_context(
            arguments.manifest,
            arguments.dataset_root,
            arguments.shuttleset_match_csv,
        )
        if arguments.command == "validate-sources":
            selected = _select_entries(context, arguments.ids)
            counts = _entry_counts(selected)
            print(f"validated {len(selected)} source identities: {_format_counts(counts)}")
            return 0
        if arguments.command == "download":
            result = run_acquisition(
                context,
                source_root=arguments.source_root,
                overlap_root=arguments.overlap_root,
                video_ids=arguments.ids,
                yt_dlp=(arguments.yt_dlp,),
            )
        elif arguments.command == "extract":
            from shuttleset22.extraction import ExtractionSettings

            settings = ExtractionSettings(
                source_commit=_clean_source_commit(REPO_ROOT),
                tracknet_dir=arguments.tracknet_dir,
                tracknet_model=arguments.tracknet_model,
                inpaint_model=arguments.inpaint_model,
                tracknet_python=arguments.tracknet_python,
                pose_interpreter=arguments.pose_interpreter,
                tracknet_stride=arguments.tracknet_stride,
                tracknet_workers=arguments.tracknet_workers,
                tracknet_batch_size=arguments.tracknet_batch_size,
                tracknet_large_video=arguments.tracknet_large_video,
                pose_shards=arguments.pose_shards,
                pose_n_max=arguments.pose_n_max,
                pose_device=arguments.pose_device,
                pose_decode_mode=arguments.pose_decode_mode,
            )
            result = run_extraction(
                context,
                source_root=arguments.source_root,
                overlap_root=arguments.overlap_root,
                output_root=arguments.output_root,
                settings=settings,
                video_ids=arguments.ids,
            )
        else:
            parser.error(f"unsupported command: {arguments.command}")
    except Exception as error:
        print(f"ShuttleSet22 workflow failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"ShuttleSet22 {arguments.command} finished: {_format_counts(result.counts())}")
    print(f"report: {result.report}")
    for failure in result.failures:
        print(f"match {failure['id']} failed: {failure['reason']}", file=sys.stderr)
    return 1 if result.failures else 0


def _entry_counts(entries: Sequence[SourceEntry]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.kind.value] = counts.get(entry.kind.value, 0) + 1
    return counts


def _format_counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{name}={value}" for name, value in sorted(counts.items())) or "none"


def _boolean_attribute(value: object, name: str) -> bool:
    result = getattr(value, name, None)
    if not isinstance(result, bool):
        raise TypeError(f"extraction result {name} must be bool")
    return result


def _path_attribute(value: object, name: str) -> Path:
    result = getattr(value, name, None)
    if not isinstance(result, Path):
        raise TypeError(f"extraction result {name} must be Path")
    return result


def _mapping_attribute(value: object, name: str) -> Mapping[str, Path]:
    result = getattr(value, name, None)
    if not isinstance(result, Mapping):
        raise TypeError(f"extraction result {name} must be a mapping")
    for key, path in result.items():
        if not isinstance(key, str) or not isinstance(path, Path):
            raise TypeError(f"extraction result {name} entries must map str to Path")
    return result
