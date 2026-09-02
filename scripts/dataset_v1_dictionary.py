"""Render the frozen v1 data dictionary as markdown.

``src/dataset_builder/schema_v1.py`` is the only source. The output is pasted
into ``docs/dataset_v1_schema.md`` between the ``<!-- dictionary:start -->``
and ``<!-- dictionary:end -->`` markers, and ``tests/test_dataset_v1_schema_doc.py``
fails when the two drift apart.

Usage: ``uv run python scripts/dataset_v1_dictionary.py``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from dataset_builder.schema_v1 import (  # noqa: E402 (must follow the sys.path insertion)
    FEATURE_DISPOSITIONS,
    PLAYER_SIGNALS,
    PRIMITIVE_ARTIFACT_NOTES,
    TABLES,
    TableSpec,
)

SIGNALS_BLURB = (
    "One directory per video under `player_signals/<video_id>/`. Every array has "
    "one row per decoded frame, so a frame index reads across all four."
)
ARTIFACTS_BLURB = (
    "One note per artifact name that can appear in the `primitive_artifacts` "
    "table. The note is the reliability warning that travels with the file."
)
DISPOSITIONS_BLURB = (
    "Every trial feature and where it ended up. Exported columns are named as "
    "`table.column`. A feature with no columns is absent from v1."
)


def _row(cells: Sequence[str]) -> str:
    """Return one markdown table row, escaping pipes so cells never split."""
    escaped = (cell.replace("|", "\\|") for cell in cells)
    return f"| {' | '.join(escaped)} |"


def _section(
    heading: str, blurb: str, headers: Sequence[str], rows: Iterable[Sequence[str]]
) -> list[str]:
    """Return one heading, its blurb, and one markdown table."""
    lines = [
        f"### {heading}",
        "",
        blurb,
        "",
        _row(headers),
        _row(["---"] * len(headers)),
    ]
    lines.extend(_row(row) for row in rows)
    lines.append("")
    return lines


def _table_section(table: TableSpec) -> list[str]:
    """Return the heading, key line, and column table for one frozen table."""
    blurb = (
        f"File `{table.filename}`. Key `({', '.join(table.key)})`.\n\n"
        f"{table.description}"
    )
    return _section(
        table.name,
        blurb,
        ("Column", "Type", "Nullable", "Reliability", "Description"),
        (
            (
                f"`{column.name}`",
                column.type.value,
                "yes" if column.nullable else "no",
                column.reliability.value,
                column.description,
            )
            for column in table.columns
        ),
    )


def render_dictionary() -> str:
    """Return the whole generated dictionary as one markdown string."""
    lines: list[str] = []
    for table in TABLES:
        lines.extend(_table_section(table))
    lines.extend(
        _section(
            "Player-signal arrays",
            SIGNALS_BLURB,
            ("Array", "File", "Shape", "Dtype", "Reliability", "Description"),
            (
                (
                    f"`{signal.name}`",
                    f"`{signal.filename}`",
                    f"`{signal.shape}`",
                    signal.dtype,
                    signal.reliability.value,
                    signal.description,
                )
                for signal in PLAYER_SIGNALS
            ),
        )
    )
    lines.extend(
        _section(
            "Primitive artifact notes",
            ARTIFACTS_BLURB,
            ("Artifact", "Reliability", "Note"),
            (
                (f"`{note.artifact}`", note.reliability.value, note.note)
                for note in PRIMITIVE_ARTIFACT_NOTES
            ),
        )
    )
    lines.extend(
        _section(
            "Feature dispositions",
            DISPOSITIONS_BLURB,
            ("Feature", "Disposition", "Columns", "Reason"),
            (
                (
                    entry.feature,
                    entry.disposition.value,
                    ", ".join(f"`{name}`" for name in entry.columns) or "none",
                    entry.reason,
                )
                for entry in FEATURE_DISPOSITIONS
            ),
        )
    )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    """Print the dictionary to standard output."""
    print(render_dictionary(), end="")


if __name__ == "__main__":
    main()
