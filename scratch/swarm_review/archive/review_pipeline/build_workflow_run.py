"""Generate workflow_run.js: embed per-module args into the adapted swarm template."""
import json
from pathlib import Path

WORK_DIR = Path(__file__).parent
RUBRIC_PATH = "/home/ariel/.claude/skills/swarm-review/rubric.md"

ORIENTATION = (
    "Badminton video analysis pipeline, a student research project. src/scraper collects "
    "source material: numbered stage scripts index candidate YouTube match videos, fetch "
    "transcripts, triage them, then download, clean, and pair accepted videos. "
    "src/annotator turns a match video into structured annotations: court detection "
    "evidence, replay/dead/composition frame masks, rally segmentation, and point-winner "
    "inference, with run_video.py and e2e_court_annotator.py as batch entry points. "
    "annotator/calibration scores annotator output against ground truth and sweeps "
    "parameter candidates to select settings. Each package has a config module holding "
    "tunable constants."
)


def metrics_excerpt(mod: dict) -> str:
    lines = [f"total_lines={mod['total_lines']}, code_lines={mod['code_lines']}"]
    lines.append("top_level_order:")
    lines.extend(f"  {entry}" for entry in mod["top_level_order"])
    flagged = [f for f in mod["functions"] if f["over_100"] or f["over_depth"]]
    if flagged:
        lines.append("threshold_flagged_functions (already recorded, do not re-report):")
        for f in flagged:
            marks = []
            if f["over_100"]:
                marks.append(f"{f['code_lines']} code lines ({f['lines'][0]}-{f['lines'][1]})")
            if f["over_depth"]:
                marks.append(f"depth {f['max_depth']} at line {f['deepest_line']}")
            lines.append(f"  {f['qualname']}: {'; '.join(marks)}")
    if mod["semicolon_joined_lines"]:
        lines.append(f"semicolon_joined_lines: {mod['semicolon_joined_lines']}")
    return "\n".join(lines)


def def_starts(mod: dict) -> list[int]:
    starts = []
    for entry in mod["top_level_order"]:
        line_str, _, label = entry.partition(": ")
        if label.startswith(("def ", "class ")):
            starts.append(int(line_str))
    return starts


def build_chunks(mod: dict) -> list[dict]:
    total = mod["total_lines"]
    if total <= 1000:
        return [{"id": "whole", "start": 1, "end": total}]
    starts = def_starts(mod)
    boundary = min(starts, key=lambda s: abs(s - total // 2))
    context_end = min(starts) - 1  # preamble: everything before the first def/class
    return [
        {"id": "chunk-1", "start": 1, "end": boundary - 1},
        {"id": "chunk-2", "start": boundary, "end": total, "context_end": context_end},
    ]


metrics = json.loads((WORK_DIR / "metrics.json").read_text())
modules = []
for mod in metrics["modules"]:
    modules.append({
        "path": mod["path"],
        "metrics": metrics_excerpt(mod),
        "chunks": build_chunks(mod),
    })

paths = [m["path"] for m in modules]
cross_groups = [
    sorted(p for p in paths if p.startswith("src/annotator/") and "/calibration/" not in p),
    sorted(p for p in paths if "/calibration/" in p),
    sorted(p for p in paths if p.startswith("src/scraper/")),
]

args = {
    "rubricPath": RUBRIC_PATH,
    "orientation": ORIENTATION,
    "modules": modules,
    "crossGroups": cross_groups,
}

template = (WORK_DIR / "workflow_template_adapted.js").read_text()
script = template.replace("__ARGS_JSON__", json.dumps(args, indent=1))
(WORK_DIR / "workflow_run.js").write_text(script)

chunk_total = sum(len(m["chunks"]) for m in modules)
split = [(m["path"], m["chunks"]) for m in modules if len(m["chunks"]) > 1]
print(f"modules={len(modules)} chunks={chunk_total} lens_jobs={chunk_total * 3} "
      f"probes={len(modules)} cross_groups={len(cross_groups)}")
for path, chunks in split:
    print(f"split {path}: " + "; ".join(
        f"{c['id']} {c['start']}-{c['end']}" + (f" ctx<= {c['context_end']}" if "context_end" in c else "")
        for c in chunks))
print(f"script bytes: {(WORK_DIR / 'workflow_run.js').stat().st_size}")
