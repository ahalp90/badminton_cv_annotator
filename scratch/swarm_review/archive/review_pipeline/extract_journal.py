"""Extract lens-scan findings and probe outputs from the workflow run to packets/.

The journal's result entries carry only agentId + result, so each result is classified by
markers in its agent transcript's first entry (which holds the full prompt). Output:
  packets/<stem>__scans.json       — [{job, findings, overflow_note}] per module
  packets/<stem>__probe.json       — probe output dict
  packets/cross_findings.json      — cross-module scan results (audit-only)
"""
import json
import re
import sys
from pathlib import Path

WORK_DIR = Path(__file__).parent
RUN_DIR = Path(
    "/home/ariel/.claude/projects/-home-ariel-Documents-COSC594-badminton-cv-annotator/"
    "a6d291df-8e22-44dc-9ef4-079e438210a6/subagents/workflows/wf_7e64ca5f-87e"
)
LENS_MARKERS = {
    "NAMING AND ABSTRACTION only": "naming",
    "STRUCTURE only": "structure",
    "DOCS AND HOUSE STYLE only": "style",
}


def classify(agent_id: str) -> dict:
    transcript = RUN_DIR / f"agent-{agent_id}.jsonl"
    with open(transcript) as fh:
        first = json.loads(fh.readline())
    prompt = first["message"]["content"]
    if isinstance(prompt, list):  # content blocks
        prompt = " ".join(block.get("text", "") for block in prompt)
    if "Cold-read comprehension probe" in prompt:
        match = re.search(r'Cold-read comprehension probe\. Read "([^"]+)"', prompt)
        return {"kind": "probe", "module": match.group(1)}
    if "Chunk metadata:" in prompt:
        meta = json.loads(prompt.split("Chunk metadata: ", 1)[1].splitlines()[0])
        lens = next((v for k, v in LENS_MARKERS.items() if k in prompt), "unknown")
        return {"kind": "scan", "module": meta["file"], "chunk_id": meta["chunk_id"], "lens": lens}
    if "Cross-module readability scan" in prompt:
        return {"kind": "cross"}
    raise ValueError(f"unclassifiable agent transcript {transcript}")


scope = [line.strip() for line in (WORK_DIR / "scope_files.txt").read_text().splitlines() if line.strip()]
# Package-qualified stems: src/annotator/config.py -> annotator_config (plain stems collide).
stem_of = {p: p.removeprefix("src/").removesuffix(".py").replace("/", "_") for p in scope}
assert len(set(stem_of.values())) == len(scope), "qualified stems still collide in packets/"

scans: dict[str, list] = {p: [] for p in scope}
probes: dict[str, dict] = {}
cross: list[dict] = []
unmatched = 0

for raw in (RUN_DIR / "journal.jsonl").read_text().splitlines():
    entry = json.loads(raw)
    if entry.get("type") != "result" or not isinstance(entry.get("result"), dict):
        continue
    identity = classify(entry["agentId"])
    result = entry["result"]
    if identity["kind"] == "scan":
        if identity["module"] not in scans:
            unmatched += 1
            continue
        scans[identity["module"]].append({
            "job": f"scan:{identity['chunk_id']}:{identity['lens']}",
            "findings": result.get("findings", []),
            "overflow_note": result.get("overflow_note"),
        })
    elif identity["kind"] == "probe":
        probes[identity["module"]] = result
    else:
        cross.append(result)

packets = WORK_DIR / "packets"
packets.mkdir(exist_ok=True)
for path, jobs in scans.items():
    (packets / f"{stem_of[path]}__scans.json").write_text(json.dumps(jobs, indent=1))
for path, probe in probes.items():
    (packets / f"{stem_of[path]}__probe.json").write_text(json.dumps(probe, indent=1))
(packets / "cross_findings.json").write_text(json.dumps(cross, indent=1))

scan_jobs = sum(len(v) for v in scans.values())
finding_total = sum(len(j["findings"]) for jobs in scans.values() for j in jobs)
print(f"scan jobs extracted: {scan_jobs}; lens findings: {finding_total}; unmatched results: {unmatched}")
print(f"probes extracted: {len(probes)}/{len(scope)}")
print(f"cross groups extracted: {len(cross)}; cross findings: {sum(len(c.get('findings', [])) for c in cross)}")
short = sorted(f"{stem_of[p]}({len(jobs)})" for p, jobs in scans.items() if len(jobs) < 3)
if short:
    print(f"modules with <3 scan jobs: {short}", file=sys.stderr)
missing_probe = sorted(stem_of[p] for p in scope if p not in probes)
if missing_probe:
    print(f"MISSING probes: {missing_probe}", file=sys.stderr)
