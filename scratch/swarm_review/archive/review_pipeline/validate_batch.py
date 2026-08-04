"""Validate one codex batch: verdict files parse, tally verdicts, report tokens.

Usage: python3 validate_batch.py <stem>...
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).parent
total = {}
probe_total = incidental_total = 0
failures = []
for stem in sys.argv[1:]:
    try:
        v = json.loads((BASE / "packets" / f"{stem}__verdict.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        failures.append(stem)
        print(f"{stem}: FAIL {e}")
        continue
    tally = {}
    for item in v.get("verdicts", []):
        tally[item.get("verdict")] = tally.get(item.get("verdict"), 0) + 1
        total[item.get("verdict")] = total.get(item.get("verdict"), 0) + 1
    probe_total += len(v.get("probe_findings") or [])
    incidental_total += len(v.get("incidental_notes") or [])
    tokens = "?"
    log_file = BASE / "codex_jobs" / f"{stem}.log"
    if log_file.exists() and "tokens used" in log_file.read_text():
        after = log_file.read_text().split("tokens used", 1)[1]
        for token in after.replace(",", "").split():
            if token.isdigit():
                tokens = int(token)
                break
    print(f"{stem}: verdicts={tally} probe_findings={len(v.get('probe_findings') or [])} "
          f"incidental={len(v.get('incidental_notes') or [])} tokens={tokens}")
print(f"batch tally: {total}; probe_findings={probe_total}; incidental={incidental_total}")
if failures:
    sys.exit(f"FAILED verdicts: {failures}")
