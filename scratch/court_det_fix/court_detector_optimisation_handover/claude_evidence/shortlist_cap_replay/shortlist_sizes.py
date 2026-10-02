"""How many courts each view's G0 and G1 overall shortlists hold, and how many parents reach scoring.

Usage: python shortlist_sizes.py <artefact dir>
"""

import gzip
import json
import sys
from pathlib import Path

print("view\tG0\tG1\tline templates\tparents\tgated courts")
for path in sorted(Path(sys.argv[1]).glob("*.json.gz")):
    with gzip.open(path) as handle:
        artefact = json.load(handle)
    populations = artefact["populations"]
    print("\t".join(str(value) for value in (
        path.name.removesuffix(".json.gz"), len(populations["G0"]), len(populations["G1"]),
        len(artefact["line_templates"]["entries"]), len(artefact["w5"]["record"]["parents"]),
        len(artefact["net_choice"]["rows"]))))
