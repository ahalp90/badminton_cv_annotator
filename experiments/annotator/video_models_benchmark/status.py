"""Show training progress without importing PyTorch; optionally watch in one session."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def show(output: Path) -> bool:
    status_path = output / "status.json.gz"
    exit_path = output / "exit-code.txt"
    state = {}
    if status_path.exists():
        with gzip.open(status_path, "rt") as stream:
            state = json.load(stream)
    phase = state.get("phase", "starting")
    finished = False
    if exit_path.exists():
        exit_code = int(exit_path.read_text())
        phase = "complete" if exit_code == 0 and phase == "complete" else f"stopped (exit {exit_code})"
        finished = True
    elif "pid" in state:
        try:
            os.kill(state["pid"], 0)
        except ProcessLookupError:
            phase, finished = "stopped (no exit code)", True
    if state.get("stage") == "continuation":
        print(f"{phase.upper()} | update {state.get('step', '—')} | continuation")
        print(f"Checks without improvement: {state.get('bad_checks', 0)}/{state['patience']}")
    else:
        print(f"{phase.upper()} | update {state.get('step', '—')}/{state.get('max_steps', '—')}")
    completed, total = state.get("completed", 0), state.get("total", 0)
    if total:
        elapsed = time.time() - state["phase_started"]
        remaining = elapsed * (total - completed) / completed if completed else None
        eta = f"{remaining / 60:.1f} min" if remaining is not None else "estimating"
        print(f"Current phase: {completed:,}/{total:,} batches ({100 * completed / total:.1f}%) | ETA {eta}")
    print(f"LR {state.get('lr', '—')} | train CE {state.get('train_loss', '—')} | val CE {state.get('val_loss', '—')}")
    print(f"Best val CE {state.get('best_loss', '—')}")
    if "updated_at" in state:
        print(f"Status age {time.time() - state['updated_at']:.0f}s | session elapsed {(time.time() - state['session_started']) / 3600:.2f}h")
    print(f"Run: {output}\nLog: {output / 'train.log'}\nRecovery: {output / 'last.pt'}")
    if phase.startswith(("failed", "stopped")) and (output / "train.log").exists():
        print("\n".join((output / "train.log").read_text().splitlines()[-15:]))
    subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"], check=True)
    return finished


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        if args.watch and sys.stdout.isatty():
            print("\033[H\033[J", end="")
        finished = show(args.output)
        if not args.watch or finished:
            break
        time.sleep(15)


if __name__ == "__main__":
    main()
