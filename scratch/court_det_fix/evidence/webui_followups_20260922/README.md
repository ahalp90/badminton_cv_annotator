# WebUI follow-ups 1 and 2

The returned package is filed here unchanged. Start with [return.md](return.md).
It covers the far-end fit diagnostic and sparse-frame replay requested in
[WEBUI_FOLLOWUPS.md](../../WEBUI_FOLLOWUPS.md). Task 3, SVD search reduction,
is not part of this return.

The return reports a ranking lead for GX86088: existing alternatives reduce
far-strip clipping, but remain imperfect. Its sparse-frame replay reports
that shared paint scoring helps while a wall-selection failure survives three
samples. These are the return's conclusions; local numerical reproduction and
integration review remain pending.

## Contents

- [Original report](return.md), including the evaluation limits and rerun command.
- [Returned standalone script](run_followups12.py), preserved without edits.
- [Results](results/): compressed tables, detailed records and reported checks.
- [Figures](results/figures/): four supplied figures, preserved without further
  image evaluation during filing.

The supplied input revision is `b47eee0ff43ec7cbebcc2d84eb9e5e98075d4eea`.
The return states that this label was not independently re-fetched. Its rerun
instructions use the original archive directory names; a checkout containing
the named pinned inputs is also supported by the returned script.

## Filing checks

Received on 22 September 2026 from `followups12_completed.tar.gz`. All 15
regular files match their archive members byte-for-byte. All compressed JSON
and CSV outputs parse successfully. The runner has not been executed or
integrated locally; reported checks inside the return remain attributed to it.

The original archive is retained at
`.recovery/webui-followups12-return-20260922.tar.gz` relative to the investigation
root. No source content was removed.
