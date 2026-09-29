# Local follow-up replay

Both returned tasks reproduce locally. The selected IDs, eligibility lists,
counts and reported conclusions match. The largest numerical difference is
5.83e-11; this is replay precision, not a court-acceptance threshold.

Run on 22 September 2026 from the repository root, exit 0:

```bash
~/.venvs/badminton-cicd/bin/python \
  scratch/court_det_fix/evidence/webui_followups_20260922/run_followups12.py \
  --inputs . \
  --output scratch/court_det_fix/evidence/webui_followups_20260922/local_replay/results \
  --task all
```

The returned script ran unchanged with one numerical thread. This small cached
replay completed locally without a parallel runner. It produced all nine data
files; figures were neither regenerated nor reviewed. Six workers remain the
setting for independent detector cases.

## Verification

[Verification records](verification.json.gz) identify the environment, all
19 numerical inputs and the per-file comparison. Every numerical input is
byte-identical to its blob at `b47eee0ff43ec7cbebcc2d84eb9e5e98075d4eea`.

Four outputs match after decompression exactly: the task-1 oracle scan,
task-1 per-frame table, task-2 decision locks and task-2 distributions.
Other data outputs differ only in floating-point values, by at most 5.83e-11.
The checks record omits `bundle_files: 31` because this replay used the checkout
rather than the collector archive. Pinned input comparison was performed
separately. Compressed file bytes differ and are not claimed to match.

The script's synthetic checks, 78 published count/winner checks and 66 sparse
selection replays pass. There are 290 locked decisions and 1,980 target rows.
The environment uses NumPy 2.4.4 and Pillow 12.2.0; the return used NumPy 2.3.5
and Pillow 12.3.0.

## What the result supports

The GX86088 pool contains existing candidates with smaller clicked errors and
less far-strip clipping than the selected fits. That supports a ranking
experiment with a fixed candidate pool. It does not establish a clean fit.
The annotation-derived plane and its sensitivity checks remain diagnostics.

Shared paint scoring improves the sparse GX results, but a severe error
survives three observed frames. The proposed adaptive two-to-three-frame rule
has not been run. Sparse selection still relies on archived registrations;
this replay does not validate sparse registration or runtime.

These results complement the inward-bias investigation. They do not explain
the mislabelled paint edge in SS03-34 or validate a correction. The next
fitting experiment should test image-based edge identity while preserving
the existing selections for comparison. No detector behaviour changed here.
