# Earliest-contact serve trajectory investigation

This investigation asks whether shuttle motion before the earliest accepted contact can improve the guess about who served.

Start with [report.md](report.md). Its **Main takeaway** gives the result and recommended next step before the detailed evidence.

## Result in brief

The main comparison uses 239 rallies with a one-to-one match between a ground-truth rally and a predicted contact sequence.

- The released alternating-fit method gets 124/239 servers right.
- Choosing the player at the earliest accepted contact gets 152/239 right.
- Applying the fixed motion correction directly gets 163/239 right.
- Applying the same correction through prepend/refit, with the same earliest-contact fallback, gets 159/239 right.

Motion is a small correction, not a complete server detector. Only 24/239 rallies have a usable shuttle path before the earliest contact. The direct and prepend/refit methods trigger on the same 15 rallies. Direct inference gets 13 of those right and prepend/refit gets 9.

The larger problem comes earlier. At the main ±10 timing tolerance, 97/239 earliest contacts do not match any annotated stroke. Later accepted contacts recover the serve in 49 of those rallies and the first return in another 36. Improving the starting contact is therefore the clearest next step.

## What is here

- [report.md](report.md): the readable, standalone result.
- [HANDOVER.md](HANDOVER.md): the current state and how to rerun the work.
- [decisions.md](decisions.md): fixed analysis choices that a follow-up must preserve.
- [findings.md](findings.md): checked results and limits in compact form.
- [plan.md](plan.md): sensible next experiments, not unfinished work in this one.
- [worklog.md](worklog.md): a short log from the cleanup onwards.
- `analyse_serve_trajectory.py`: builds the row-level analysis and checked result tables.
- `trajectory_features.py`: alignment, accepted-sequence and trajectory calculations.
- `experiment_data.py`: loads the frozen inputs and ground truth.
- `report_outputs.py`: regenerates the report and six plots.
- `validate_outputs.py`: independently rebuilds the results from the frozen sources.
- `test_trajectory_features.py`: focused tests.
- [ARCHIVE/ARCHIVE_MAP.md](ARCHIVE/ARCHIVE_MAP.md): the full two-day plan, worklog, findings, decisions and retained review records.

The live documents are deliberately short. Use `ARCHIVE/` only when you need the reasoning history or an earlier review.

## Local data and outputs

The frozen release asset, linked input arrays and generated outputs are local and ignored by Git:

- `assets/shuttleset-current-annotator-reference-v1/`
- `inputs/`
- `outputs/`

Run these commands from the repository root:

```bash
PYTHONPATH=src ~/.venvs/badminton-cicd/bin/python scratch/serve_start_trajectory_exploration/prepare_inputs.py
PYTHONPATH=src ~/.venvs/badminton-cicd/bin/python scratch/serve_start_trajectory_exploration/analyse_serve_trajectory.py
PYTHONPATH=src ~/.venvs/badminton-cicd/bin/python scratch/serve_start_trajectory_exploration/validate_outputs.py
```

The first command checks every linked track and pose input against its recorded MD5. The analysis command rebuilds the six compressed numerical outputs, the report and the six plots. The validator reloads the frozen sources and independently checks the saved rows, calculations, report and plots.
