# Follow-up 2: clean-interface VLM choice

## Bottom line

Use **InternVideo3** first in the remaining follow-up work.

Intern identified the server on 23 of 32 rally starts. Qwen was correct on 14.
That result supports Intern's existing advantages on scene checking and marked
shuttle-track checking. The optional automatic-evidence prompt was not needed
to make the choice.

Neither model handled the full rally-start task well. Both labelled all 32
cases `visible`, including all 13 where the reviewed contact was off-frame,
omitted by the broadcast or unclear. Each model located only 1 of
the 19 visible contacts within the accepted timing tolerance. Intern is the
relative choice between these two models, not a validated serve-state or
contact-time judge.

## What ran

Both models saw the same 32 clips built from the existing reviewed rally-start
set. The fixed case identities came from that set. Its serve-state and contact
labels were kept out of clip selection and used only for the final score.

The automatic builder looked near the first three accepted contact guesses. It
used a nearby PySceneDetect cut when the court was present for at least 80% of
the following second. This selected 26 starts. The remaining six clips were
centred on the earliest accepted contact.

Every clip contained 120 consecutive source frames at the video's native frame
rate. All 19 reviewed visible contacts fell inside their clips. Each frame was
labelled with a clip index from 0 to 119 and its original source-frame number.
A gold border marked the selected camera cut when one was used.

The models received the same prompt. They had to return:

```json
{
  "server": "top | bottom | unclear",
  "serve_state": "visible | off_frame | broadcast_omitted | unclear",
  "contact_frame": "integer | null"
}
```

An exact frame was permitted only when physical service contact was visible.
The prompt defined top and bottom by the normal full-court view. It also warned
that the marked cut could be wrong.

The runs used these fixed model revisions:

- `yanziang/InternVideo3-8B-Instruct` at
  `c4602918b65225650d152db2850fe34e01d21fcd`
- `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` at
  `d9748a51ae66354c4dad665aab2c71f26cf2c8cd`

Both runs completed all 32 cases and consumed all 120 frames per case.
The paired full run started at 15:16 and finished at 15:23 on 21 August 2026.
Both model processes exited with code 0 and left the GPU empty.

## Result

| Measure | Intern | Qwen |
|---|---:|---:|
| Server correct | **23 / 32 (71.9%)** | 14 / 32 (43.8%) |
| Serve state correct | 19 / 32 (59.4%) | 19 / 32 (59.4%) |
| Visible contact exact | 0 / 19 | **1 / 19** |
| Visible contact within project tolerance | 1 / 19 (5.3%) | 1 / 19 (5.3%) |
| Visible contact within ±10 base-30 frames | 1 / 19 | 1 / 19 |
| Visible contact within ±15 base-30 frames | 1 / 19 | **2 / 19** |
| Mean absolute error when contact was visible | 46.8 source frames | **32.3 source frames** |
| Exact frame claimed without visible contact | 13 / 13 | 13 / 13 |
| Cases with an `unclear` answer | 0 / 32 | 0 / 32 |

The serve-state score needs context. Both models answered `visible` on every
case. Their 19 correct answers therefore reproduce the number of visible cases
in the benchmark. Neither model recognised any of the eight off-frame cases,
four broadcast omissions or the one unclear case.

Intern's server result was the useful separation. Intern gave a balanced 17
bottom and 15 top answers against truth containing 20 bottom and 12 top
servers. Qwen answered top on 28 of 32 cases. Intern alone was correct on 12
server labels, while Qwen alone was correct on three.

Contact timing was poor for both models. Intern's most common frame answers
were 118 and 4. Qwen's most common answer was 11. These repeated positions and
the large timing errors show that neither model reliably tracked physical
contact through the clip.

## Inspected mistakes

The inspected clips matched the aggregate failure:

- `rally-start-sset_01-c02` was reviewed as broadcast-omitted. The clip began
  with active play, moved through a broadcast transition and returned to court.
  Both models called contact visible at clip frame 2.
- `rally-start-sset_21-c06` showed close serve preparation followed by a cut to
  active rally play. The reviewed contact was off-frame. Intern claimed frame
  3 and Qwen claimed frame 119.
- `rally-start-sset_21-c31` was the single unclear case. It moved from a close
  player view to court action. Both models called contact visible and supplied
  an exact frame instead of abstaining.
- `rally-start-sset_15-c05` was Qwen's one exact visible-contact time, but Qwen
  named the wrong server. Intern also named the wrong server and was 33 frames
  early.
- `rally-start-sset_21-c28` was Intern's best complete visible case. Intern
  named the correct server and was two frames late. Qwen named the wrong server
  and was 23 frames early.

These examples do not establish why the models favoured particular frame
positions. The design changed camera content around automatic cuts and marked
those cuts with a border. This gate does not isolate which part influenced the
repeated early and late answers.

## Decision

Choose Intern as the starting model for Follow-ups 3 and 4.

Intern won the only useful separation in this gate: server identity. The
retained evidence also favours Intern for the marked shuttle-track check and
for the precision-first scene trade-off. Qwen retained more standard-view live
scenes and had slightly lower contact-time error here, but neither timing result
was practically useful.

The models were not genuinely tied, so the optional compact automatic-evidence
arm was not run. Later work must not treat Intern's rally-start state or timing
answers as dependable merely because Intern was selected.

If Follow-up 4 finds a materially better evidence format, that frozen format
may trigger one paired Qwen confirmation. A changed model choice would apply to
the enhanced interface. It would not alter this clean comparison.

## Limits

This is a paired local model test on 32 reviewed starts from three fixture
videos. Twenty-six cases come from `sset_21`, so it is not a broad broadcast
sample. The clips also start from existing automatic contact and cut evidence;
the experiment does not test an unconstrained search through a full match.

Intern wrote 14 frame numbers with leading zeroes, such as `004`. Leading-zero
integers are invalid JSON. Scoring applied one narrow normalisation that changed
only that token to the same integer without zero padding. The raw replies and
their original parser errors remain unchanged in the evidence archive. Qwen's
32 replies parsed without normalisation.

The server result is a model-selection signal. It is not an end-to-end
annotator result. A later pipeline still needs independent checks before using
model-generated serve state or timing.

## Evidence

- [`../../experiments/rally_start_trials.py`](../../experiments/rally_start_trials.py)
  contains the clip builder, fixed prompt, model runner, normalisation and
  scorer. [`../../experiments/rally_start_remote.sh`](../../experiments/rally_start_remote.sh)
  contains the Carmack run boundary and GPU-release check.
- [`evidence/2_rally_start_score.json.gz`](evidence/2_rally_start_score.json.gz)
  contains both row-level scores and the recorded normalisation count.
- [`evidence/2_rally_start_remote_runs.tar.gz`](evidence/2_rally_start_remote_runs.tar.gz)
  contains all 64 raw replies, exact prompts, model identities, sampling
  records, terminal statuses and post-run GPU checks. Carmack paths were
  replaced with portable labels; raw model text was not changed.
- [`evidence/2_rally_start_manifest.json.gz`](evidence/2_rally_start_manifest.json.gz)
  contains the 32 inference cases without human truth.
- [`evidence/2_rally_start_truth.json.gz`](evidence/2_rally_start_truth.json.gz)
  contains the separate scoring labels.
- [`evidence/2_rally_start_provenance.json.gz`](evidence/2_rally_start_provenance.json.gz)
  records clip selection, input and clip hashes, and build settings.
- [`2_final_model_gate.json.gz`](2_final_model_gate.json.gz) contains the main
  result and decision in machine-readable form.

The manifest SHA-256 is
`399f8545cdf3cd193c51a070e7e871cec905bbc77aa44a7c33234606c0beffc4`.
The truth SHA-256 is
`722677d0332ba4b015a9aa11cae1ed5f0e61c071b0ec236cce0f5a3f265754a9`.
The prompt SHA-256 is
`4088923a22a7cccb5014a8892e1e33fb470d8d0a5d0c7064f5ba2732a4ebae7d`.
