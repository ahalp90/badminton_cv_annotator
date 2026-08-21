# Follow-up 1: Qwen scene comparison

## Bottom line

InternVideo3 remains the provisional choice after both models were tested on
the same 463 short scene clips. Qwen kept more standard-view live targets, but
it kept fewer unusual-view live targets and accepted more targets containing
non-live footage as live.

This is not the final model choice. Follow-up 2 tests both models on rally-start
serve reconstruction, which is closer to the later task.

## What ran

Qwen saw the frozen 463-case input previously run with Intern. Every case used
the same 120 consecutive frames, marked target, prompt, human truth and scorer.
All 463 Qwen replies parsed successfully.

The material evaluation contains 347 targets. It excludes 116 targets shorter
than the accepted timing margin. The remaining cases contain 290 standard-view
live targets, 10 unusual-view live targets and 47 targets containing some
replay, cutaway or other footage.

Qwen used `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` at revision
`d9748a51ae66354c4dad665aab2c71f26cf2c8cd`. The run started at 13:36 and
finished at 14:06 on 21 August 2026.

## Main result

| Material targets | Qwen | Intern |
|---|---:|---:|
| Standard-view live kept | **288 / 290 (99.3%)** | 270 / 290 (93.1%) |
| Unusual-view live kept | 0 / 10 (0.0%) | **6 / 10 (60.0%)** |
| Targets containing non-live footage sent for further checking | 15 / 47 (31.9%) | **21 / 47 (44.7%)** |
| Pure replay sent for further checking | 1 / 25 (4.0%) | **5 / 25 (20.0%)** |
| All live targets kept | **288 / 300 (96.0%)** | 276 / 300 (92.0%) |
| Kept targets that were truly live | 288 / 320 (90.0%) | **276 / 302 (91.4%)** |
| Correct route overall | **303 / 347 (87.3%)** | 297 / 347 (85.6%) |

Qwen's higher overall route count came from standard-view live footage. It
correctly retained 18 standard-view targets that Intern sent for further
checking. There were no standard-view targets where Intern alone was correct.

That gain did not extend to the harder groups. Qwen labelled all 10 meaningful
unusual-view live targets as replay. Intern retained six. Among the 47 targets
containing non-live footage, Intern alone caught eight while Qwen alone caught
two. Both models missed 24.

The pure-replay result is the clearest limit. Qwen accepted 24 of 25 pure
replays as live. Intern accepted 20 of 25. Neither local scene check is a
dependable replay filter.

## Representative mistakes

The inspected cases matched the aggregate result:

- `sset_01-r000-S00021` showed ordinary full-court serve preparation. Qwen
  called it live and Intern called it cutaway. This is a representative Qwen
  improvement on standard-view footage.
- `sset_01-r009-S00051` showed current live action from a close side view. Qwen
  called it replay and Intern called it live. This represents Qwen's complete
  miss on the unusual-view group.
- `sset_01-r043-S00169` was a pure replay showing close-court action. Qwen
  called it live while Intern sent it for further checking as cutaway.
- `sset_15-r008-S00110` was a pure replay from the standard full-court angle.
  Both models called it live.
- `sset_15-r099-S00565` was a close-up cutaway. Qwen sent it for checking as
  replay while Intern called it live. Qwen chose the wrong content name but
  took the correct route.
- `sset_21-r078-S00440` was a short replay target. Qwen called it replay while
  Intern called it live.

The local pixels can look like active badminton in both current play and a
replay. A four-second local view often lacks the earlier broadcast sequence
needed to distinguish them. Qwen also appears to rely too heavily on camera
view: it called every unusual-view live target a replay while still missing
most true replays. This observation applies to this fixed benchmark, not to the
model in general.

## Provisional decision

Prefer Intern provisionally.

Intern had higher precision among targets it accepted as live. It also handled
unusual live views and non-live footage better. Those errors matter more for a
precision-first annotation route than Qwen's stronger retention of ordinary
live footage.

The completed supporting tasks point in the same direction overall. Intern was
stronger on the marked shuttle-track check. Qwen was stronger on the isolated
contact-window task, but its filter did not improve the 12 selected full-rally
evaluations.

Follow-up 2 must still test both models. Serve reconstruction is a different
task, and this scene result does not settle it.

## Evidence

- [`evidence/1_scene_qwen_score.json.gz`](evidence/1_scene_qwen_score.json.gz)
  contains the complete Qwen row-level score.
- [`evidence/1_scene_intern_reference_score.json.gz`](evidence/1_scene_intern_reference_score.json.gz)
  contains the frozen Intern rows used for the paired comparison.
- [`evidence/1_scene_manifest.json.gz`](evidence/1_scene_manifest.json.gz)
  contains the exact 463-case input manifest.
- [`evidence/1_scene_qwen_remote_runs.tar.gz`](evidence/1_scene_qwen_remote_runs.tar.gz)
  contains all 463 raw Qwen replies, the terminal status and the post-run GPU
  check. Carmack paths were replaced with portable clip labels; raw model text
  was not changed.
- [`1_scene_comparison.json.gz`](1_scene_comparison.json.gz) contains the main
  counts and decision in machine-readable form.

The manifest SHA-256 is
`dc11d79588d68a169cba253291e746ddf995e8ba742d0499fc8a5fb346f91661`.
The truth SHA-256 is
`114a00d87e1c66345f11937205830716bace2de41880a83bfd40bd5e29c2fc78`.
The prompt SHA-256 is
`88865596cd0e44bc68abf769be52a223853415164d870d7f9ca278ec4c07c5d5`.
