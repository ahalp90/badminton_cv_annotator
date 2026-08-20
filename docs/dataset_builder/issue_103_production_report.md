# Issue 103 ShuttleSet production result

Status: completed and validated on 17 August 2026

## Result

The supported fixed-source production run completed all 40 eligible
ShuttleSet videos on its first attempt. It assembled 3,527 rally records, with
no failed, unavailable, or validation-failed videos. The four manifest
exclusions were `sset_09`, `sset_10`, `sset_12`, and `sset_27`.

The run processed 4,442,098 source frames, or 44.695 hours of video. TrackNet
used stride 8 and batch size 16. Every shuttle result retained its Inpaint
sidecar. Commentary processing was disabled in the frozen configuration.

This run establishes corpus completeness, artifact integrity, and recovery
behaviour for the fixed-source path. It is not a new annotation-accuracy
measurement.

## Frozen identity

- Source commit: `ad8da4f297e9278a9cc39bf216026545a7bbab05`
- Artifact root: `/scratch/cmarti/issue103_ad8da4f/artifacts`
- Effective configuration SHA-256:
  `6e2a15ea3c44c4bc3cf8b38c461cdfd55c359178b49854080521949c07e93b20`
- Source manifest SHA-256:
  `83ce308fe55771f600f717be30c1e352e483b75a2fe4b2c20c66a1e6ba2e7ba8`
- CPU preflight SHA-256:
  `f27c28642e0155da18e3b247d0b6d70365e3269c332c92ede30efc72b4d419fb`
- Corpus completeness report SHA-256:
  `3cb420316e2c7d87e4278cfb137d5027c851eb1251f18a8393bf2e686340bb77`

The production environment used an NVIDIA L40 on Carmack. The run pinned the
TrackNet, InpaintNet, CourtKeyNet, RTMDet, and RTMPose model identities and the
coordinator, TrackNet, pose, FFmpeg, and FFprobe interpreters.

## Stage completion

All 40 eligible videos completed these per-video stages:

- TrackNet input preparation;
- shuttle extraction and guard evidence;
- pose extraction;
- court evidence;
- annotation;
- primitive projection; and
- artifact-index publication.

Metadata, selection, assembly, and reporting also completed. Fixed-source
mode intentionally bypassed search, transcript acquisition, triage, download,
and commentary cleaning. Commentary pairing was skipped because commentary
was disabled.

## Resume and readback

The first production attempt exited successfully at
`2026-08-17T15:07:22Z`. The supported no-op resume then completed at
`2026-08-17T15:37:26Z`.

The run manifest was identical before and after the no-op resume. Its SHA-256
was `84f91c139decdc4fe29957b8dd56cdd400491ba2b5aa190684fd3aa0e84a55db`.

Independent readback confirmed:

- 40 unique, reloadable artifact indexes;
- all indexed outputs and bound model files matched their recorded digests;
- frame alignment passed for TrackNet input, shuttle tracks, guard codes,
  Inpaint masks, pose arrays, court masks, annotation outputs, commentary
  pairing, and primitive projections; and
- the handoff contained one validated artifact index for every eligible
  video.

Shuttle, pose, and court artifacts remain pinned. After Issue #96 selects the
final annotation configuration, the supported replay path can rebuild
annotation and primitive projection without repeating those vision stages.
