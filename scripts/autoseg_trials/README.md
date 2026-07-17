# autoseg_trials

Frozen run-time originals of the s27-s29 contact-detection and rally-segmentation trial harnesses, committed so they stop living in throwaway /tmp worktrees. These are versioned records, not tooling: they were written to run from a worktree root with sys.path seams and their outputs beside them, so re-running them needs path adjustments. The numbers they produced live in the campaign readouts (docs migration pending; currently Ariel-local under local_scratch/autograder_architecture/).

- m_sticky_gate_arm.py / m_sticky_gate_arm_r7.py: s27 sticky-anchor gate arms at suppression radius 9 / 7
- m_miss_junk_census.py: s27 miss census and junk-signal quartiles
- s27_promotion_pin.py: md5 pin for the pre-swap chain; FAILS BY DESIGN since the s28 sticky swap (frozen record)
- s28_sticky_pin.py: DEPRECATED P-era acceptance pins, both radii (r9 pilot ddc4f60b / vid15 e0fa8941; r7 c259e147 / c332cfb2). Kept byte-frozen as the pre-base-30 record; FAILS BY DESIGN on the base-30 tree. Superseded by s28_sticky_pin_r30.py
- s28_sticky_pin_r30.py: acceptance pins re-earned after the fps constants flipped to base-30 integers (r9 pilot d4f60a1a / vid15 117920c2; r7 684e0c10 / 304c09f5). Note the r9 arm no longer matches the H yardstick record (b9988d38 / a977890b): the shipped chain's default suppression radius at 25 fps scales to 8 under base-30, so the pre-flip coincidence between "default" and "radius 9" is gone. Run recipe: copy to a worktree root beside a copy of h_end_to_end.py whose worktree path points at the same tree, measurements/ on PYTHONPATH
- s28_sticky_measure.py: the four-cell sticky measurement (known wrinkle: import-order shadowing if run beside stale pin-script copies; tidy queued). Imports the P pin module by name: historical P consumer
- s29_sweep_measure.py: the 95/95 burst_ratio x visible-run filter sweep (this is the re-earn copy that produced the record CSVs; the builder's copy differed only cosmetically). Also a historical P consumer
- h_end_to_end.py: end-to-end yardstick driver (console record: h_end_to_end_console_s28.txt). Binds an absolute wt_s24 worktree path at import: repoint it when reusing (it silently runs the OLD tree otherwise; this bit the base-30 re-pin's first re-earn round)
