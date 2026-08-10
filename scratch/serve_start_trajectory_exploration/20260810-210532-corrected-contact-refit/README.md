# Incoming shuttle before the first accepted contact

This investigation asks one question:

> Do we see the shuttle travelling towards the player at the first accepted contact? If so, does adding one earlier shot by the other player make the server attribution correct?

The first accepted contact's player comes directly from the shuttle and player geometry at that frame. The old fitted server label is never used to choose the player or measure the incoming path.

The analysis uses all rallies in `sset_01`, `sset_15` and `sset_21`. It looks back at most 30 base-30 frames. Any path used as evidence must stay inside one continuous court scene and pass simple visibility, recurrence, movement and jump checks.

When incoming motion is found, the experiment adds one missing shot before the accepted-contact sequence. It tries that addition in two ways:

- add an unknown-player shot, which shows the effect of correcting the contact count;
- add a shot by the other player, which also gives one extra vote to the player implied by the incoming motion.

The second version is not a separate measurement of player identity. It trusts the first experiment one extra time.

## Result

The motion rule found 11 of 16 clear first returns and made 3 false calls. In all 16 covered rallies where the rule fired, directly naming the other player as server was right 13 times. The two alternating refits were right 8 and 9 times, so the direct motion inference is the useful result. See `report.md` for the denominators, threshold plot and limits.

## Files

- `plan.md`: exact experiment and exclusions.
- `findings.md`: verified repository and data facts.
- `prepare_inputs.py`: link and verify local frozen inputs.
- `trajectory_features.py`: small path calculations.
- `experiment_data.py`: load frozen results and rebuild direct contact geometry.
- `analyse_serve_trajectory.py`: run both experiments and make the tables and plots.
- `validate_outputs.py`: recalculate the reported counts from compressed tables.
- `report.md`: plain-language result, written after the experiment runs.

Generated inputs, tables, plots and delegated-agent records remain ignored. Generated NumPy arrays use `.npy.xz`; JSON and CSV use `.json.gz` and `.csv.gz`.
