# Projective court patterns: useful matching, unresolved automatic directions

The spacing matcher produces usable courts when given suitable perspective
directions. Automatic direction selection still discards useful precision,
and paint ranking can favour wrong structures when the candidate pool expands.
A further GX0 control produces two visually approved, essentially ideal courts
from discarded observed directions. The next useful change is to direction
retention or refinement, followed by
ranking checks against the saved false winners. Production detection is unchanged.
An untested [SVD refinement option](followup_status.md#candidate-experiment-svd-direction-refinement)
fits within that next step; nearest-neighbour lookup is a conditional optimisation.

The experiments progressed through four stages on existing development data:

| Stage | Main result | Report and comparison |
|---|---|---|
| Direction-based pruning of the old rectangle search | Recovers the approved GX frame-5 seed, but several other fits have substantial marking errors | [Results](vp_pruning_results.md) · [Gallery](vp_pruning_visual_check.html) |
| Marking diagnosis and fixed-assignment refits | Three ShuttleSet refits are visually good; GX and Amateur-2 select wrong structures or identities | [Results](marking_diagnosis_results.md) · [Gallery](marking_followup_visual_check.html) |
| Spacing matcher with supplied directions | Usable generated courts in all eight inspected views; paint ranking helps Amateur-2 and worsens GX frame 0 | [Results](axis_matching_results.md) · [Gallery](axis_matching_visual_check.html) |
| Spacing matcher with automatic directions | Visual inspection finds usable choices on six views across the two rankings; GX0, GX5 and Amateur-2 frame28019 remain badly fitted | [Results](automatic_axes_results.md) · [Gallery](automatic_axes_visual_check.html) |

The pruning population contains 47 cached views. The detailed follow-ups use
nine views from five videos: GX, Amateur-2, Amateur-3, ShuttleSet 03 and ShuttleSet
21. ShuttleSet views are cached median composites. These are development
comparisons, with no designated holdouts. The six-view count requires choosing
between rankings after inspection; it is not an automatic success rate.
[Automatic panel judgements](automatic_axes_visual_judgements.md) preserve the
individual outcomes. Visual judgements belong to the exact
panels inspected; small corner errors do not establish correct internal markings.

[Methods and validation](methods.md) define the comparisons and their limits.
[Follow-up status](followup_status.md) maps the findings to the staged research
options. [Measurements](measurements.json.gz) preserve all three automatic
comparison arms and the first two label-guided direction controls. The subsequent
[GX0 control measurements](gx0_control_measurements.json.gz) and
[inspected comparison](gx0_control_visual_check.html) record its separate approval. The galleries use
shared local images and work without JavaScript.
