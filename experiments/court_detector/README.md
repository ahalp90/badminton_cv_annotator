# Court detector: development and experiments

**[The development story](report.md)** explains why the detector was needed, how it
works, and what the experiments taught. It is the main account of the work.

The detector finds court geometry in badminton footage so image positions can be
mapped onto the court floor. Its development centred on recognising the playing
court among many plausible line arrangements, then combining evidence through
occlusion and camera cuts.

## Detailed comparisons

These reports expand on particular questions in the story. Each includes links
to its measurements and images.

| Report | Question |
| --- | --- |
| [Earlier approaches](comparisons/earlier_approaches.md) | Why were the early line-search approaches set aside? |
| [Development evaluation](comparisons/development_evaluation.md) | What supported the scoring, composition, pooling and performance choices? |
| [Release evaluation](comparisons/release.md) · [gallery](comparisons/gallery.md) | How well do the released courts match the supplied labels, and what errors remain? |
| [Player-check trial](comparisons/search.md) | Did relaxing player checks recover useful courts or introduce false detections? |
| [Original extraction](comparisons/baseline.md) | How were rally courts measured, and why did sharing discard good courts? |

## Running or extending the work

The [detector overview](../../docs/court_detector/README.md) explains the maintained
system. The [usage guide](../../docs/court_detector/usage.md) covers inputs, setup,
CLI options and outputs. The [design guide](../../docs/court_detector/design.md) explains current
choices. The [released dataset](../../data/court_detections/sset_and_sset22/extractions_20261003/README.md)
contains the existing 86-video extraction and a loading example.

The remaining folders support reproduction and development:

- [tools/](tools/README.md): commands and runners for rebuilding the comparisons
- [evidence/](evidence/README.md): saved predictions, tables, images and clips
- [saved_views/](saved_views/README.md): fixed inputs and helpers used by detector tests
