# Whole-rally chooser comparison

**Combining the evidence recovers substantially more complete rallies at ±10, with a larger timing cost at ±5.** The richest chooser reaches **235 correct rallies from 182** across eight reused validation videos and 677 predicted sections: 56 repairs and three losses. At ±5 it reaches 195 from 175, with 44 repairs and 24 losses. The same predictions are used at both tolerances, expressed as frames on a 30 fps clock.

The question was whether opening repairs, original player-side disagreement and saved physical measurements help when a model judges whole-rally alternatives together. They do help in this comparison. The direct physical inputs mainly reduce damage once the opening and side evidence are present.

The larger goal is to recover usable badminton rallies for annotation. This result makes the combined chooser a candidate for the later integration round. It does not establish which rallies can be accepted automatically. All changes remain in experiment scripts.

## Evaluation contract

A section is the detector's proposed stretch of video for a rally. It is fully correct only when it contains every contact of exactly one labelled rally, predicts each contact once within the timing allowance, and assigns every player side correctly. The fixed side vote selects the alternating Top/Bottom sequence that agrees with more original side guesses.

Development (D) contains 32 videos and 2,850 sections. Validation (V) contains eight videos and 677 sections. V is excluded from fitting, but had been examined previously; it is reused validation, not a fresh test. All sections remain in evaluation. Unknown or ambiguous labels exclude options from teaching only. Every successful whole alternative is a positive training example, including keep when the original section is correct. Labels never choose prediction options.

The fixed alternatives were keep; add or replace the first contact using two earlier candidates; delete one event; and start-plus-delete forms. There was no later insertion, extra candidate generation, or vision rerun. The chooser therefore tests selection among a fixed set of upstream alternatives, rather than discovering a new contact sequence.

Training and cutoff selection used a ±10-frame tolerance on a 30-fps clock, with the time scale applied once. The same V predictions were also scored at ±5 frames. Cutoffs of 0, 0.5 and 0.9 were selected on D net-correct results. All three policies chose 0; an edit still had to beat keep strictly. No zero-loss veto was applied.

Three upper-model feature variants were compared:

| Variant | Features | What it adds |
| --- | ---: | --- |
| Summaries | 37 | Rally summary evidence |
| Opening + sides | 49 | Summaries plus local opening scores and eight raw side features |
| Opening + sides + physics | 304 | The middle variant plus direct frozen physical measurements |

The four opening-score features describe the chosen start edit and the best available start edit in its section. Each receives a score from a summary model and a physical model. The side features measure missing guesses and disagreement with alternation, before and after each edit. Direct physics adds 85 saved measurements at each of the original first contact, proposed first contact and deleted contact.

The new opening-score fits exclude both their prediction group and the whole model's outer evaluation group. Cached upstream detector scores retain their earlier dependence across groups, so D remains descriptive. The whole model uses histogram gradient boosting: 200 iterations, 15 leaves, learning rate 0.05, minimum leaf size 20, L2 regularisation penalty 1, balanced classes, no early stopping, and seed 20260905. Local opening models retain the earlier 100-iteration, seven-leaf settings.

## Whole-rally results

Each cell gives fully correct rallies, then repairs / losses against the unchanged detector plus side vote. D has 2,850 sections; V has 677. Repairs and losses compare the same labelled rally identities.

| Variant | D ±10 | D ±5 | V ±10 | V ±5 |
| --- | --- | --- | --- | --- |
| Unchanged detector + vote | 802 (—) | 710 (—) | 182 (—) | 175 (—) |
| Summaries | 844 (73/31) | 629 (71/152) | 191 (19/10) | 146 (13/42) |
| Opening + sides | 987 (210/25) | 800 (169/79) | 233 (60/9) | 194 (46/27) |
| Opening + sides + physics | 991 (205/16) | 823 (166/53) | 235 (56/3) | 195 (44/24) |

The richest model's ±10 result is 235 of 677 sections, about 35% of all V sections. That proportion describes correct whole rallies under this candidate set; it is not an acceptance certainty. At ±5, losses increase sharply for every variant, showing that precise timing remains a material constraint.

The middle model already contains indirect physical evidence through one of its two local opening models, alongside the raw side features. Adding direct physics therefore has a narrower interpretation: it mainly reduces damage. At ±10, losses fall from 9 to 3 while correct outcomes rise from 233 to 235. This supports retaining physical measurements in the combined chooser despite their weaker standalone result. The design does not isolate the contribution of opening evidence from side evidence.

The earlier summary/opening chooser is the lower-intervention reference: it repairs 16 validation rallies at ±10 and 14 at ±5, with zero baseline losses. Direct comparisons below show what replacing each earlier chooser would gain and give up. Every row uses the same 677 validation sections.

| Earlier chooser | Earlier correct ±10 / ±5 | Richest combined: gained / lost at ±10 | Gained / lost at ±5 |
| --- | ---: | ---: | ---: |
| Historical chooser | 188 / 181 | 51 / 4 | 39 / 25 |
| Summary/opening standalone | 198 / 189 | 42 / 5 | 33 / 27 |

## Which edits help and hurt

At ±10, the richest chooser repairs 44 rallies by adding the opening contact, nine by deleting an event, one by replacing the opening, and two through a start edit plus deletion. Repairs occur in all eight validation videos, with a positive net gain in each. The larger result is therefore mostly better opening selection, with a useful deletion contribution.

Its three ±10 losses are two replacements that lose a timing match and one addition that leaves an extra event. At ±5, **23 of the 24 losses are replacements that lose a timing match**; the other is the same extra-event addition. This gives a concrete next target: avoid replacing an already accurate first contact when keeping it is also plausible.

Direct comparison with the middle model gives eight newly correct and six lost rallies at ±10, or 15 and 14 at ±5. The direct physical inputs reduce baseline damage overall, but the two models still recover different rallies.

## Contact-level costs and data accounting

At ±10, the richest V chooser selected 360 edits. Among them, 317 were judgeable and 43 were unknown. The edits newly matched 137 labelled ground-truth contacts, lost 8 labelled contacts, and added 112 unmatched contacts. There are 159 beneficial contact edits: each recovers a labelled contact or removes an unmatched prediction, without losing a labelled contact or adding an unmatched prediction. Across the complete video streams, the share of predictions matched to a label falls from 89.94% to 88.29%. Missing labels can leave valid predictions unmatched, so this is precision against the available labels. Whole-rally gains can therefore coexist with unnecessary or unmatched contact edits.

The ±5 contact picture is worse: the richest chooser has 123 beneficial contact edits, 39 labelled contacts lost, and 181 unnecessary additions among the same 360 selected edits. This aligns with the larger whole-rally loss count at the tighter tolerance.

The development pool contains 134,884 alternatives; validation contains 27,279. Of the development alternatives, 126,120 have eligible labels for teaching. All 38,217 requested frame joins to saved measurements succeed. Missing measurement cells remain NaN, which the model can handle.

## Cost, validation, and next step

The relevant compute concern is extra hours across the dataset. The measured costs give no reason to reject this saved-evidence chooser on that basis: the full comparison took 241.6 seconds including training and evaluation. The final rich fit took 7.99 seconds; V prediction and selection took 0.331 seconds; applying selections took 0.043 seconds. The prediction/selection and application timings exclude loading features, matrix construction, local-opening prediction, side voting and upstream work. Loading physical measurements for 40 videos took 15.6 seconds, and the V static feature matrix took 3.37 seconds. There was no separate validation-only loading time or end-to-end timing.

The focused checks passed 51 tests. The full suite passed 2,040 tests with 29 skips and no failures. Scoped Ruff passed (exit 0). Whole-project Ruff exited 1 with 915 existing errors outside this pass. Pyrefly exited 1 with 11 existing import errors; its project profile excludes scratch scripts. Both test runs exited 0. The change was scripts-only: no `src` or end-to-end code changed.

The bounded next issue is to preserve precise timing while reducing unnecessary edits. Retain the combined candidate chooser and an earlier lower-intervention reference for comparison, then reassess the contact-level trade-off before any later integration. The present result supports continued investigation of whole-rally selection; it does not justify automatic acceptance.

The [comparison result](results/whole_rally_result.json.gz) retains section identities, contact pairs, all development cutoff results and timings. [Frozen validation predictions](results/whole_rally_predictions.json.gz) were saved before validation labels were loaded. No settings changed after those results were examined. See the [comparison script](scripts/run_whole_rally_comparison.py) and [earlier component results](README.md).

Run from the repository root with the existing saved inputs available:

```bash
export PYTHONPATH="$PWD/src:$PWD"
python -m scratch.contact_det_closing_pass.scripts.run_whole_rally_comparison \
  --feature-root scratch/contact_det_full_ds_fit/raw/full_raw \
  --output-root scratch/contact_det_closing_pass/results
```

This completes the bounded whole-alternative comparison. Later-contact insertion, a broader serve shortlist and reliable automatic acceptance remain separate avenues. Production integration remains for the later round.
