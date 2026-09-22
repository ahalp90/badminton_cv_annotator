# Does the court-fitting objective favour an inset boundary?

Please independently reason through this question using the actual code:

> With correct paint-side labels, under what conditions does this objective
> recover the outer court boundary? Can frozen sample selection, centre
> assignments, weights or the stripe-width model still favour an inset fit?

Choose the most informative line of analysis. A sound explanation that rules
out a suspected mechanism is as useful as a counterexample. The aim is to
predict a discriminating local test before another fitting experiment.

## Repository and starting evidence

Repository: `ahalp90/badminton_cv_annotator` on GitHub. Pin source and results to
`13f02fdbf8954ead15dbc7241a696a314473f9c5` rather than the moving branch.
Read `.github/AGENTS.md` for shared rules. Named files are available under:

```text
https://raw.githubusercontent.com/ahalp90/badminton_cv_annotator/13f02fdbf8954ead15dbc7241a696a314473f9c5/<path>
```

Start with these small files:

- `scratch/court_det_fix/edge_polarity/README.md`
- `scratch/court_det_fix/edge_polarity/run_probe.py`
- `experiments/annotator/independent_court/fixed_stripe_refit.py`
- `experiments/annotator/independent_court/paint_geometry.py`
- `experiments/annotator/independent_court/assignment.py`
- `experiments/annotator/independent_court/stripe_observations.py`

Fetch the probe's `results.json.gz` and `test_probe.py` if useful. Resolve
projection conventions in `experiments/annotator/independent_court/detector.py`
and support constants in `junction_observations.py` only as needed. Record any
additional source used. If an essential file is inaccessible, identify the gap
instead of assuming its behaviour. No full repository clone is needed.

The court coordinates describe outside paint boundaries. Assignment chooses a
marking and centre/edge position geometrically, then the refitter freezes that
interpretation. The saved probe changes only edge labels when signed brightness
contradicts them; it keeps parents, fitted points and weights fixed.

In seven development cases, baseline refits reproduce within 1.1e-9 native
pixels. Polarity partly reduces the left inset, but results are mixed. On
SS03-34, fragment 236 is a verified inner edge assigned as outer. Correcting
strong contradictions moves the upper-left corner 1.078 working pixels left
and 0.529 up. The user's preferred diagnostic shift was 3 left and 2 up.
The existing paint score falls slightly; that score is distinct from the
refitter's least-squares objective. The previously approved GX5 fit barely moves.

## Useful output

Explain the objective and its assumptions, with exact file/function references.
Separate an ideal noiseless model from the effects of measured fragment
positions, blur, initial-fit-dependent selection and ambiguous edge identity.
Correct polarity identifies a side; it need not locate the physical boundary
exactly. Distinguish a correct solution attaining minimum loss from the data
uniquely determining that solution. Do not assume that an observed inset proves
an optimiser or implementation bug.

Supply a concise derivation or a minimal executable synthetic example for the
most consequential claim. State the coordinate system, observable inputs and
expected result. A small NumPy/SciPy calculation is welcome; do not run a full
detector, parameter sweep or image review. Keep labels out of any proposed
automatic rule. A fixed corner offset is not an acceptable correction.

Recommend the single most useful local test, including its predicted numerical
effect and what result would reject your explanation. State what the code or
synthetic example establishes, and what still needs real-fragment measurements.
If the objective is sound under the relevant assumptions, say so and identify
the observation or assignment assumption that most needs checking.

Return a short Markdown report and any small runnable example as files.
Preserve contrary findings and list checks actually run. No repository edits
or production integration are requested. Avoid a broad redesign or a catalogue
of unranked possibilities; stop when the next discriminating test is clear.
