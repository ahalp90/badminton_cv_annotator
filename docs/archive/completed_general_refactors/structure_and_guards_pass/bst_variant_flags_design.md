# BST variant flags (PPF / CG / AP): design and revival

Revival record for the three variant flags on `BST` (`use_ppf`, `use_cg`,
`use_ap`) and the four unused partial constructors they feed. The next
commit makes CG and AP always-on: it deletes the flags from
`src/bst_x/model/bst.py`, de-branches the forward, drops the four unused
partials (`BST_0`, `BST_PPF`, `BST_CG`, `BST_AP`), and leaves
`BST_CG_AP` as the one graph (redefined as a plain alias
`BST_CG_AP = BST`). This doc captures what the flags gated and every
seam they touched, so re-adding a variant later is a wiring job rather
than an archaeology dig.

Written against the live code the moment before removal. Line numbers
are as-read at that point; find everything by symbol, the line numbers
are just a starting nudge.

**Why they're going.** The variants exist because the BST paper (Chang's
lineage) ablated them: PPF, CG and AP are the paper's own optional
modules, and the five partials mirror its ablation table. This project
only ever trained the fully-loaded combo. Every checkpoint on disk is
the `BST_CG_AP` graph (`use_ppf=True, use_cg=True, use_ap=True`): 124
weight files, all the same graph, zero files for any of the other four
variants. Both `bst_x_train` and `bst_x_infer` build `'BST_X'`, which is
just an alias for `BST_CG_AP` in the registry. So the flag machinery
carries four variants nobody builds and a set of forward branches that
only ever take one path. It comes out until we actually want the
optionality back.

---

## 1. The three flags and what each gates

The flags live on the `BST.__init__` signature in
`src/bst_x/model/bst.py` (~L155): `use_ppf=True, use_cg=False,
use_ap=False`. Each one gates one optional module at construction and
one branch in `forward`. The class docstring (~L136-150) carries the
"Three boolean flags" paragraph and the original variant-mapping table.

### `use_ppf`: Pose Position Fusion

Gates `self.mlp_positions = MLP(2, out_dim=in_dim, ...) if use_ppf else
None` (~L167). In the forward (~L294), when on, it projects each
player's 2D court position up to `in_dim` and fuses it into the skeleton
features multiplicatively with a residual (`JnB = JnB * pos_impact +
JnB`), before the TCN. The idea is to let where a player is standing
modulate their pose features. When off, `mlp_positions` is `None` and
the raw skeleton features go straight to the TCN.

### `use_cg`: Clean Gate

Gates `self.mlp_clean = MLP(d_model, d_model, d_model, ...) if use_cg
else None` (~L201). In the forward (~L414), when on, it takes the
element-wise minimum of the two players' shuttle-interaction summaries
(the shared signal), runs it through `mlp_clean` to learn the "dirt",
and subtracts `cg_factor * dirt` from the shuttle summary. The point is
to strip shared player noise out of the shuttle representation before
the head reads it. When off, the shuttle summary passes through
untouched.

### `use_ap`: Aim Player

Gates `self.cos_sim = nn.CosineSimilarity() if use_ap else None` (~L197).
In the forward (~L396), when on, it measures cosine similarity between
each player's shuttle-interaction summary and the shuttle summary, maps
the difference to an `alpha` in [0, 1], and scales each player's
conclusion vector by an effective alpha blended via `ap_factor`. The
idea is to weight the player who is actually aiming at the shuttle more
heavily. When off, both players' conclusions carry equal weight.

### The head_dim rule, and why it depends on the flags

The final head reads a concatenation of per-stream CLS summaries, and
its input width is fixed at construction (it is the first Linear's
`in_features`). So `head_dim` has to match exactly what the forward
concatenates:

```python
head_dim = d_model * 2 if (use_ap and not use_cg) else d_model * 3
```

The default is three streams stacked: `p1_conclusion`, `p2_conclusion`,
`shuttle_cls`, so `3 * d_model`. The single exception is AP-on,
CG-off: AP already folds the shuttle's relevance into the per-player
alpha weighting, so feeding `shuttle_cls` to the head as well would
double-count it. That combo drops the shuttle stream and concatenates
just the two players, so `2 * d_model`. The matching branch is at the
head concat in forward (~L424): `if self.use_ap and not self.use_cg`
takes the two-stream path, everything else takes the three-stream path.

So the head width is not a free knob, it is a function of the flag
combination. CG, whenever present, keeps `shuttle_cls` in the head (it
edits that vector rather than removing it), which is why any config with
`use_cg=True` lands back at `3 * d_model`. In `BST_CG_AP` both CG and AP
are on, so CG wins the tie: `head_dim = d_model * 3`, and the two-stream
branch is unreachable. That is exactly why the removal can hardwire
`head_dim = d_model * 3` and delete the two-stream branch: the only
combo that needed `2 * d_model` was AP-without-CG, which nobody builds.

### The five variants in the registry

`src/bst_x/model/bst.py` pre-fills the flag combinations as
module-level partials (~L440-444):

```python
BST_0     = partial(BST, use_ppf=False, use_cg=False, use_ap=False)
BST_PPF   = partial(BST, use_ppf=True,  use_cg=False, use_ap=False)
BST_CG    = partial(BST, use_ppf=True,  use_cg=True,  use_ap=False)
BST_AP    = partial(BST, use_ppf=True,  use_cg=False, use_ap=True)
BST_CG_AP = partial(BST, use_ppf=True,  use_cg=True,  use_ap=True)
```

`src/bst_x/bst_x_common.py` imports all five (~L22) and maps them by
name in the `MODELS` dict (~L32-40), plus a `'BST_X'` alias pointing at
`BST_CG_AP` and the parked `# 'BST_X_RGB': ...` placeholder for the
future X3D-S fusion variant. `build_bst_x_network` looks a model up by
`model_name` and builds it. Train and infer both pass `'BST_X'`, so the
live path resolves to `BST_CG_AP` every time.

Consumers of the non-CG_AP partials, so the removal knows what breaks:

| Partial | Where it's used |
|---|---|
| `BST_0` | `tests/test_integration.py` (L55 import, L133 direct build) |
| `BST_PPF` | `MODELS['BST']` in `bst_x_common.py`; `__main__` demo in `bst.py` |
| `BST_CG` | `MODELS['BST_CG']`; `__main__` demo |
| `BST_AP` | `MODELS['BST_AP']`; `__main__` demo |

`test_integration.py` is the only production-side consumer of a
non-CG_AP variant. The removal repoints its `BST_0` to `BST_CG_AP` (or
the bare `BST`). The `__main__` five-variant demo dict (~L462-468)
collapses to the single CG_AP build; the print/FLOP loop below it stays.

---

## 2. The schedule-factor buffers (they survive the removal)

`cg_factor` and `ap_factor` are scalar buffers registered in
`__init__` (~L213-214), init to 1.0:

```python
self.register_buffer('cg_factor', torch.tensor(1.0))
self.register_buffer('ap_factor', torch.tensor(1.0))
```

The trainer overwrites them once per epoch via `set_schedule_factors`
(~L218) from `bst_x_train.py:685`, which passes the same `aux_factor`
into both. This runs a warm-start schedule: the factors start at 1.0
(full CG/AP) and decay toward 0.0 so the transformer backbone
increasingly stands on its own. In the forward, `cg_factor` scales the
dirt subtraction (`cg_factor=0` bypasses CG), and `ap_factor` blends the
alpha toward pass-through (`ap_factor=0` makes both player multipliers
exactly 1, i.e. no AP reweighting).

### Why buffers, not Parameters, and why they persist

They are not learned: the optimiser never touches them, the schedule
sets them from outside. But they still have to move with `.to(device)`
and land in the checkpoint, which is exactly what `register_buffer`
gives that a plain Python float would not.

The persistence matters for mid-fade checkpoints. Because the buffers
are in `state_dict` under the keys `cg_factor` and `ap_factor`, a
checkpoint saved halfway through the schedule (say `cg_factor` has
decayed to 0.4) stores 0.4, and loading it restores the model at that
exact point in the fade. Hold these as plain floats or non-persistent
buffers instead, and a reloaded checkpoint would silently reset to 1.0
(full CG/AP): resuming a run or running inference from a mid-schedule
checkpoint would then use the wrong factor without complaint. The
buffer keys are the thing that keeps the fade state honest across a
save/load round-trip.

### How they interact with the flags, and why the removal keeps them verbatim

The buffers are registered unconditionally, regardless of the flags:
even `BST_0` (no CG, no AP) carries both `cg_factor` and `ap_factor` in
its state_dict, they just never get read in its forward. `cg_factor` is
only read inside the `if self.use_cg` block; `ap_factor` only inside the
`if self.use_ap` block. So the flags gate whether the factors do
anything, but not whether the buffers exist.

That unconditional registration is why the removal leaves them alone.
With `use_cg` and `use_ap` always true, the forward blocks that read the
factors become the sole path, but the buffer reads inside those blocks
are byte-identical to what runs today. The `register_buffer`
calls and `set_schedule_factors` carry over verbatim, and the trainer's
call site at `bst_x_train.py:685` does not change. This is also part of
why the removal is state_dict-neutral: the buffer keys were already in
every checkpoint.

---

## 3. The Stage-6 fusion seam this leans on

The X3D-S integration plan
(`docs/architecture_notes/x3d_integration_macro_plan/x3d_integration_macro_plan.md`)
has an open question about how the coming video stream slots into BST.
The variant-flag pattern is one of the two candidate mechanisms. Open
question 6.A #5 (in the Stage 6 section):

> **Where the fused feature lives in the model.** New separate `Arch1`
> module that wraps BST + X3D-S + fusion, or BST extended with an
> optional `x3d_branch` flag like the existing PPF/CG/AP toggles? Latter
> keeps the codebase smaller; former isolates the X3D-S code.

The critical-files inventory (same doc) echoes it: "`src/bst_x/model/bst.py`
(Stage 6; possibly extend with optional `x3d_branch` flag, or leave
untouched and wrap)".

Right now the tree carries a worked example of the "flag on BST" route:
PPF/CG/AP show the full shape of bolting an optional branch onto BST via
a boolean flag, a conditional module at construction, a conditional
block in forward, and a partial in `MODELS`. If Stage 6 picks the
`x3d_branch` route, that pattern is the template to copy.

The removal deletes that worked example. It does not block Stage 6, but
it shifts the calculus for the still-open decision:

- the wrapper route (separate `Arch1` module) is unaffected: it never
  leaned on the flag pattern
- the `x3d_branch`-flag route loses its in-tree exemplar, so an
  implementer taking it would reconstruct the shape from this doc rather
  than copy a live one, which tilts the still-open call toward the
  wrapper
- the other declared extension seams survive: the `MODELS`
  name-to-constructor table (the `BST_X_RGB` placeholder rides on it)
  and `build_bst_x_network`'s `model_name` dispatch both stay intact, so
  a new fusion variant can still register a name either way

This doc is the archaeology-avoider for the flag route specifically: if
Stage 6 wants `x3d_branch` as a BST flag, §1 and §4 are the shape to
rebuild.

---

## 4. State_dict impact, and re-adding a variant

### Why the removal renames no checkpoint keys

Every checkpoint on disk is the `BST_CG_AP` graph, so making the flags
always-on keeps the same param-bearing submodules (`mlp_positions`,
`mlp_clean`, `mlp_head`, the TCNs, the transformers) under the same
attribute names. State_dict keys are unchanged and every checkpoint
loads. Two details reinforce that:

- `self.cos_sim` is `nn.CosineSimilarity`, which has no parameters or
  buffers, so it contributes no state_dict keys. Keeping it, dropping
  it, or inlining `F.cosine_similarity` are all state_dict-neutral
- `cg_factor` / `ap_factor` are already in every checkpoint (§2), so
  keeping them is neutral

Keys would change only for the non-CG_AP variants: `BST_0` has
`mlp_positions=None`, so it emits no `mlp_positions.*` keys, and so on.
Nobody has weights for those (124 of 124 files are the CG_AP graph), so
removing them breaks zero checkpoint loads.

### Post-removal state to re-wire against

After the removal: `BST.__init__` has no `use_ppf` / `use_cg` / `use_ap`
parameters, the three optional modules are created unconditionally,
`head_dim` is hardwired to `d_model * 3`, the forward has no PPF/AP/CG
branches (all three run, the head concat is always three-stream),
`MODELS` carries only `BST_CG_AP` / `BST_X` (with `BST_CG_AP = BST` as a
plain alias), the four partials are gone, and `test_integration.py`'s
`BST_0` has been repointed.

### Re-adding a variant: the checklist

- re-add `use_ppf` / `use_cg` / `use_ap` to `BST.__init__` (the old
  defaults were `use_ppf=True, use_cg=False, use_ap=False`) and restore
  the `self.use_ppf` / `use_cg` / `use_ap` assignments
- re-conditionalise the three module builds: `mlp_positions` (`if
  use_ppf else None`), `cos_sim` (`if use_ap else None`), `mlp_clean`
  (`if use_cg else None`)
- restore the head_dim derivation: `head_dim = d_model * 2 if (use_ap
  and not use_cg) else d_model * 3`
- re-branch the forward: the PPF fusion on `use_ppf`, the AP block on
  `use_ap`, the CG block on `use_cg`, and the head concat on `use_ap and
  not use_cg`
- re-create the four partials (`BST_0`, `BST_PPF`, `BST_CG`, `BST_AP`)
  in `bst.py`, re-add them to the `bst.py` import and the `MODELS` dict
  in `bst_x_common.py`
- restore the class docstring's flag paragraph and variant-mapping
  table, and the `__main__` five-variant demo dict
- point `test_integration.py` back at `BST_0` if the direct-build test
  is wanted again

### Checkpoint note for a revival

Existing `BST_CG_AP` checkpoints keep loading unchanged: they always
had all three modules. A newly-built non-CG_AP variant is a different
graph with a different key set (e.g. `BST_0` has no `mlp_positions.*` or
`mlp_clean.*` keys), so it can neither load a CG_AP checkpoint nor lend
its weights to one. That is expected: they are different networks, and
nothing on disk was ever trained as anything but CG_AP. A revived
variant starts from scratch or from its own fresh training.
