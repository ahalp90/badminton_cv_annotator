"""Classifier stroke taxonomies and label derivation."""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# English <-> Chinese stroke name mappings (CSV I/O only; pipeline runs in English)
# ---------------------------------------------------------------------------
EN_TO_ZH: dict[str, str] = {
    "net_shot": "放小球",
    "return_net": "擋小球",
    "smash": "殺球",
    "wrist_smash": "點扣",
    "lob": "挑球",
    "defensive_return_lob": "防守回挑",
    "clear": "長球",
    "drive": "平球",
    "driven_flight": "小平球",
    "back_court_drive": "後場抽平球",
    "drop": "切球",
    "passive_drop": "過渡切球",
    "push": "推球",
    "rush": "撲球",
    "defensive_return_drive": "防守回抽",
    "cross_court_net_shot": "勾球",
    "short_service": "發短球",
    "long_service": "發長球",
    "unknown": "未知球種",
}

ZH_TO_EN: dict[str, str] = {v: k for k, v in EN_TO_ZH.items()}
STROKE_TYPES_19 = list(EN_TO_ZH.keys())
STROKE_TYPES_19_ZH = list(EN_TO_ZH.values())


# ---------------------------------------------------------------------------
# Stroke-type base lists (inputs to the Taxonomy objects below)
# ---------------------------------------------------------------------------
# Naming convention: STROKE_TYPES_<count>_<provenance>. Count = unprefixed base
# types only (no Top_/Bottom_, no 'unknown'; both applied at Taxonomy construction).
# Provenance:
#   _RAW     -- derived from the 19 by stripping specific raw types
#   _MERGED  -- BST paper 25-class base set (12 merged stroke types)
#   _UNE_V1  -- project UNE-v1 merge target (14; keeps wrist_smash and passive_drop)

STROKE_TYPES_12_MERGED = [
    "net_shot",
    "return_net",
    "smash",
    "lob",
    "clear",
    "drive",
    "drop",
    "push",
    "rush",
    "cross_court_net_shot",
    "short_service",
    "long_service",
]

STROKE_TYPES_14_UNE_V1 = [
    "net_shot",
    "return_net",
    "smash",
    "wrist_smash",
    "lob",
    "clear",
    "drive",
    "drop",
    "passive_drop",
    "push",
    "rush",
    "cross_court_net_shot",
    "short_service",
    "long_service",
]

STROKE_TYPES_18_RAW = [s for s in STROKE_TYPES_19 if s != "unknown"]


# ---------------------------------------------------------------------------
# Class merging maps: raw_type_en (CSV) -> merged-target name
# ---------------------------------------------------------------------------
# Paper-faithful BST 25-class merge per supplementary Table G.
MERGE_MAP_25: dict[str, str] = {
    "wrist_smash": "smash",
    "defensive_return_lob": "lob",
    "driven_flight": "drive",
    "back_court_drive": "drive",
    "passive_drop": "drop",
    "defensive_return_drive": "drive",
}

# UNE-v1: keeps wrist_smash and passive_drop distinct (high-info subtypes for
# the project's analysis); still folds driven_flight into drive.
UNE_MERGE_V1_MAP: dict[str, str] = {
    "defensive_return_lob": "lob",
    "driven_flight": "drive",
    "back_court_drive": "drive",
    "defensive_return_drive": "drive",
}

# Classes that appear in ``taxonomy.classes`` without a Top_/Bottom_ prefix
# even under a sided taxonomy. Read by derive_class_index when it builds the
# label string from a row's raw type + side.
NOSIDE_CLASSES: frozenset[str] = frozenset({"unknown"})


# ---------------------------------------------------------------------------
# Taxonomy: contractual class definitions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Taxonomy:
    """A pinned class taxonomy for training and evaluation.

    Each Taxonomy commits its class list explicitly; do not live-remap.
    ``__post_init__`` requires ``'unknown'`` (when present) at index ``-1``.

    :param name: e.g. ``'bst_25'``, ``'une_v1_14'``.
    :param classes: ordered class list; if sides, includes ``Top_``/``Bottom_`` entries.
    :param merge_map: rare raw types -> parent names (e.g. ``'driven_flight'``
        -> ``'drive'``), or ``None`` when no merging applies.
    :param has_sides: whether taxonomy uses ``Top_``/``Bottom_`` prefixes.
    :param excluded_base_stroke_types: raw types (CSV-level) to drop before
        merge or side-prefixing, or ``None`` when nothing is dropped.
    :param excluded_from_training: class labels retained in the taxonomy but
        omitted from its trainable class list.
    """

    name: str
    classes: tuple[str, ...]
    merge_map: dict[str, str] | None
    has_sides: bool
    excluded_base_stroke_types: frozenset[str] | None
    excluded_from_training: frozenset[str] | None = None

    def __post_init__(self):
        if "unknown" in self.classes and self.classes[-1] != "unknown":
            raise ValueError(
                f"taxonomy {self.name!r}: unknown must sit at index -1; "
                f"found at index {self.classes.index('unknown')}."
            )
        missing = (self.excluded_from_training or frozenset()).difference(self.classes)
        if missing:
            raise ValueError(
                f"taxonomy {self.name!r}: training exclusions not in classes: "
                f"{sorted(missing)}"
            )

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def has_unknown(self) -> bool:
        return "unknown" in self.classes

    def class_list(self) -> list[str]:
        """Return the ordered class list."""
        return list(self.classes)

    def trainable_class_list(self) -> list[str]:
        """Return the class list with training exclusions removed."""
        excluded = self.excluded_from_training or frozenset()
        return [label for label in self.classes if label not in excluded]

    @property
    def n_trainable_classes(self) -> int:
        """Number of classes the model actually trains on."""
        return len(self.trainable_class_list())


def _sided_classes(base: list[str], with_unknown: bool) -> tuple[str, ...]:
    """Build a (Top_..., Bottom_..., 'unknown'?) class tuple from base names.

    Helper for ``Taxonomy.classes`` when ``has_sides=True``.

    :param base: ordered unprefixed stroke names.
    :param with_unknown: if True, append ``'unknown'`` at index -1.
    :return: tuple of class names in defined order.
    """
    classes = [f"Top_{label}" for label in base]
    classes.extend(f"Bottom_{label}" for label in base)
    if with_unknown:
        classes.append("unknown")
    return tuple(classes)


# ---------------------------------------------------------------------------
# Taxonomy registry
# ---------------------------------------------------------------------------
# Six BST-X taxonomies. Each commits its class list explicitly; the
# Taxonomy.__post_init__ check enforces 'unknown' at index -1 when present.

TAXONOMY_BST_25 = Taxonomy(
    name="bst_25",
    classes=_sided_classes(STROKE_TYPES_12_MERGED, with_unknown=True),
    merge_map=MERGE_MAP_25,
    has_sides=True,
    excluded_base_stroke_types=None,  # keeps unknown rows
)

TAXONOMY_BST_24 = Taxonomy(
    name="bst_24",
    classes=_sided_classes(STROKE_TYPES_12_MERGED, with_unknown=False),
    merge_map=MERGE_MAP_25,
    has_sides=True,
    excluded_base_stroke_types=frozenset({"unknown"}),
)

TAXONOMY_BST_12 = Taxonomy(
    name="bst_12",
    classes=tuple(STROKE_TYPES_12_MERGED),
    merge_map=MERGE_MAP_25,
    has_sides=False,
    excluded_base_stroke_types=frozenset({"unknown"}),
)

TAXONOMY_UNE_V1_14 = Taxonomy(
    name="une_v1_14",
    classes=tuple(STROKE_TYPES_14_UNE_V1),
    merge_map=UNE_MERGE_V1_MAP,
    has_sides=False,
    excluded_base_stroke_types=frozenset({"unknown"}),
)

TAXONOMY_UNE_V1_15 = Taxonomy(
    name="une_v1_15",
    classes=tuple(STROKE_TYPES_14_UNE_V1) + ("unknown",),
    merge_map=UNE_MERGE_V1_MAP,
    has_sides=False,
    excluded_base_stroke_types=None,
)

TAXONOMY_SHUTTLESET_18 = Taxonomy(
    name="shuttleset_18",
    classes=tuple(STROKE_TYPES_18_RAW),
    merge_map=None,
    has_sides=False,
    excluded_base_stroke_types=frozenset({"unknown"}),
)

# Deployed BRIC class order from runtime manifests.
_UNE_MERGE_V1_NOSIDES_CLASSES = (
    "net_shot",
    "return_net",
    "smash",
    "wrist_smash",
    "lob",
    "clear",
    "drive",
    "drop",
    "passive_drop",
    "push",
    "rush",
    "cross_court_net_shot",
    "short_service",
    "long_service",
    "unknown",
)

TAXONOMY_UNE_MERGE_V1_NOSIDES = Taxonomy(
    name="une_merge_v1_nosides",
    classes=_UNE_MERGE_V1_NOSIDES_CLASSES,
    merge_map=UNE_MERGE_V1_MAP,
    has_sides=False,
    excluded_base_stroke_types=frozenset({"unknown"}),
    excluded_from_training=frozenset({"unknown"}),
)

# Stored raw BRIC label space. This is intentionally a pinned literal.
TAXONOMY_RAW_35 = Taxonomy(
    name="raw_35",
    classes=(
        "Top_net_shot",
        "Top_return_net",
        "Top_smash",
        "Top_wrist_smash",
        "Top_lob",
        "Top_defensive_return_lob",
        "Top_clear",
        "Top_drive",
        "Top_back_court_drive",
        "Top_drop",
        "Top_passive_drop",
        "Top_push",
        "Top_rush",
        "Top_defensive_return_drive",
        "Top_cross_court_net_shot",
        "Top_short_service",
        "Top_long_service",
        "Bottom_net_shot",
        "Bottom_return_net",
        "Bottom_smash",
        "Bottom_wrist_smash",
        "Bottom_lob",
        "Bottom_defensive_return_lob",
        "Bottom_clear",
        "Bottom_drive",
        "Bottom_back_court_drive",
        "Bottom_drop",
        "Bottom_passive_drop",
        "Bottom_push",
        "Bottom_rush",
        "Bottom_defensive_return_drive",
        "Bottom_cross_court_net_shot",
        "Bottom_short_service",
        "Bottom_long_service",
        "unknown",
    ),
    merge_map=None,
    has_sides=True,
    excluded_base_stroke_types=frozenset({"driven_flight", "unknown"}),
    excluded_from_training=frozenset({"unknown"}),
)

DEFAULT_TAXONOMY = "une_merge_v1_nosides"

BST_X_TAXONOMIES: dict[str, Taxonomy] = {
    taxonomy.name: taxonomy
    for taxonomy in (
        TAXONOMY_BST_25,
        TAXONOMY_BST_24,
        TAXONOMY_BST_12,
        TAXONOMY_UNE_V1_14,
        TAXONOMY_UNE_V1_15,
        TAXONOMY_SHUTTLESET_18,
    )
}

TAXONOMIES: dict[str, Taxonomy] = {
    **BST_X_TAXONOMIES,
    TAXONOMY_UNE_MERGE_V1_NOSIDES.name: TAXONOMY_UNE_MERGE_V1_NOSIDES,
    TAXONOMY_RAW_35.name: TAXONOMY_RAW_35,
}


def taxonomy_lookup(name: str) -> Taxonomy:
    """Check the taxonomy is registered.  raises KeyError"""
    if name in TAXONOMIES:
        return TAXONOMIES[name]
    raise KeyError(f"taxonomy {name!r} not registered; known: {sorted(TAXONOMIES)}")


def derive_class_index(taxonomy: Taxonomy, raw_type: str, side: str) -> int | None:
    """The class index a stroke maps to under this taxonomy, or None if dropped.

    Three rules in order: drop the stroke when its raw type is in
    ``excluded_base_stroke_types``; merge rare subtypes via ``merge_map``
    (e.g. ``'driven_flight'`` -> ``'drive'``); then prepend ``Top_``/``Bottom_``
    for sided taxonomies (skipped when the merged type is in
    ``NOSIDE_CLASSES``).

    :param taxonomy: the taxonomy to label under.
    :param raw_type: ``raw_type_en`` from ``clips_master.csv``, e.g. ``'smash'``, ``'driven_flight'``.
    :param side: ``'Top'`` or ``'Bottom'``. Ignored on nosides taxonomies or when the merged type is side-agnostic.
    :return: index in ``[0, taxonomy.n_classes)``, or ``None`` if stroke is filtered out.
    """
    excluded = taxonomy.excluded_base_stroke_types or frozenset()
    if raw_type in excluded:
        return None

    merged = (taxonomy.merge_map or {}).get(raw_type, raw_type)  # unmapped types pass through
    label = (
        f"{side}_{merged}"
        if taxonomy.has_sides and merged not in NOSIDE_CLASSES
        else merged
    )
    try:
        return taxonomy.classes.index(label)
    except ValueError as error:
        raise ValueError(
            f"taxonomy {taxonomy.name!r}: derived label {label!r} "
            f"(raw_type={raw_type!r}, side={side!r}) not in classes "
            f"{list(taxonomy.classes)}"
        ) from error
