"""Classifier stroke taxonomies and label derivation.

The BST-X taxonomies are authoritative. Two legacy BRIC registry entries are
kept as explicit class tuples because stored model metadata depends on their
names and ordering.
"""

from dataclasses import dataclass


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

MERGE_MAP_25: dict[str, str] = {
    "wrist_smash": "smash",
    "defensive_return_lob": "lob",
    "driven_flight": "drive",
    "back_court_drive": "drive",
    "passive_drop": "drop",
    "defensive_return_drive": "drive",
}

UNE_MERGE_V1_MAP: dict[str, str] = {
    "defensive_return_lob": "lob",
    "driven_flight": "drive",
    "back_court_drive": "drive",
    "defensive_return_drive": "drive",
}

NOSIDE_CLASSES: frozenset[str] = frozenset({"unknown"})


@dataclass(frozen=True)
class Taxonomy:
    """A pinned classifier label space."""

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
        return list(self.classes)

    def trainable_class_list(self) -> list[str]:
        excluded = self.excluded_from_training or frozenset()
        return [label for label in self.classes if label not in excluded]

    @property
    def n_trainable_classes(self) -> int:
        return len(self.trainable_class_list())


def _sided_classes(base: list[str], with_unknown: bool) -> tuple[str, ...]:
    classes = [f"Top_{label}" for label in base]
    classes.extend(f"Bottom_{label}" for label in base)
    if with_unknown:
        classes.append("unknown")
    return tuple(classes)


TAXONOMY_BST_25 = Taxonomy(
    name="bst_25",
    classes=_sided_classes(STROKE_TYPES_12_MERGED, with_unknown=True),
    merge_map=MERGE_MAP_25,
    has_sides=True,
    excluded_base_stroke_types=None,
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

# Stored BRIC contract. Keep this literal order aligned with deployed manifests.
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

TAXONOMIES: dict[str, Taxonomy] = {
    taxonomy.name: taxonomy
    for taxonomy in (
        TAXONOMY_BST_25,
        TAXONOMY_BST_24,
        TAXONOMY_BST_12,
        TAXONOMY_UNE_V1_14,
        TAXONOMY_UNE_V1_15,
        TAXONOMY_SHUTTLESET_18,
        TAXONOMY_UNE_MERGE_V1_NOSIDES,
        TAXONOMY_RAW_35,
    )
}


def taxonomy_lookup(name: str) -> Taxonomy:
    if name in TAXONOMIES:
        return TAXONOMIES[name]
    raise KeyError(f"taxonomy {name!r} not registered; known: {sorted(TAXONOMIES)}")


def derive_class_index(taxonomy: Taxonomy, raw_type: str, side: str) -> int | None:
    """Return a class index after exclusion, merging, and side prefixing."""
    excluded = taxonomy.excluded_base_stroke_types or frozenset()
    if raw_type in excluded:
        return None

    merged = (taxonomy.merge_map or {}).get(raw_type, raw_type)
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
