"""Constants, thresholds, synonyms and stopwords."""

from __future__ import annotations

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "area",
        "at",
        "go",
        "head",
        "in",
        "into",
        "move",
        "near",
        "next",
        "of",
        "on",
        "the",
        "to",
        "toward",
        "towards",
        "visible",
    }
)

_DEFAULT_GROUNDING_URL = "http://127.0.0.1:18913/ground"

_DEFAULT_GROUNDING_TIMEOUT_S = 8.0

_DEFAULT_GROUNDING_SCORE_THRESHOLD = 0.30

_DEFAULT_CROP_EXPAND_RATIO = 0.30

_DEFAULT_MAX_GROUNDING_CROPS = 0

_DEFAULT_GROUNDING_TOP_K = 5

_VISUAL_PHRASE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "sofa": ("sofa", "couch", "loveseat"),
    "couch": ("sofa", "couch", "loveseat"),
    "loveseat": ("sofa", "couch", "loveseat"),
    "chair": ("chair", "armchair", "dining chair"),
    "armchair": ("chair", "armchair"),
    "picture": ("picture", "painting", "framed picture", "wall art"),
    "painting": ("picture", "painting", "framed picture", "wall art"),
    "table": ("table", "coffee table", "console table", "dining table"),
    "door": ("door", "doorway", "open doorway"),
    "doorway": ("door", "doorway", "open doorway"),
}

_SEMANTIC_GROUP_TERMS: dict[str, tuple[str, ...]] = {
    "seating": (
        "chair",
        "armchair",
        "sofa",
        "couch",
        "loveseat",
        "stool",
        "bench",
        "seat",
        "ottoman",
    ),
    "surface": (
        "table",
        "desk",
        "counter",
        "countertop",
        "kitchen island",
        "coffee table",
    ),
    "floor": ("rug", "carpet", "mat"),
    "opening": ("door", "doorway", "entrance", "corridor", "opening"),
    "structure": ("wall", "partition", "railing", "pillar", "window"),
    "decor": ("painting", "picture", "wall art", "decorative painting"),
    "lighting": ("lamp", "light", "spotlight"),
    "small_object": ("fruit", "pear", "dish", "bowl"),
}

_SEATING_HARD_REJECT_GROUPS = frozenset(
    {"structure", "decor", "lighting", "small_object"}
)

