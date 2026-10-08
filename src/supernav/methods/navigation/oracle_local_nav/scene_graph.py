"""Scene-graph record iteration, target-ref generation, and label matching."""

from __future__ import annotations

import re
from typing import Any, Mapping

from supernav.methods.navigation.oracle_local_nav.config import _STOPWORDS

def _object_records(scene_graph: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for item in scene_graph.get("objects") or []:
        if isinstance(item, Mapping):
            records.append(item)
    indexed = scene_graph.get("object_index_by_label")
    if isinstance(indexed, Mapping):
        for label, items in indexed.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, Mapping):
                    row = dict(item)
                    row.setdefault("label", label)
                    records.append(row)
    return records

def _object_target_ref(obj: Mapping[str, Any], index: int) -> str:
    for key in ("id", "object_id"):
        value = obj.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    label = _normalize_text(str(obj.get("label") or obj.get("category") or "object"))
    label = label.replace(" ", "_") or "object"
    return f"sgobj_{index}_{label}"

def _label_match_score(label: str, *, target_label: str, instruction: str) -> float:
    label_norm = _normalize_text(label)
    if not label_norm:
        return 0.0
    explicit = _normalize_text(target_label)
    if explicit:
        if label_norm == explicit:
            return 4.0
        if explicit in label_norm or label_norm in explicit:
            return 3.0
        if _token_overlap(label_norm, explicit):
            return 2.0
        return 0.0

    inst = _normalize_text(instruction)
    if not inst:
        return 0.0
    if label_norm in inst:
        return 2.5
    overlap = _token_overlap(label_norm, inst)
    if overlap:
        return 1.0 + overlap
    return 0.0

def _normalize_text(text: str) -> str:
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOPWORDS
    ]
    return " ".join(tokens)

def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)

