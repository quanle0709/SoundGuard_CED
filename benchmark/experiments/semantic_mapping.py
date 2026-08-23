"""Offline formal taxonomy loader and deterministic candidate mapper."""

from __future__ import annotations

import json
from pathlib import Path


TAXONOMY_PATH = Path(__file__).with_name("soundguard_taxonomy.json")
ALLOWED_MAPPING_TYPES = {"EXACT", "STRONG_SEMANTIC_ALIAS"}


def load_taxonomy() -> dict:
    return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))


def active_rules() -> dict[str, dict]:
    return {row["raw_label"]: row for row in load_taxonomy()["rules"]
            if row["mapping_type"] in ALLOWED_MAPPING_TYPES and row["soundguard_category"]}


def map_candidate(raw_label: str, baseline_category: str | None = None) -> tuple[str | None, dict | None]:
    rule = active_rules().get(str(raw_label))
    if rule:
        return rule["soundguard_category"], rule
    return baseline_category, None
