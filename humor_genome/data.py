"""Loader for the curated demo joke set."""

from __future__ import annotations

import json
import os
from typing import List, Dict

_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "examples", "jokes.json"
)


def load_examples() -> List[Dict[str, str]]:
    try:
        with open(_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
