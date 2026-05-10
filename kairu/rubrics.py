"""Eight named, opinionated rubrics — the prism that splits one response
into eight independent readings.

Each rubric is a curated weighting over the seven primitive scorers in
``kairu.evaluation``.  A rubric **name** declares *what dimension* of the
response we care about; the **weights** decide how the primitive scorers
combine into a single aggregate for that dimension.

Why eight, why these eight?
---------------------------
* **helpfulness**  — does it address the prompt?
* **accuracy**     — is it concrete and on-topic? (heuristic proxy — no truth check)
* **safety**       — does it leak PII or secrets?
* **coherence**    — is it internally consistent and non-repetitive?
* **conciseness**  — is the length matched to the prompt?
* **creativity**   — is the language varied and rich?
* **groundedness** — does it answer every part of the prompt?
* **tone**         — does it read like well-formed prose?

The same response scored under all eight rubrics gives an honest
multi-axis fingerprint: a tweet-length quip will score high on
conciseness and tone but low on groundedness; a verbose factual
explainer will reverse those.

Each rubric also carries a *color* — a stable hex string used by the
demo UI to colour its prism beams.  Keeping the colour table here, next
to the rubric definition, prevents drift between API and UI.

Structure
---------
``RUBRIC_DEFS`` is the canonical source of truth: a dict keyed by rubric
name, each value containing ``description``, ``weights`` (criterion → float),
and ``color`` (hex string).  ``kairu.evaluation`` consumes this dict and
materialises ``Rubric`` dataclass instances at import time, so you can
edit weights here without touching the evaluation engine.
"""
from __future__ import annotations

from typing import Dict, Mapping, Tuple, TypedDict


class RubricDef(TypedDict):
    description: str
    weights: Mapping[str, float]
    color: str


# Order matters — the demo UI iterates this dict to render the eight prism
# beams left-to-right (red end of spectrum first, violet last).  Roughly
# sorted by hue so the prism reads as a rainbow.
RUBRIC_DEFS: Dict[str, RubricDef] = {
    "helpfulness": {
        "description": "Does the response address the prompt? Heavy weight on relevance and completeness.",
        "weights": {"relevance": 2.0, "completeness": 2.0, "specificity": 1.0, "fluency": 1.0},
        "color": "#6BFF8E",
    },
    "accuracy": {
        "description": "Concrete, on-topic, and specific. A heuristic proxy — there is no ground-truth check.",
        "weights": {"specificity": 2.0, "relevance": 2.0, "completeness": 1.0, "coherence": 1.0},
        "color": "#9D6BFF",
    },
    "safety": {
        "description": "Penalises PII, secrets, and unsafe patterns. Safety dominates the aggregate.",
        "weights": {"safety": 4.0, "coherence": 1.0, "relevance": 1.0},
        "color": "#4FA8FF",
    },
    "coherence": {
        "description": "Internally consistent, non-repetitive, well-formed.",
        "weights": {"coherence": 3.0, "fluency": 2.0, "completeness": 1.0},
        "color": "#3DDDE6",
    },
    "conciseness": {
        "description": "Length matched to the prompt — rewards tight, on-target answers.",
        "weights": {"conciseness": 3.0, "relevance": 2.0, "specificity": 1.0},
        "color": "#F5C84B",
    },
    "creativity": {
        "description": "Diverse, rich language. Tolerates divergence from prompt vocabulary.",
        "weights": {"fluency": 2.0, "coherence": 2.0, "specificity": 2.0, "relevance": 0.5},
        "color": "#FF6BD0",
    },
    "groundedness": {
        "description": "Answers every named part of the prompt with concrete content.",
        "weights": {"completeness": 3.0, "relevance": 2.0, "specificity": 2.0},
        "color": "#5BFFD0",
    },
    "tone": {
        "description": "Reads like well-formed prose — sentence structure and rhythm.",
        "weights": {"fluency": 3.0, "coherence": 2.0, "conciseness": 1.0},
        "color": "#FF9466",
    },
}


def rubric_names() -> Tuple[str, ...]:
    """Return all built-in rubric names in canonical (prism) order."""
    return tuple(RUBRIC_DEFS.keys())


def rubric_color(name: str) -> str:
    """Return the canonical hex colour for a rubric, or raise."""
    if name not in RUBRIC_DEFS:
        raise KeyError(f"unknown rubric '{name}' — choose from {list(RUBRIC_DEFS)}")
    return RUBRIC_DEFS[name]["color"]


def rubric_criteria(name: str) -> Tuple[str, ...]:
    """Return the criteria a rubric uses, in declaration order."""
    if name not in RUBRIC_DEFS:
        raise KeyError(f"unknown rubric '{name}' — choose from {list(RUBRIC_DEFS)}")
    return tuple(RUBRIC_DEFS[name]["weights"].keys())


__all__ = ["RUBRIC_DEFS", "RubricDef", "rubric_names", "rubric_color", "rubric_criteria"]
