"""FastAPI HTTP layer over kairu.evaluation — rubric scoring + A/B compare.

Endpoints
---------
``POST /evaluate``  — score a single (prompt, response) pair under a rubric.
``POST /compare``   — A/B-compare two responses to the same prompt.
``GET  /rubrics``   — enumerate built-in rubrics and criteria.
``POST /batch``     — evaluate a list of pairs; JSON or text/csv response.
``GET  /health``    — liveness / version probe.

This is a *thin* layer.  Every endpoint validates input at the boundary, then
delegates to a real function in `kairu.evaluation`.  No business logic lives
in this file — the API is interchangeable with a CLI driver.

Boundary contracts (CLAUDE.md):
  * prompt + response capped at MAX_TEXT_CHARS to bound CPU per request.
  * batch capped at MAX_BATCH_ITEMS.
  * unknown rubric / criterion → 422 with the offending name in the message.
  * non-string fields → 422 (FastAPI / pydantic surface this for free).
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Mapping, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

import kairu
from kairu.evaluation import (
    CRITERIA,
    RUBRICS,
    compare,
    evaluate,
    evaluate_batch,
    to_csv,
)
from kairu.rubrics import RUBRIC_DEFS

logger = logging.getLogger("kairu.api")

MAX_TEXT_CHARS: int = int(os.environ.get("KAIRU_API_MAX_TEXT", "32768"))
MAX_BATCH_ITEMS: int = int(os.environ.get("KAIRU_API_MAX_BATCH", "256"))


# ─────────────────────────────────────────────────────────────────────────
# Request / response models
# ─────────────────────────────────────────────────────────────────────────


class EvaluateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    response: str = Field(..., min_length=1)
    rubric: Optional[str] = None
    criteria: Optional[List[str]] = None
    weights: Optional[Dict[str, float]] = None


class NamedRubricRequest(BaseModel):
    """Body for ``POST /evaluate/rubric/{name}`` — rubric is in the path."""
    prompt: str = Field(..., min_length=1)
    response: str = Field(..., min_length=1)
    weights: Optional[Dict[str, float]] = None


class CompareRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    response_a: str = Field(..., min_length=1)
    response_b: str = Field(..., min_length=1)
    rubric: Optional[str] = None
    criteria: Optional[List[str]] = None
    weights: Optional[Dict[str, float]] = None
    label_a: str = "a"
    label_b: str = "b"


class BatchItem(BaseModel):
    id: Optional[str] = None
    prompt: str = Field(..., min_length=1)
    response: str = Field(..., min_length=1)


class BatchRequest(BaseModel):
    items: List[BatchItem]
    rubric: Optional[str] = None
    criteria: Optional[List[str]] = None
    weights: Optional[Dict[str, float]] = None
    format: str = Field("json", pattern="^(json|csv)$")


# ─────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────


def _check_text(name: str, value: str) -> None:
    if len(value) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"{name} exceeds {MAX_TEXT_CHARS} chars (got {len(value)})",
        )


def _to_eval_kwargs(
    rubric: Optional[str],
    criteria: Optional[List[str]],
    weights: Optional[Mapping[str, float]],
) -> Dict[str, object]:
    out: Dict[str, object] = {}
    if rubric is not None:
        out["rubric"] = rubric
    if criteria is not None:
        out["criteria"] = criteria
    if weights is not None:
        out["weights"] = weights
    return out


# ─────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(
        title="kairu evaluation API",
        version=kairu.__version__,
        description="Rubric-based response evaluation and A/B comparison over HTTP.",
    )

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "version": kairu.__version__, "service": "kairu-eval"}

    @app.get("/rubrics")
    def rubrics() -> Dict[str, object]:
        return {
            "rubrics": [
                {
                    "name": r.name,
                    "description": r.description,
                    "criteria": list(r.criteria),
                    "weights": dict(r.weights),
                    "color": RUBRIC_DEFS[r.name]["color"] if r.name in RUBRIC_DEFS else None,
                }
                for r in RUBRICS.values()
            ],
            "criteria": [
                {"name": name, "description": desc}
                for name, (_, desc) in CRITERIA.items()
            ],
        }

    @app.get("/rubrics/{name}")
    def rubric_detail(name: str) -> Dict[str, object]:
        if name not in RUBRICS:
            raise HTTPException(status_code=404, detail=f"unknown rubric '{name}'")
        r = RUBRICS[name]
        return {
            "name": r.name,
            "description": r.description,
            "criteria": list(r.criteria),
            "weights": dict(r.weights),
            "color": RUBRIC_DEFS[name]["color"] if name in RUBRIC_DEFS else None,
        }

    @app.post("/evaluate")
    def evaluate_endpoint(req: EvaluateRequest) -> Dict[str, object]:
        _check_text("prompt", req.prompt)
        _check_text("response", req.response)
        try:
            ev = evaluate(
                req.prompt, req.response,
                **_to_eval_kwargs(req.rubric, req.criteria, req.weights),
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return ev.to_dict()

    @app.post("/evaluate/rubric/{name}")
    def evaluate_named_rubric(name: str, req: NamedRubricRequest) -> Dict[str, object]:
        if name not in RUBRICS:
            raise HTTPException(status_code=404, detail=f"unknown rubric '{name}'")
        _check_text("prompt", req.prompt)
        _check_text("response", req.response)
        try:
            ev = evaluate(req.prompt, req.response, rubric=name, weights=req.weights)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        out = ev.to_dict()
        out["color"] = RUBRIC_DEFS[name]["color"] if name in RUBRIC_DEFS else None
        return out

    @app.post("/compare")
    def compare_endpoint(req: CompareRequest) -> Dict[str, object]:
        _check_text("prompt", req.prompt)
        _check_text("response_a", req.response_a)
        _check_text("response_b", req.response_b)
        try:
            cmp = compare(
                req.prompt, req.response_a, req.response_b,
                label_a=req.label_a, label_b=req.label_b,
                **_to_eval_kwargs(req.rubric, req.criteria, req.weights),
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return cmp.to_dict()

    @app.post("/batch")
    def batch_endpoint(req: BatchRequest):
        if len(req.items) == 0:
            raise HTTPException(status_code=422, detail="items must be non-empty")
        if len(req.items) > MAX_BATCH_ITEMS:
            raise HTTPException(
                status_code=413,
                detail=f"batch exceeds {MAX_BATCH_ITEMS} items (got {len(req.items)})",
            )
        for item in req.items:
            _check_text("item.prompt", item.prompt)
            _check_text("item.response", item.response)
        try:
            rows = evaluate_batch(
                [item.model_dump() for item in req.items],
                **_to_eval_kwargs(req.rubric, req.criteria, req.weights),
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if req.format == "csv":
            return PlainTextResponse(content=to_csv(rows), media_type="text/csv")
        return {"results": rows, "count": len(rows)}

    return app


app = create_app()
