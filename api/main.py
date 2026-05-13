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

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

import kairu
from kairu.audit import AuditLog, hash_inputs, open_default_audit
from kairu.benchmarks import BENCHMARKS, percentile_rank
from kairu.evaluation import (
    CRITERIA,
    RUBRICS,
    RUBRIC_REGISTRY,
    compare,
    evaluate,
    evaluate_batch,
    get_rubric_version,
    list_rubric_versions,
    register_rubric,
    to_csv,
)
from kairu.rubrics import RUBRIC_DEFS
from kairu.significance import paired_t_test, per_criterion_diffs

JUDGE_MODEL_ID: str = os.environ.get("KAIRU_JUDGE_MODEL", "kairu-heuristic-v1")

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


class RegisterRubricRequest(BaseModel):
    """Body for ``POST /rubrics`` — register a new rubric version.

    ``criteria`` and ``weights`` must agree on key set. ``version`` is
    optional; when omitted the patch level of ``base_version`` (or the
    latest known version) is auto-incremented.
    """
    name: str = Field(..., min_length=1, max_length=64)
    description: str = Field("", max_length=512)
    criteria: List[str] = Field(..., min_length=1)
    weights: Dict[str, float]
    version: Optional[str] = Field(None, pattern=r"^\d+\.\d+\.\d+$")
    base_version: Optional[str] = Field(None, pattern=r"^\d+\.\d+\.\d+$")


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


def create_app(audit: Optional[AuditLog] = None) -> FastAPI:
    app = FastAPI(
        title="kairu evaluation API",
        version=kairu.__version__,
        description="Rubric-based response evaluation and A/B comparison over HTTP.",
    )
    app.state.audit = audit if audit is not None else open_default_audit()

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
                    "version": r.version,
                    "versions": list(list_rubric_versions(r.name)),
                    "color": RUBRIC_DEFS[r.name]["color"] if r.name in RUBRIC_DEFS else None,
                }
                for r in RUBRICS.values()
            ],
            "criteria": [
                {"name": name, "description": desc}
                for name, (_, desc) in CRITERIA.items()
            ],
        }

    @app.post("/rubrics")
    def register_new_rubric(req: RegisterRubricRequest) -> Dict[str, object]:
        # Body validation that pydantic cannot express cleanly.
        if set(req.criteria) != set(req.weights):
            raise HTTPException(
                status_code=422,
                detail="criteria and weights must reference the same keys",
            )
        try:
            r = register_rubric(
                req.name,
                criteria=req.criteria,
                weights=req.weights,
                description=req.description or "",
                base_version=req.base_version,
                version=req.version,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "name": r.name,
            "version": r.version,
            "criteria": list(r.criteria),
            "weights": dict(r.weights),
            "description": r.description,
            "active": True,
            "all_versions": list(list_rubric_versions(r.name)),
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
        rubric_obj = RUBRICS.get(ev.rubric, RUBRICS["default"])
        out = ev.to_dict()
        out["rubric_version"] = rubric_obj.version
        # Per-criterion percentile vs reference corpus → drives the violin UI.
        out["benchmarks"] = {
            s.name: {
                "you": s.score,
                "rank": percentile_rank(s.name, s.score) if s.name in BENCHMARKS else None,
                "p25": BENCHMARKS[s.name].p25 if s.name in BENCHMARKS else None,
                "p50": BENCHMARKS[s.name].p50 if s.name in BENCHMARKS else None,
                "p75": BENCHMARKS[s.name].p75 if s.name in BENCHMARKS else None,
            }
            for s in ev.scores
        }
        audit_id = app.state.audit.record(
            input_hash=hash_inputs(req.prompt, req.response),
            rubric_name=ev.rubric,
            rubric_version=rubric_obj.version,
            judge_model=JUDGE_MODEL_ID,
            endpoint="/evaluate",
            scores={s.name: s.score for s in ev.scores},
            reasoning={"aggregate": ev.aggregate},
        )
        out["audit_id"] = audit_id
        return out

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

        out = cmp.to_dict()

        # Paired t-test on per-criterion scores: rejects a heuristic winner
        # when the difference is not statistically reliable.
        a_scores = {cc.name: cc.score_a for cc in cmp.per_criterion}
        b_scores = {cc.name: cc.score_b for cc in cmp.per_criterion}
        try:
            a_arr, b_arr, _ = per_criterion_diffs(a_scores, b_scores)
            sig = paired_t_test(a_arr, b_arr)
            out["significance"] = sig.to_dict()
            # Override the heuristic winner only when stats say tie.
            if sig.winner == "tie":
                out["statistical_winner"] = "tie"
            else:
                out["statistical_winner"] = sig.winner
        except ValueError as exc:
            # < 2 paired observations or zero overlap — surface as null block
            # rather than failing the whole comparison.
            out["significance"] = {"error": str(exc)}
            out["statistical_winner"] = None

        rubric_obj = RUBRICS.get(cmp.rubric, RUBRICS["default"])
        out["rubric_version"] = rubric_obj.version

        audit_id = app.state.audit.record(
            input_hash=hash_inputs(req.prompt, req.response_a, req.response_b),
            rubric_name=cmp.rubric,
            rubric_version=rubric_obj.version,
            judge_model=JUDGE_MODEL_ID,
            endpoint="/compare",
            scores={
                **{f"a.{k}": v for k, v in a_scores.items()},
                **{f"b.{k}": v for k, v in b_scores.items()},
            },
            reasoning={
                "winner_heuristic": cmp.winner,
                "winner_statistical": out.get("statistical_winner"),
                "margin": cmp.margin,
            },
        )
        out["audit_id"] = audit_id
        return out

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

    # ── Benchmark distributions ───────────────────────────────────────

    @app.get("/benchmarks")
    def benchmarks_index() -> Dict[str, object]:
        return {
            "criteria": list(BENCHMARKS.keys()),
            "corpus_size": next(iter(BENCHMARKS.values())).n if BENCHMARKS else 0,
        }

    @app.get("/benchmarks/{criterion}")
    def benchmarks_detail(criterion: str) -> Dict[str, object]:
        if criterion not in BENCHMARKS:
            raise HTTPException(
                status_code=404,
                detail=f"unknown criterion '{criterion}'",
            )
        return BENCHMARKS[criterion].to_dict()

    # ── Audit log ─────────────────────────────────────────────────────

    @app.get("/audit")
    def audit_query(
        start: Optional[str] = Query(None, description="ISO 8601 lower bound"),
        end:   Optional[str] = Query(None, description="ISO 8601 upper bound"),
        rubric_name:    Optional[str] = None,
        rubric_version: Optional[str] = None,
        limit:  int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, object]:
        log: AuditLog = app.state.audit
        try:
            rows = log.query(
                start=start, end=end,
                rubric_name=rubric_name, rubric_version=rubric_version,
                limit=limit, offset=offset,
            )
            total = log.count(
                start=start, end=end,
                rubric_name=rubric_name, rubric_version=rubric_version,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "items": [r.to_dict() for r in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    @app.get("/audit.csv")
    def audit_export_csv(
        start: Optional[str] = None,
        end:   Optional[str] = None,
        rubric_name:    Optional[str] = None,
        rubric_version: Optional[str] = None,
    ):
        log: AuditLog = app.state.audit
        body = log.export_csv(
            start=start, end=end,
            rubric_name=rubric_name, rubric_version=rubric_version,
        )
        return PlainTextResponse(content=body, media_type="text/csv")

    return app


app = create_app()
