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
    compare,
    evaluate,
    evaluate_batch,
    list_rubric_versions,
    register_rubric,
    to_csv,
)
from kairu.ci_regression import (
    BaselineStore,
    DEFAULT_REGRESSION_THRESHOLD,
    check_against_baseline,
    open_default_store,
    snapshot_baseline,
)
from kairu.ensemble import (
    DEFAULT_DISAGREEMENT_THRESHOLD,
    EnsembleResult,
    JudgeConfig,
    ensemble_compare,
    ensemble_evaluate,
)
from kairu.calibration import (
    BiasProfile,
    BiasProfileStore,
    CalibrationPair,
    CalibratedEnsembleResult,
    build_bias_profile,
    compute_uncalibrated_bias_bound,
    correct_ensemble_scores,
)
from kairu.constitutional import GeneratedRubric, generate_rubric
from kairu.log_eval import DEFAULT_LOG_THRESHOLD, evaluate_log
from kairu.rubrics import RUBRIC_DEFS
from kairu.trajectory import TrajectoryStep, evaluate_trajectory
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


# ── v0.16 ensemble + CI + log-eval request models ────────────────────────


class JudgeConfigRequest(BaseModel):
    """Wire shape for a single judge's perspective."""

    name: str = Field(..., min_length=1, max_length=64)
    rubric: Optional[str] = "default"
    criteria: Optional[List[str]] = None
    weights: Optional[Dict[str, float]] = None
    seed: Optional[int] = None
    noise: float = Field(0.0, ge=0.0, le=1.0)

    def to_config(self) -> JudgeConfig:
        return JudgeConfig(
            name=self.name,
            rubric=self.rubric,
            criteria=tuple(self.criteria) if self.criteria else None,
            weights=self.weights,
            seed=self.seed,
            noise=self.noise,
        )


class EnsembleEvaluateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    response: str = Field(..., min_length=1)
    judges: List[JudgeConfigRequest] = Field(..., min_length=1, max_length=16)
    disagreement_threshold: float = Field(
        DEFAULT_DISAGREEMENT_THRESHOLD, ge=0.0, le=1.0
    )


class EnsembleCompareRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    response_a: str = Field(..., min_length=1)
    response_b: str = Field(..., min_length=1)
    judges: List[JudgeConfigRequest] = Field(..., min_length=1, max_length=16)
    disagreement_threshold: float = Field(
        DEFAULT_DISAGREEMENT_THRESHOLD, ge=0.0, le=1.0
    )


class CIItem(BaseModel):
    input: str = Field(..., min_length=1)
    output: str = Field(..., min_length=1)


class CIBaselineRequest(BaseModel):
    items: List[CIItem] = Field(..., min_length=1)
    rubric: Optional[str] = None
    criteria: Optional[List[str]] = None
    weights: Optional[Dict[str, float]] = None
    label: str = Field("", max_length=256)


class CICheckRequest(BaseModel):
    snapshot_id: str = Field(..., min_length=1, max_length=128)
    items: List[CIItem] = Field(..., min_length=1)
    threshold: float = Field(DEFAULT_REGRESSION_THRESHOLD, ge=0.0, le=1.0)
    rubric: Optional[str] = None
    criteria: Optional[List[str]] = None
    weights: Optional[Dict[str, float]] = None


class LogEvalItem(BaseModel):
    input: str = Field(..., min_length=1)
    output: str = Field(..., min_length=1)
    metadata: Optional[Dict[str, str]] = None


class LogEvalRequest(BaseModel):
    items: List[LogEvalItem] = Field(..., min_length=1)
    rubric: Optional[str] = None
    criteria: Optional[List[str]] = None
    weights: Optional[Dict[str, float]] = None
    threshold: float = Field(DEFAULT_LOG_THRESHOLD, ge=0.0, le=1.0)


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


class GenerateRubricRequest(BaseModel):
    """Body for ``POST /rubrics/generate`` — extract rubric from policy text."""

    text: str = Field(..., min_length=10)
    name: str = Field(..., min_length=1, max_length=64)
    max_clauses: int = Field(20, ge=1, le=50)


class TrajectoryStepRequest(BaseModel):
    """Wire shape for one step in an agentic trajectory."""

    step: int = Field(..., ge=0)
    tool_call: Optional[str] = None
    observation: Optional[str] = None
    response: str = Field(..., min_length=1)

    def to_step(self) -> TrajectoryStep:
        """Convert to the library dataclass."""
        return TrajectoryStep(
            step=self.step,
            tool_call=self.tool_call,
            observation=self.observation,
            response=self.response,
        )


class TrajectoryRequest(BaseModel):
    """Body for ``POST /eval/trajectory``."""

    goal: str = Field(..., min_length=1)
    steps: List[TrajectoryStepRequest] = Field(..., min_length=1, max_length=100)
    optimal_steps: Optional[int] = Field(None, ge=1)


# ── v0.18 calibration request models ─────────────────────────────────────


class CalibrationPairRequest(BaseModel):
    """Wire shape for one human-annotated calibration example."""

    prompt: str = Field(..., min_length=1)
    response: str = Field(..., min_length=1)
    human_scores: Dict[str, float] = Field(..., min_length=1)

    def to_pair(self) -> CalibrationPair:
        return CalibrationPair(
            prompt=self.prompt,
            response=self.response,
            human_scores=self.human_scores,
        )


class BuildBiasProfileRequest(BaseModel):
    """Body for ``POST /calibration`` — build and store a bias profile."""

    judges: List[JudgeConfigRequest] = Field(..., min_length=1, max_length=16)
    calibration_pairs: List[CalibrationPairRequest] = Field(
        ..., min_length=1, max_length=256
    )
    rubric: str = Field("default", min_length=1, max_length=64)
    disagreement_threshold: float = Field(
        DEFAULT_DISAGREEMENT_THRESHOLD, ge=0.0, le=1.0
    )


class CalibratedEnsembleRequest(BaseModel):
    """Body for ``POST /evaluate/ensemble/calibrated``."""

    prompt: str = Field(..., min_length=1)
    response: str = Field(..., min_length=1)
    judges: List[JudgeConfigRequest] = Field(..., min_length=1, max_length=16)
    disagreement_threshold: float = Field(
        DEFAULT_DISAGREEMENT_THRESHOLD, ge=0.0, le=1.0
    )
    rubric: str = Field("default", min_length=1, max_length=64)


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


def create_app(
    audit: Optional[AuditLog] = None,
    baseline_store: Optional[BaselineStore] = None,
    bias_profile_store: Optional[BiasProfileStore] = None,
) -> FastAPI:
    app = FastAPI(
        title="kairu evaluation API",
        version=kairu.__version__,
        description="Rubric-based response evaluation and A/B comparison over HTTP.",
    )
    app.state.audit = audit if audit is not None else open_default_audit()
    app.state.baselines = (
        baseline_store if baseline_store is not None else open_default_store()
    )
    app.state.bias_profiles = (
        bias_profile_store if bias_profile_store is not None else BiasProfileStore()
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
                    "version": r.version,
                    "versions": list(list_rubric_versions(r.name)),
                    "color": RUBRIC_DEFS[r.name]["color"]
                    if r.name in RUBRIC_DEFS
                    else None,
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
                req.prompt,
                req.response,
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
                "rank": percentile_rank(s.name, s.score)
                if s.name in BENCHMARKS
                else None,
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
                req.prompt,
                req.response_a,
                req.response_b,
                label_a=req.label_a,
                label_b=req.label_b,
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
        end: Optional[str] = Query(None, description="ISO 8601 upper bound"),
        rubric_name: Optional[str] = None,
        rubric_version: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, object]:
        log: AuditLog = app.state.audit
        try:
            rows = log.query(
                start=start,
                end=end,
                rubric_name=rubric_name,
                rubric_version=rubric_version,
                limit=limit,
                offset=offset,
            )
            total = log.count(
                start=start,
                end=end,
                rubric_name=rubric_name,
                rubric_version=rubric_version,
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
        end: Optional[str] = None,
        rubric_name: Optional[str] = None,
        rubric_version: Optional[str] = None,
    ):
        log: AuditLog = app.state.audit
        body = log.export_csv(
            start=start,
            end=end,
            rubric_name=rubric_name,
            rubric_version=rubric_version,
        )
        return PlainTextResponse(content=body, media_type="text/csv")

    # ── v0.16 judge ensemble ──────────────────────────────────────────

    def _validate_judges(judges: List[JudgeConfigRequest]) -> List[JudgeConfig]:
        names = [j.name for j in judges]
        if len(set(names)) != len(names):
            raise HTTPException(status_code=422, detail="judge names must be unique")
        configs: List[JudgeConfig] = []
        for j in judges:
            try:
                configs.append(j.to_config())
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        return configs

    @app.post("/evaluate/ensemble")
    def evaluate_ensemble_endpoint(req: EnsembleEvaluateRequest):
        _check_text("prompt", req.prompt)
        _check_text("response", req.response)
        judges = _validate_judges(req.judges)
        try:
            result = ensemble_evaluate(
                req.prompt,
                req.response,
                judges,
                disagreement_threshold=req.disagreement_threshold,
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return result.to_dict()

    @app.post("/compare/ensemble")
    def compare_ensemble_endpoint(req: EnsembleCompareRequest):
        _check_text("prompt", req.prompt)
        _check_text("response_a", req.response_a)
        _check_text("response_b", req.response_b)
        judges = _validate_judges(req.judges)
        try:
            result = ensemble_compare(
                req.prompt,
                req.response_a,
                req.response_b,
                judges,
                disagreement_threshold=req.disagreement_threshold,
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return result.to_dict()

    # ── v0.16 CI regression gating ────────────────────────────────────

    def _validate_ci_items(items: List[CIItem]) -> List[Dict[str, str]]:
        if len(items) > MAX_BATCH_ITEMS:
            raise HTTPException(
                status_code=413,
                detail=f"items exceed {MAX_BATCH_ITEMS} (got {len(items)})",
            )
        for i, it in enumerate(items):
            _check_text(f"items[{i}].input", it.input)
            _check_text(f"items[{i}].output", it.output)
        return [{"input": it.input, "output": it.output} for it in items]

    @app.post("/ci/baseline")
    def ci_baseline_endpoint(req: CIBaselineRequest):
        items = _validate_ci_items(req.items)
        try:
            snap = snapshot_baseline(
                items,
                rubric=req.rubric,
                criteria=req.criteria,
                weights=req.weights,
                judge_model=JUDGE_MODEL_ID,
                label=req.label,
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        store: BaselineStore = app.state.baselines
        store.save(snap)
        return snap.to_dict()

    @app.get("/ci/baselines")
    def ci_baselines_index() -> Dict[str, object]:
        store: BaselineStore = app.state.baselines
        ids = store.list()
        return {
            "snapshots": [
                {
                    "snapshot_id": sid,
                    "created_utc": store.load(sid).created_utc,
                    "rubric_name": store.load(sid).rubric_name,
                    "rubric_version": store.load(sid).rubric_version,
                    "n_items": store.load(sid).n_items,
                    "mean_aggregate": store.load(sid).mean_aggregate,
                    "label": store.load(sid).label,
                }
                for sid in ids
            ],
            "count": len(ids),
        }

    @app.get("/ci/baselines/{snapshot_id}")
    def ci_baselines_get(snapshot_id: str) -> Dict[str, object]:
        store: BaselineStore = app.state.baselines
        try:
            snap = store.load(snapshot_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"snapshot '{snapshot_id}' not found",
            ) from exc
        return snap.to_dict()

    @app.post("/ci/check")
    def ci_check_endpoint(req: CICheckRequest):
        items = _validate_ci_items(req.items)
        store: BaselineStore = app.state.baselines
        try:
            baseline = store.load(req.snapshot_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"snapshot '{req.snapshot_id}' not found",
            ) from exc
        try:
            report = check_against_baseline(
                baseline,
                items,
                threshold=req.threshold,
                rubric=req.rubric,
                criteria=req.criteria,
                weights=req.weights,
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return report.to_dict()

    # ── v0.16 log → eval pipeline ─────────────────────────────────────

    @app.post("/eval_from_log")
    def eval_from_log_endpoint(req: LogEvalRequest):
        if len(req.items) > MAX_BATCH_ITEMS:
            raise HTTPException(
                status_code=413,
                detail=f"items exceed {MAX_BATCH_ITEMS} (got {len(req.items)})",
            )
        for i, it in enumerate(req.items):
            _check_text(f"items[{i}].input", it.input)
            _check_text(f"items[{i}].output", it.output)
        # Flatten the request schema's nested ``metadata`` dict to match
        # evaluate_log's "anything that isn't input/output is metadata" rule.
        items_dicts: List[Dict[str, object]] = []
        for it in req.items:
            entry: Dict[str, object] = {"input": it.input, "output": it.output}
            if it.metadata:
                for k, v in it.metadata.items():
                    if k in ("input", "output"):
                        continue
                    entry[k] = v
            items_dicts.append(entry)
        try:
            report = evaluate_log(
                items_dicts,
                rubric=req.rubric,
                criteria=req.criteria,
                weights=req.weights,
                threshold=req.threshold,
                judge_model=JUDGE_MODEL_ID,
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return report.to_dict()

    # ── v0.17 constitutional rubric generation ────────────────────────

    @app.post("/rubrics/generate")
    def rubrics_generate(req: GenerateRubricRequest) -> Dict[str, object]:
        _check_text("text", req.text)
        try:
            result: GeneratedRubric = generate_rubric(
                req.text, req.name, max_clauses=req.max_clauses
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "name": result.name,
            "n_clauses": result.n_clauses,
            "n_positive": result.n_positive,
            "n_negative": result.n_negative,
            "criteria": list(result.criteria),
            "weights": dict(result.weights),
            "clauses": [
                {
                    "text": c.text,
                    "polarity": c.polarity,
                    "trigger": c.trigger,
                    "criterion_name": c.criterion_name,
                    "weight": c.weight,
                }
                for c in result.clauses
            ],
        }

    # ── v0.17 agentic trajectory scoring ─────────────────────────────

    @app.post("/eval/trajectory")
    def eval_trajectory(req: TrajectoryRequest) -> Dict[str, object]:
        _check_text("goal", req.goal)
        for i, s in enumerate(req.steps):
            if s.tool_call is not None:
                _check_text(f"steps[{i}].tool_call", s.tool_call)
            if s.observation is not None:
                _check_text(f"steps[{i}].observation", s.observation)
            _check_text(f"steps[{i}].response", s.response)
        try:
            result = evaluate_trajectory(
                req.goal,
                [s.to_step() for s in req.steps],
                optimal_steps=req.optimal_steps,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "goal": result.goal,
            "n_steps": result.n_steps,
            "tool_selection": result.tool_selection,
            "error_recovery": result.error_recovery,
            "goal_completion": result.goal_completion,
            "efficiency": result.efficiency,
            "aggregate": result.aggregate,
            "steps": [
                {
                    "step": ss.step,
                    "tool_selection": ss.tool_selection,
                    "error_recovery": ss.error_recovery,
                    "goal_progress": ss.goal_progress,
                    "efficiency": ss.efficiency,
                    "score": ss.score,
                }
                for ss in result.steps
            ],
        }

    # ── v0.18 judge bias calibration ──────────────────────────────────

    @app.post("/calibration")
    def build_calibration_endpoint(req: BuildBiasProfileRequest) -> Dict[str, object]:
        """Build a ``BiasProfile`` from human-labeled pairs and store it."""
        judges = _validate_judges(req.judges)
        pairs = []
        for i, p in enumerate(req.calibration_pairs):
            _check_text(f"calibration_pairs[{i}].prompt", p.prompt)
            _check_text(f"calibration_pairs[{i}].response", p.response)
            try:
                pairs.append(p.to_pair())
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            profile: BiasProfile = build_bias_profile(
                judges,
                pairs,
                rubric=req.rubric,
                disagreement_threshold=req.disagreement_threshold,
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        store: BiasProfileStore = app.state.bias_profiles
        store.save(profile)
        return profile.to_dict()

    @app.get("/calibration/{rubric}")
    def get_calibration_endpoint(rubric: str) -> Dict[str, object]:
        """Retrieve the stored ``BiasProfile`` for a rubric."""
        store: BiasProfileStore = app.state.bias_profiles
        profile = store.load(rubric)
        if profile is None:
            raise HTTPException(
                status_code=404,
                detail=f"no calibration profile found for rubric '{rubric}'",
            )
        return profile.to_dict()

    @app.get("/calibration")
    def list_calibration_endpoint() -> Dict[str, object]:
        """List rubrics that have stored bias profiles."""
        store: BiasProfileStore = app.state.bias_profiles
        return {"rubrics": store.list(), "count": len(store.list())}

    @app.post("/evaluate/ensemble/calibrated")
    def evaluate_ensemble_calibrated_endpoint(
        req: CalibratedEnsembleRequest,
    ) -> Dict[str, object]:
        """Run ensemble evaluation with stored bias-profile correction.

        Loads the ``BiasProfile`` for ``req.rubric``.  If no profile is
        stored, returns the raw ensemble result plus an
        ``uncalibrated_bias_bound`` derived from inter-judge stdev alone.
        """
        _check_text("prompt", req.prompt)
        _check_text("response", req.response)
        judges = _validate_judges(req.judges)
        try:
            raw: EnsembleResult = ensemble_evaluate(
                req.prompt,
                req.response,
                judges,
                disagreement_threshold=req.disagreement_threshold,
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        store: BiasProfileStore = app.state.bias_profiles
        profile = store.load(req.rubric)

        if profile is None:
            out = raw.to_dict()
            out["calibrated"] = False
            out["uncalibrated_bias_bound"] = compute_uncalibrated_bias_bound(
                raw.stdev_scores
            )
            return out

        calibrated: CalibratedEnsembleResult = correct_ensemble_scores(raw, profile)
        out = calibrated.to_dict()
        out["calibrated"] = True
        return out

    return app


app = create_app()
