"""FastAPI router for the Rubric Marketplace — ``/marketplace`` endpoints.

Routes
------
``GET  /marketplace``             — list entries, optionally filtered by domain / keyword.
``GET  /marketplace/domains``     — enumerate valid domain tags.
``GET  /marketplace/{name}``      — fetch a specific entry (latest or ?version=).
``POST /marketplace``             — publish a new community rubric.
``POST /marketplace/{name}/import``
                                  — register a marketplace rubric into the local
                                    RUBRIC_REGISTRY so it can be used with ``/evaluate``.

Input limits (enforced at the boundary per CLAUDE.md security rules):
  * name ≤ 64 chars
  * description ≤ 512 chars
  * source_url ≤ 256 chars
  * rubric ≤ 20 criteria
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from kairu.evaluation import CRITERIA, register_rubric
from kairu.marketplace import DOMAINS, MarketplaceEntry, MarketplaceStore

__all__ = ["router"]

logger = logging.getLogger("kairu.api.marketplace")

MAX_NAME_LEN: int = 64
MAX_DESC_LEN: int = 512
MAX_URL_LEN: int = 256
MAX_RUBRIC_CRITERIA: int = 20

router = APIRouter(prefix="/marketplace", tags=["marketplace"])


# ── Pydantic models ───────────────────────────────────────────────────────────


class PublishRequest(BaseModel):
    """Body for ``POST /marketplace``."""

    name: str = Field(..., min_length=1, max_length=MAX_NAME_LEN)
    version: str = Field("1.0.0", max_length=20)
    domain: str = Field(..., max_length=32)
    description: str = Field(..., min_length=1, max_length=MAX_DESC_LEN)
    rubric: Dict[str, float]
    source_url: str = Field("", max_length=MAX_URL_LEN)


# ── helpers ───────────────────────────────────────────────────────────────────


def _entry_dict(entry: MarketplaceEntry) -> Dict[str, Any]:
    return {
        "name": entry.name,
        "version": entry.version,
        "domain": entry.domain,
        "description": entry.description,
        "rubric": entry.rubric,
        "signature": entry.signature[:16] + "…",  # truncated for display
        "signature_full": entry.signature,
        "source_url": entry.source_url,
        "created_utc": entry.created_utc,
    }


def _store(request: Request) -> MarketplaceStore:
    return request.app.state.marketplace  # type: ignore[no-any-return]


# ── routes ────────────────────────────────────────────────────────────────────


@router.get("")
def list_marketplace(
    request: Request,
    domain: Optional[str] = None,
    q: Optional[str] = None,
) -> Dict[str, Any]:
    """List community rubrics, newest first. Filter by ``domain`` and/or keyword ``q``."""
    entries = _store(request).list_entries(domain=domain, q=q)
    return {"entries": [_entry_dict(e) for e in entries], "count": len(entries)}


@router.get("/domains")
def list_domains() -> Dict[str, List[str]]:
    """Return the enumerated domain tags."""
    return {"domains": list(DOMAINS)}


@router.get("/{name}")
def get_marketplace_entry(
    request: Request,
    name: str,
    version: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch a specific rubric by name (and optionally version)."""
    entry = _store(request).get(name, version=version)
    if entry is None:
        raise HTTPException(
            status_code=404, detail=f"Marketplace rubric '{name}' not found"
        )
    return _entry_dict(entry)


@router.post("")
def publish_rubric(request: Request, req: PublishRequest) -> Dict[str, Any]:
    """Publish a new community rubric. Domain must be one of the enumerated tags."""
    if req.domain not in DOMAINS:
        raise HTTPException(
            status_code=422,
            detail=f"domain must be one of {list(DOMAINS)}",
        )
    if not req.rubric:
        raise HTTPException(
            status_code=422, detail="rubric must contain at least one criterion"
        )
    if len(req.rubric) > MAX_RUBRIC_CRITERIA:
        raise HTTPException(
            status_code=422,
            detail=f"rubric may have at most {MAX_RUBRIC_CRITERIA} criteria",
        )
    try:
        entry = _store(request).publish(
            name=req.name,
            version=req.version,
            domain=req.domain,
            description=req.description,
            rubric=req.rubric,
            source_url=req.source_url,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    logger.info(
        "marketplace: published %s@%s domain=%s", req.name, req.version, req.domain
    )
    return {"published": True, **_entry_dict(entry)}


@router.post("/{name}/import")
def import_rubric(
    request: Request,
    name: str,
    version: Optional[str] = None,
) -> Dict[str, Any]:
    """Import a marketplace rubric into the local RUBRIC_REGISTRY.

    After a successful import the rubric is usable via ``POST /evaluate`` with
    ``rubric=<name>``. Fails with 422 if the entry references criteria that are
    not built into kairu's scorer set.
    """
    entry = _store(request).get(name, version=version)
    if entry is None:
        raise HTTPException(
            status_code=404, detail=f"Marketplace rubric '{name}' not found"
        )
    unknown = [c for c in entry.rubric if c not in CRITERIA]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown criteria: {unknown}. Available: {sorted(CRITERIA)}",
        )
    try:
        rubric = register_rubric(
            entry.name,
            criteria=list(entry.rubric.keys()),
            weights=entry.rubric,
            description=entry.description,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    logger.info(
        "marketplace: imported %s@%s into RUBRIC_REGISTRY as version %s",
        entry.name,
        entry.version,
        rubric.version,
    )
    return {
        "imported": True,
        "name": entry.name,
        "marketplace_version": entry.version,
        "registry_version": rubric.version,
        "criteria": list(entry.rubric.keys()),
        "signature": entry.signature[:16] + "…",
    }
