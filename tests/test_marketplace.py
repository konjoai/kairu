"""Tests for kairu.marketplace — MarketplaceStore, MarketplaceEntry, and API endpoints."""

from __future__ import annotations

import pytest

from kairu.marketplace import (
    DOMAINS,
    MarketplaceEntry,
    MarketplaceStore,
    compute_signature,
    open_default_marketplace_store,
    seed_community_rubrics,
)

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from httpx import ASGITransport, AsyncClient  # noqa: E402

from api.main import create_app  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests — MarketplaceStore
# ─────────────────────────────────────────────────────────────────────────────


def test_marketplace_store_construction() -> None:
    """MarketplaceStore can be created in-memory without error."""
    store = MarketplaceStore()
    assert store is not None
    store.close()


def test_publish_returns_entry_with_correct_fields() -> None:
    """publish() returns a MarketplaceEntry with all submitted values."""
    store = MarketplaceStore()
    entry = store.publish(
        name="test_rubric",
        version="1.0.0",
        domain="general",
        description="A test rubric.",
        rubric={"relevance": 0.5, "coherence": 0.5},
        source_url="https://example.com",
    )
    assert entry.name == "test_rubric"
    assert entry.version == "1.0.0"
    assert entry.domain == "general"
    assert entry.description == "A test rubric."
    assert entry.rubric == {"relevance": 0.5, "coherence": 0.5}
    assert entry.source_url == "https://example.com"
    assert isinstance(entry.signature, str) and len(entry.signature) == 64
    assert entry.created_utc > 0
    store.close()


def test_publish_invalid_domain_raises() -> None:
    """publish() raises ValueError when domain is not in DOMAINS."""
    store = MarketplaceStore()
    with pytest.raises(ValueError, match="domain must be"):
        store.publish("x", "1.0.0", "fantasy_land", "desc", {"relevance": 1.0})
    store.close()


def test_publish_empty_rubric_raises() -> None:
    """publish() raises ValueError when rubric dict is empty."""
    store = MarketplaceStore()
    with pytest.raises(ValueError, match="at least one criterion"):
        store.publish("x", "1.0.0", "general", "desc", {})
    store.close()


def test_get_returns_none_for_unknown_name() -> None:
    """get() returns None when the named rubric does not exist."""
    store = MarketplaceStore()
    assert store.get("nonexistent_rubric_xyz") is None
    store.close()


def test_get_returns_latest_version_when_version_omitted() -> None:
    """get() returns the most recently published entry when version is omitted."""
    store = MarketplaceStore()
    store.publish("multi", "1.0.0", "general", "first", {"relevance": 1.0})
    store.publish("multi", "2.0.0", "general", "second", {"coherence": 1.0})
    entry = store.get("multi")
    assert entry is not None
    assert entry.version == "2.0.0"
    store.close()


def test_get_by_name_and_version() -> None:
    """get() with an explicit version returns the matching entry."""
    store = MarketplaceStore()
    store.publish("versioned", "1.0.0", "general", "v1", {"relevance": 1.0})
    store.publish("versioned", "2.0.0", "general", "v2", {"coherence": 1.0})
    entry = store.get("versioned", version="1.0.0")
    assert entry is not None
    assert entry.version == "1.0.0"
    assert entry.description == "v1"
    store.close()


def test_list_entries_returns_all() -> None:
    """list_entries() without filters returns all published rubrics."""
    store = MarketplaceStore()
    store.publish("r1", "1.0.0", "medical", "desc1", {"relevance": 1.0})
    store.publish("r2", "1.0.0", "legal", "desc2", {"coherence": 1.0})
    entries = store.list_entries()
    names = {e.name for e in entries}
    assert {"r1", "r2"} == names
    store.close()


def test_list_entries_domain_filter() -> None:
    """list_entries(domain=) returns only entries in that domain."""
    store = MarketplaceStore()
    store.publish("m1", "1.0.0", "medical", "med", {"relevance": 1.0})
    store.publish("l1", "1.0.0", "legal", "leg", {"coherence": 1.0})
    medical = store.list_entries(domain="medical")
    assert len(medical) == 1
    assert medical[0].name == "m1"
    store.close()


def test_list_entries_keyword_search() -> None:
    """list_entries(q=) filters by name or description substring."""
    store = MarketplaceStore()
    store.publish("alpha_rubric", "1.0.0", "general", "something", {"relevance": 1.0})
    store.publish(
        "beta_rubric", "1.0.0", "general", "alpha flavour", {"coherence": 1.0}
    )
    store.publish("gamma_rubric", "1.0.0", "general", "other", {"fluency": 1.0})
    results = store.list_entries(q="alpha")
    names = {e.name for e in results}
    assert "alpha_rubric" in names
    assert "beta_rubric" in names
    assert "gamma_rubric" not in names
    store.close()


def test_marketplace_entry_is_frozen() -> None:
    """MarketplaceEntry is frozen — attribute assignment raises."""
    entry = MarketplaceEntry(
        name="x",
        version="1.0.0",
        domain="general",
        description="d",
        rubric={},
        signature="abc",
        source_url="",
        created_utc=1.0,
    )
    with pytest.raises((AttributeError, TypeError)):
        entry.name = "y"  # type: ignore[misc]


def test_compute_signature_is_deterministic() -> None:
    """compute_signature returns the same hash for the same inputs."""
    rubric = {"relevance": 0.5, "coherence": 0.5}
    sig1 = compute_signature("rubric_a", "1.0.0", rubric)
    sig2 = compute_signature("rubric_a", "1.0.0", rubric)
    assert sig1 == sig2
    assert len(sig1) == 64  # SHA-256 hex


def test_compute_signature_differs_for_different_inputs() -> None:
    """compute_signature differs when name, version, or rubric changes."""
    rubric = {"relevance": 1.0}
    s1 = compute_signature("a", "1.0.0", rubric)
    s2 = compute_signature("b", "1.0.0", rubric)
    s3 = compute_signature("a", "2.0.0", rubric)
    s4 = compute_signature("a", "1.0.0", {"coherence": 1.0})
    assert len({s1, s2, s3, s4}) == 4


def test_seed_community_rubrics_seeds_four_entries() -> None:
    """seed_community_rubrics populates a fresh store with 4 community rubrics."""
    store = MarketplaceStore()
    seed_community_rubrics(store)
    entries = store.list_entries()
    assert len(entries) == 4
    domains = {e.domain for e in entries}
    assert {"medical", "legal", "creative_writing", "code_review"} == domains
    store.close()


def test_seed_community_rubrics_is_idempotent() -> None:
    """seed_community_rubrics called twice does not create duplicate entries."""
    store = MarketplaceStore()
    seed_community_rubrics(store)
    seed_community_rubrics(store)
    assert len(store.list_entries()) == 4
    store.close()


def test_open_default_marketplace_store_returns_store() -> None:
    """open_default_marketplace_store() returns a MarketplaceStore."""
    store = open_default_marketplace_store()
    assert isinstance(store, MarketplaceStore)
    store.close()


def test_domains_constant_contains_expected_values() -> None:
    """DOMAINS contains all expected domain strings."""
    for domain in (
        "medical",
        "legal",
        "creative_writing",
        "code_review",
        "general",
        "safety",
        "education",
    ):
        assert domain in DOMAINS


# ─────────────────────────────────────────────────────────────────────────────
# API endpoint tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def marketplace_app():
    """Fresh app with an isolated in-memory marketplace store."""
    store = MarketplaceStore()
    seed_community_rubrics(store)
    application = create_app(marketplace_store=store)
    return application


@pytest.fixture
async def client(marketplace_app) -> AsyncClient:
    """Async HTTP client bound to the marketplace-enabled app."""
    async with AsyncClient(
        transport=ASGITransport(app=marketplace_app), base_url="http://t"
    ) as c:
        yield c


async def test_api_get_marketplace_returns_200_with_seeded_entries(
    client: AsyncClient,
) -> None:
    """GET /marketplace returns 200 with seeded community rubrics."""
    r = await client.get("/marketplace")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 4
    assert len(body["entries"]) == 4


async def test_api_get_marketplace_domain_filter(client: AsyncClient) -> None:
    """GET /marketplace?domain=medical returns only medical rubrics."""
    r = await client.get("/marketplace?domain=medical")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["entries"][0]["domain"] == "medical"


async def test_api_get_domains_returns_all_domains(client: AsyncClient) -> None:
    """GET /marketplace/domains returns the full domain list."""
    r = await client.get("/marketplace/domains")
    assert r.status_code == 200
    domains = r.json()["domains"]
    assert "medical" in domains
    assert "legal" in domains
    assert "code_review" in domains


async def test_api_get_marketplace_entry_by_name(client: AsyncClient) -> None:
    """GET /marketplace/{name} returns the matching entry."""
    r = await client.get("/marketplace/medical_qa")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "medical_qa"
    assert body["domain"] == "medical"
    assert "rubric" in body
    assert "signature" in body


async def test_api_get_marketplace_entry_404_for_unknown(client: AsyncClient) -> None:
    """GET /marketplace/{name} returns 404 for an unknown rubric."""
    r = await client.get("/marketplace/nonexistent_rubric_xyz")
    assert r.status_code == 404


async def test_api_post_marketplace_publishes_new_entry(client: AsyncClient) -> None:
    """POST /marketplace returns 200 with published=True."""
    r = await client.post(
        "/marketplace",
        json={
            "name": "edu_rubric",
            "version": "1.0.0",
            "domain": "education",
            "description": "Evaluates educational content quality.",
            "rubric": {"relevance": 0.4, "completeness": 0.4, "fluency": 0.2},
            "source_url": "",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["published"] is True
    assert body["name"] == "edu_rubric"
    assert body["domain"] == "education"


async def test_api_post_marketplace_invalid_domain_returns_422(
    client: AsyncClient,
) -> None:
    """POST /marketplace with an invalid domain returns 422."""
    r = await client.post(
        "/marketplace",
        json={
            "name": "bad_domain",
            "version": "1.0.0",
            "domain": "fictional",
            "description": "Bad domain.",
            "rubric": {"relevance": 1.0},
        },
    )
    assert r.status_code == 422


async def test_api_post_marketplace_import_adds_to_registry(
    client: AsyncClient,
) -> None:
    """POST /marketplace/{name}/import returns 200 with imported=True."""
    r = await client.post("/marketplace/medical_qa/import")
    assert r.status_code == 200
    body = r.json()
    assert body["imported"] is True
    assert body["name"] == "medical_qa"
    assert "criteria" in body
    assert "registry_version" in body


async def test_api_post_marketplace_import_404_for_unknown(client: AsyncClient) -> None:
    """POST /marketplace/{name}/import returns 404 for unknown entry."""
    r = await client.post("/marketplace/nonexistent_xyz/import")
    assert r.status_code == 404
