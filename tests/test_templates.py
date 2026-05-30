"""Tests for kairu.templates — SQLite-backed evaluation template store."""
from __future__ import annotations

from pathlib import Path

import pytest

from kairu.templates import EvaluationTemplate, TemplateStore


def test_save_then_get_roundtrip():
    store = TemplateStore()
    tpl = store.save(
        "my-template",
        description="balanced default",
        rubric="default",
        weights={"relevance": 1.5},
    )
    assert isinstance(tpl, EvaluationTemplate)
    assert tpl.name == "my-template"
    assert tpl.rubric == "default"
    assert tpl.weights == {"relevance": 1.5}
    assert tpl.created_utc == tpl.updated_utc  # first write

    got = store.get("my-template")
    assert got.name == tpl.name
    assert got.rubric == tpl.rubric


def test_save_replaces_existing_and_preserves_created_utc():
    store = TemplateStore()
    t1 = store.save("foo", rubric="default")
    t2 = store.save("foo", rubric="helpfulness")
    assert t2.created_utc == t1.created_utc
    assert t2.updated_utc >= t1.updated_utc
    assert t2.rubric == "helpfulness"


def test_get_unknown_raises_keyerror():
    store = TemplateStore()
    with pytest.raises(KeyError):
        store.get("no-such-template")


def test_delete_returns_bool():
    store = TemplateStore()
    store.save("doomed", rubric="default")
    assert store.delete("doomed") is True
    assert store.delete("doomed") is False  # already gone


def test_list_orders_newest_first():
    store = TemplateStore()
    store.save("first",  rubric="default")
    store.save("second", rubric="default")
    items = store.list()
    assert [t.name for t in items[:2]] == ["second", "first"]


def test_template_with_judges_materialises_configs():
    store = TemplateStore()
    tpl = store.save(
        "ensemble-default", rubric="default",
        judges=[
            {"name": "j1", "rubric": "default"},
            {"name": "j2", "rubric": "helpfulness", "noise": 0.05},
        ],
    )
    cfgs = tpl.judge_configs()
    assert cfgs is not None and len(cfgs) == 2
    assert cfgs[0].name == "j1"
    assert cfgs[1].noise == 0.05


def test_save_rejects_empty_or_oversized_name():
    store = TemplateStore()
    with pytest.raises(ValueError):
        store.save("", rubric="default")
    with pytest.raises(ValueError):
        store.save("a" * 65, rubric="default")


def test_save_rejects_bad_name_chars():
    store = TemplateStore()
    with pytest.raises(ValueError):
        store.save("with spaces", rubric="default")
    with pytest.raises(ValueError):
        store.save("slash/path", rubric="default")


def test_save_requires_at_least_one_dimension():
    store = TemplateStore()
    with pytest.raises(ValueError):
        store.save("empty", description="nothing here")


def test_persistence_across_store_instances(tmp_path: Path):
    db = str(tmp_path / "templates.db")
    s1 = TemplateStore(db)
    s1.save("persisted", rubric="default")
    s1.close()
    s2 = TemplateStore(db)
    assert s2.get("persisted").rubric == "default"
    s2.close()


def test_to_dict_is_json_serialisable():
    import json
    store = TemplateStore()
    tpl = store.save("json-ok", rubric="default", criteria=["relevance", "fluency"])
    json.dumps(tpl.to_dict())  # no raise


def test_len_tracks_writes():
    store = TemplateStore()
    assert len(store) == 0
    store.save("a", rubric="default")
    store.save("b", rubric="default")
    assert len(store) == 2
