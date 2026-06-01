"""Tests for kairu.prompts — saved prompt library."""
from __future__ import annotations

from pathlib import Path

import pytest

from kairu.prompts import Prompt, PromptStore


def test_save_then_get_roundtrip():
    store = PromptStore()
    p = store.save("greet", text="hello world", description="basic test", tags=["smoke"])
    assert isinstance(p, Prompt)
    assert p.name == "greet"
    assert p.text == "hello world"
    assert p.tags == ["smoke"]
    assert p.created_utc == p.updated_utc

    got = store.get("greet")
    assert got.text == "hello world"


def test_save_replaces_and_preserves_created_utc():
    store = PromptStore()
    p1 = store.save("greet", text="hello")
    p2 = store.save("greet", text="hello v2")
    assert p2.created_utc == p1.created_utc
    assert p2.updated_utc >= p1.updated_utc
    assert p2.text == "hello v2"


def test_get_unknown_raises_keyerror():
    store = PromptStore()
    with pytest.raises(KeyError):
        store.get("no-such-prompt")


def test_delete_returns_bool():
    store = PromptStore()
    store.save("doomed", text="bye")
    assert store.delete("doomed") is True
    assert store.delete("doomed") is False
    with pytest.raises(KeyError):
        store.get("doomed")


def test_list_orders_newest_first():
    store = PromptStore()
    store.save("first",  text="a")
    store.save("second", text="b")
    items = store.list()
    assert items[0].name == "second"
    assert items[1].name == "first"


def test_list_filters_by_tag():
    store = PromptStore()
    store.save("a", text="x", tags=["smoke", "safety"])
    store.save("b", text="y", tags=["regression"])
    store.save("c", text="z", tags=["smoke"])
    smoke = [p.name for p in store.list(tag="smoke")]
    assert set(smoke) == {"a", "c"}
    assert [p.name for p in store.list(tag="regression")] == ["b"]
    assert store.list(tag="nonexistent") == []


def test_save_rejects_empty_or_oversized_name():
    store = PromptStore()
    with pytest.raises(ValueError):
        store.save("", text="x")
    with pytest.raises(ValueError):
        store.save("a" * 65, text="x")


def test_save_rejects_bad_name_chars():
    store = PromptStore()
    with pytest.raises(ValueError):
        store.save("with spaces", text="x")
    with pytest.raises(ValueError):
        store.save("slash/path", text="x")


def test_save_rejects_empty_text():
    store = PromptStore()
    with pytest.raises(ValueError):
        store.save("name", text="")
    with pytest.raises(ValueError):
        store.save("name", text="   ")


def test_save_rejects_oversized_text():
    store = PromptStore()
    huge = "a" * 200_001
    with pytest.raises(ValueError):
        store.save("name", text=huge)


def test_tags_normalised_lowercase_and_deduped():
    store = PromptStore()
    p = store.save("name", text="x", tags=["Smoke", "SMOKE", "safety", "  smoke  "])
    assert p.tags == ["smoke", "safety"]


def test_tags_reject_oversized():
    store = PromptStore()
    with pytest.raises(ValueError):
        store.save("name", text="x", tags=["a" * 33])


def test_tags_reject_non_string():
    store = PromptStore()
    with pytest.raises(TypeError):
        store.save("name", text="x", tags=[42])  # type: ignore[list-item]


def test_persistence_across_store_instances(tmp_path: Path):
    db = str(tmp_path / "prompts.db")
    s1 = PromptStore(db)
    s1.save("persisted", text="forever", tags=["t1"])
    s1.close()
    s2 = PromptStore(db)
    got = s2.get("persisted")
    assert got.text == "forever"
    assert got.tags == ["t1"]
    s2.close()


def test_to_dict_is_json_serialisable():
    import json
    store = PromptStore()
    p = store.save("x", text="hello", tags=["smoke"])
    encoded = json.dumps(p.to_dict())
    decoded = json.loads(encoded)
    assert decoded["tags"] == ["smoke"]


def test_len_tracks_writes():
    store = PromptStore()
    assert len(store) == 0
    store.save("a", text="x")
    store.save("b", text="y")
    assert len(store) == 2
    store.delete("a")
    assert len(store) == 1
