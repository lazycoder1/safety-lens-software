"""Focused tests for duplicate alert-evidence hardlink reuse."""

import os

import alert_store


def _reset_cache():
    with alert_store._snapshot_link_cache_lock:
        alert_store._snapshot_link_cache.clear()


def test_identical_snapshots_keep_unique_names_and_share_inode(tmp_path):
    _reset_cache()
    first = tmp_path / "alert-1.jpg"
    second = tmp_path / "alert-2.jpg"
    content = b"same-frame-evidence" * 100

    assert alert_store._link_or_write_snapshot_once(first, content) is True
    assert alert_store._link_or_write_snapshot_once(second, content) is True

    assert first.read_bytes() == second.read_bytes() == content
    assert first.name != second.name
    assert first.stat().st_ino == second.stat().st_ino
    assert first.stat().st_nlink == 2

    first.unlink()
    assert second.read_bytes() == content
    assert second.stat().st_nlink == 1


def test_different_snapshot_content_is_not_linked(tmp_path):
    _reset_cache()
    first = tmp_path / "alert-1.jpg"
    second = tmp_path / "alert-2.jpg"

    alert_store._link_or_write_snapshot_once(first, b"frame-one")
    alert_store._link_or_write_snapshot_once(second, b"frame-two")

    assert first.stat().st_ino != second.stat().st_ino


def test_missing_cached_source_falls_back_to_safe_write(tmp_path):
    _reset_cache()
    first = tmp_path / "alert-1.jpg"
    second = tmp_path / "alert-2.jpg"
    content = b"same-frame-evidence"
    alert_store._link_or_write_snapshot_once(first, content)
    first.unlink()

    assert alert_store._link_or_write_snapshot_once(second, content) is True
    assert second.read_bytes() == content


def test_snapshot_link_cache_is_bounded(tmp_path, monkeypatch):
    _reset_cache()
    monkeypatch.setattr(alert_store, "_SNAPSHOT_LINK_CACHE_MAX_ENTRIES", 2)

    for index in range(3):
        alert_store._link_or_write_snapshot_once(
            tmp_path / f"alert-{index}.jpg",
            f"frame-{index}".encode(),
        )

    assert len(alert_store._snapshot_link_cache) == 2


def test_existing_alert_evidence_is_never_overwritten(tmp_path):
    _reset_cache()
    target = tmp_path / "alert-1.jpg"
    target.write_bytes(b"durable-original")

    assert alert_store._link_or_write_snapshot_once(target, b"replacement") is False
    assert target.read_bytes() == b"durable-original"
    assert os.stat(target).st_nlink == 1
