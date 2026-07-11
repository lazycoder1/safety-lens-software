"""Focused tests for bounded plate-list matcher caching."""

import plate_store


class _Cursor:
    def __init__(self, rows, calls):
        self._rows = rows
        self._calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query):
        assert query == "SELECT * FROM plate_lists WHERE active = TRUE"
        self._calls.append(query)

    def fetchall(self):
        return self._rows


class _Connection:
    def __init__(self, rows, calls):
        self._rows = rows
        self._calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def cursor(self, **_kwargs):
        return _Cursor(self._rows, self._calls)

    def commit(self):
        return None

    def rollback(self):
        return None


class _MutationCursor:
    def __init__(self, row):
        self._row = row

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, *_args, **_kwargs):
        return None

    def fetchone(self):
        return self._row


class _MutationConnection(_Connection):
    def __init__(self, row):
        self._row = row

    def cursor(self, **_kwargs):
        return _MutationCursor(self._row)


def _plate_row(plate, list_type, *, row_id, created_at):
    return {
        "id": row_id,
        "plate_text": plate,
        "normalized_plate": plate,
        "list_type": list_type,
        "owner_name": "",
        "vehicle_desc": "",
        "valid_from": None,
        "valid_until": None,
        "created_at": created_at,
        "active": True,
    }


def test_exact_and_similar_matching_share_active_list_query(monkeypatch):
    calls = []
    rows = [
        _plate_row(
            "KA05MN4523",
            "whitelist",
            row_id="white-1",
            created_at="2026-01-01T00:00:00+00:00",
        ),
        _plate_row(
            "KA05MN4523",
            "blocked",
            row_id="blocked-1",
            created_at="2025-01-01T00:00:00+00:00",
        ),
    ]
    monkeypatch.setattr(plate_store, "get_conn", lambda: _Connection(rows, calls))
    monkeypatch.setattr(plate_store.time, "monotonic", lambda: 100.0)
    plate_store.invalidate_active_plates_cache()

    exact = plate_store.find_matching_plate("KA05MN4523")
    similar = plate_store.find_similar_plate("KA05MN4528")

    assert exact["id"] == "blocked-1"
    assert similar["id"] == "blocked-1"
    assert len(calls) == 1


def test_active_plate_cache_refreshes_after_ttl_or_invalidation(monkeypatch):
    calls = []
    clock = [100.0]
    rows = [
        _plate_row(
            "KA05MN4523",
            "blocked",
            row_id="blocked-1",
            created_at="2026-01-01T00:00:00+00:00",
        )
    ]
    monkeypatch.setattr(plate_store, "get_conn", lambda: _Connection(rows, calls))
    monkeypatch.setattr(plate_store.time, "monotonic", lambda: clock[0])
    plate_store.invalidate_active_plates_cache()

    plate_store.find_matching_plate("KA05MN4523")
    plate_store.find_matching_plate("KA05MN4523")
    assert len(calls) == 1

    clock[0] += plate_store._ACTIVE_PLATES_CACHE_TTL_SECONDS + 0.01
    plate_store.find_matching_plate("KA05MN4523")
    assert len(calls) == 2

    plate_store.invalidate_active_plates_cache()
    plate_store.find_matching_plate("KA05MN4523")
    assert len(calls) == 3


def test_plate_matchers_filter_cached_validity_windows(monkeypatch):
    calls = []
    expired = _plate_row(
        "KA05MN4523",
        "blocked",
        row_id="expired",
        created_at="2026-01-01T00:00:00+00:00",
    )
    expired["valid_until"] = "2000-01-01T00:00:00+00:00"
    future = _plate_row(
        "KA05MN4523",
        "blocked",
        row_id="future",
        created_at="2026-01-01T00:00:00+00:00",
    )
    future["valid_from"] = "2999-01-01T00:00:00+00:00"
    current = _plate_row(
        "KA05MN4523",
        "whitelist",
        row_id="current",
        created_at="2025-01-01T00:00:00+00:00",
    )
    monkeypatch.setattr(
        plate_store,
        "get_conn",
        lambda: _Connection([expired, future, current], calls),
    )
    plate_store.invalidate_active_plates_cache()

    assert plate_store.find_matching_plate("KA05MN4523")["id"] == "current"
    assert plate_store.find_similar_plate("KA05MN4528")["id"] == "current"
    assert len(calls) == 1


def test_plate_list_mutations_invalidate_match_cache(monkeypatch):
    row = _plate_row(
        "KA05MN4523",
        "blocked",
        row_id="entry-1",
        created_at="2026-01-01T00:00:00+00:00",
    )
    invalidations = []
    monkeypatch.setattr(
        plate_store,
        "get_conn",
        lambda: _MutationConnection(row),
    )
    monkeypatch.setattr(
        plate_store,
        "invalidate_active_plates_cache",
        lambda: invalidations.append(True),
    )

    plate_store.create_plate_entry(
        plate_text="KA05MN4523",
        list_type="blocked",
    )
    plate_store.update_plate_entry(
        "entry-1",
        plate_text="KA05MN4523",
        list_type="blocked",
    )
    plate_store.deactivate_plate_entry("entry-1")

    assert invalidations == [True, True, True]
