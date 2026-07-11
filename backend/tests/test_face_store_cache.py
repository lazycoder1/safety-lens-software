"""Focused tests for bounded face-enrollment cache behavior."""

import face_store


class _FakeCursor:
    def __init__(self, rows, query_calls):
        self._rows = rows
        self._query_calls = query_calls

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query):
        assert query == "SELECT * FROM enrolled_faces WHERE active = TRUE"
        self._query_calls.append(query)

    def fetchall(self):
        return self._rows


class _FakeConnection:
    def __init__(self, rows, query_calls):
        self._rows = rows
        self._query_calls = query_calls

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def cursor(self, **_kwargs):
        return _FakeCursor(self._rows, self._query_calls)


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


class _MutationConnection:
    def __init__(self, row):
        self._row = row

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def cursor(self, **_kwargs):
        return _MutationCursor(self._row)

    def commit(self):
        return None


def _enrollment_row(*, active=True):
    return {
        "id": "face-1",
        "name": "Worker",
        "face_group": "employees",
        "valid_until": None,
        "enrolled_at": "2026-07-11T00:00:00+00:00",
        "consent_method": "Written form",
        "consent_confirmed": True,
        "photo_path": None,
        "embedding": [0.1, 0.2],
        "active": active,
    }


def test_active_faces_cache_avoids_repeated_queries_until_ttl(monkeypatch):
    query_calls = []
    clock = [100.0]
    rows = [{"id": "face-1", "embedding": [0.1, 0.2], "active": True}]
    monkeypatch.setattr(
        face_store,
        "get_conn",
        lambda: _FakeConnection(rows, query_calls),
    )
    monkeypatch.setattr(face_store.time, "monotonic", lambda: clock[0])
    face_store.invalidate_active_faces_cache()

    first = face_store._active_faces_with_embeddings()
    second = face_store._active_faces_with_embeddings()

    assert first == second == rows
    assert len(query_calls) == 1

    clock[0] += face_store._ACTIVE_FACES_CACHE_TTL_SECONDS + 0.01
    assert face_store._active_faces_with_embeddings() == rows
    assert len(query_calls) == 2


def test_active_faces_cache_can_be_invalidated_immediately(monkeypatch):
    query_calls = []
    rows = [{"id": "face-1", "embedding": [0.1, 0.2], "active": True}]
    monkeypatch.setattr(
        face_store,
        "get_conn",
        lambda: _FakeConnection(rows, query_calls),
    )
    monkeypatch.setattr(face_store.time, "monotonic", lambda: 100.0)
    face_store.invalidate_active_faces_cache()

    face_store._active_faces_with_embeddings()
    face_store.invalidate_active_faces_cache()
    face_store._active_faces_with_embeddings()

    assert len(query_calls) == 2


def test_create_enrollment_invalidates_active_faces_cache(monkeypatch):
    invalidations = []
    monkeypatch.setattr(face_store, "_vector_available", False)
    monkeypatch.setattr(
        face_store,
        "get_conn",
        lambda: _MutationConnection(_enrollment_row()),
    )
    monkeypatch.setattr(
        face_store,
        "invalidate_active_faces_cache",
        lambda: invalidations.append(True),
    )

    created = face_store.create_enrollment(
        name="Worker",
        group="employees",
        valid_until=None,
        consent_method="Written form",
        embedding=[0.1, 0.2],
    )

    assert created["id"] == "face-1"
    assert invalidations == [True]


def test_deactivate_face_invalidates_active_faces_cache(monkeypatch):
    invalidations = []
    monkeypatch.setattr(
        face_store,
        "get_conn",
        lambda: _MutationConnection(_enrollment_row(active=False)),
    )
    monkeypatch.setattr(
        face_store,
        "invalidate_active_faces_cache",
        lambda: invalidations.append(True),
    )

    deactivated = face_store.deactivate_face("face-1")

    assert deactivated["active"] is False
    assert invalidations == [True]
