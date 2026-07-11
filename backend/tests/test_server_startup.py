import asyncio
from types import SimpleNamespace

import licensing
import server


class _CompletedTask:
    def done(self):
        return True


def test_startup_schedules_only_defined_background_tasks(monkeypatch):
    scheduled = []

    def capture_task(coroutine, **kwargs):
        scheduled.append((coroutine.cr_code.co_name, kwargs.get("name")))
        coroutine.close()
        return _CompletedTask()

    for module, name in (
        (server.db, "init_pool"),
        (server.auth_store, "init_jwt_secret"),
        (server.alert_store, "init_db"),
        (server.face_store, "init_db"),
        (server.plate_store, "init_db"),
        (server.audit_store, "init_db"),
        (server.auth_store, "init_auth_db"),
        (server.error_store, "init_db"),
    ):
        monkeypatch.setattr(module, name, lambda: None)
    monkeypatch.setattr(server, "setup_logging", lambda: None)
    monkeypatch.setattr(server, "load_config", lambda: {})
    monkeypatch.setattr(
        server.licensing,
        "init_licensing",
        lambda: SimpleNamespace(
            state=licensing.LicenseState.VALID,
            reason="valid",
        ),
    )

    async def start_delivery_workers():
        return None

    monkeypatch.setattr(server, "start_alert_delivery_workers", start_delivery_workers)
    monkeypatch.setattr(server.asyncio, "create_task", capture_task)

    asyncio.run(server.startup())

    assert scheduled == [
        ("heartbeat_refresh_loop", None),
        ("retention_cleanup_loop", None),
        ("auto_resolve_loop", None),
        ("scheduled_report_loop", None),
        ("_deferred_model_startup", "camera-model-startup"),
    ]
