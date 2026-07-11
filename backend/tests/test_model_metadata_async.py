import asyncio
import threading

from starlette.requests import Request

import server
from routers import cameras as camera_routes
from routers import models as model_routes


def _request(method: str, path: str) -> Request:
    request = Request(
        {
            "type": "http",
            "method": method,
            "path": path,
            "headers": [],
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
        }
    )
    request.state.user = {"sub": "test", "username": "test", "role": "admin"}
    request.state.request_id = "request-1"
    return request


def test_async_model_routes_run_model_manager_calls_off_event_loop(monkeypatch):
    worker_threads: dict[str, int] = {}

    def recorded(name, result):
        def invoke(*_args):
            worker_threads[name] = threading.get_ident()
            return result

        return invoke

    monkeypatch.setattr(
        model_routes.model_manager,
        "list_models",
        recorded("list", []),
    )
    monkeypatch.setattr(
        model_routes.model_manager,
        "install_models",
        recorded("install", {"id": "job-1"}),
    )
    monkeypatch.setattr(
        model_routes.model_manager,
        "get_install_job",
        recorded("get", {"id": "job-1"}),
    )
    monkeypatch.setattr(
        model_routes.model_manager,
        "retry_install_job",
        recorded("retry", {"id": "job-1"}),
    )

    async def exercise_routes():
        event_loop_thread = threading.get_ident()
        await model_routes.api_list_models()
        await model_routes.api_install_models(
            model_routes.ModelInstallRequest(model_keys=["coco_primary"])
        )
        await model_routes.api_get_install_job("job-1")
        await model_routes.api_retry_install_job("job-1")
        return event_loop_thread

    event_loop_thread = asyncio.run(exercise_routes())

    assert set(worker_threads) == {"list", "install", "get", "retry"}
    assert all(thread_id != event_loop_thread for thread_id in worker_threads.values())


def test_ten_camera_list_fetches_one_snapshot_and_reuses_it(monkeypatch):
    cameras = {
        f"cam-{index}": {
            "name": f"Camera {index}",
            "execution_plan": {"required_model_keys": ["coco_primary"]},
        }
        for index in range(10)
    }
    cfg = {"cameras": cameras}
    readiness = {"coco_primary": True}
    snapshot_calls: list[tuple[list[str], int]] = []
    payload_readiness: list[dict[str, bool]] = []

    monkeypatch.setattr(camera_routes, "get_config", lambda: cfg)
    monkeypatch.setattr(
        camera_routes,
        "normalize_camera_record",
        lambda camera, _cfg: (camera, False),
    )

    def snapshot(model_keys):
        snapshot_calls.append((model_keys, threading.get_ident()))
        return readiness

    def public_payload(camera_id, _camera, _cfg, request_readiness=None):
        payload_readiness.append(request_readiness)
        return {"id": camera_id}

    monkeypatch.setattr(
        camera_routes.model_manager,
        "model_readiness_snapshot",
        snapshot,
    )
    monkeypatch.setattr(camera_routes, "_camera_public_payload", public_payload)

    async def list_cameras():
        event_loop_thread = threading.get_ident()
        result = await camera_routes.get_cameras()
        return event_loop_thread, result

    event_loop_thread, result = asyncio.run(list_cameras())

    assert result == [{"id": f"cam-{index}"} for index in range(10)]
    assert len(snapshot_calls) == 1
    assert snapshot_calls[0][0] == ["coco_primary"]
    assert snapshot_calls[0][1] != event_loop_thread
    assert len(payload_readiness) == 10
    assert all(value is readiness for value in payload_readiness)


def test_async_camera_delete_offloads_worker_stop(monkeypatch):
    cfg = {"cameras": {"cam-1": {"name": "Camera 1"}}}
    stop_threads: list[int] = []

    monkeypatch.setattr(camera_routes, "get_config", lambda: cfg)
    monkeypatch.setattr(camera_routes, "save_config", lambda _cfg: None)
    monkeypatch.setattr(camera_routes.stream_fanout, "retire", lambda _camera_id: None)
    monkeypatch.setattr(camera_routes.audit_store, "log_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(camera_routes.audit_store, "build_actor_context", lambda _request: {})

    def stop_camera(camera_id):
        assert camera_id == "cam-1"
        stop_threads.append(threading.get_ident())
        return True

    monkeypatch.setattr(camera_routes, "stop_camera", stop_camera)

    async def delete_camera():
        event_loop_thread = threading.get_ident()
        response = await camera_routes.api_delete_camera(
            "cam-1",
            _request("DELETE", "/api/cameras/cam-1"),
        )
        return event_loop_thread, response

    event_loop_thread, response = asyncio.run(delete_camera())

    assert response == {"deleted": "cam-1"}
    assert stop_threads and stop_threads[0] != event_loop_thread


def test_deferred_camera_startup_is_offloaded_and_event_loop_stays_responsive(
    monkeypatch,
):
    entered = threading.Event()
    release = threading.Event()
    startup_threads: list[int] = []

    monkeypatch.setattr(server.state, "load_model", lambda: None)
    monkeypatch.setattr(server, "get_config", lambda: {"cameras": {"cam-1": {}}})
    monkeypatch.setattr(server, "_ensure_safety_rules", lambda _cfg: None)
    monkeypatch.setattr(server, "camera_lifecycle_shutting_down", lambda: True)

    def blocking_camera_startup(_cfg):
        startup_threads.append(threading.get_ident())
        entered.set()
        assert release.wait(timeout=2)
        return []

    monkeypatch.setattr(server, "_start_configured_cameras", blocking_camera_startup)

    async def exercise_startup():
        event_loop_thread = threading.get_ident()
        ticks = 0
        startup_task = asyncio.create_task(server._deferred_model_startup())

        async def ticker():
            nonlocal ticks
            while not startup_task.done():
                ticks += 1
                await asyncio.sleep(0.001)

        ticker_task = asyncio.create_task(ticker())
        try:
            for _ in range(200):
                if entered.is_set():
                    break
                await asyncio.sleep(0.001)
            assert entered.is_set()
            ticks_at_entry = ticks
            await asyncio.sleep(0.02)
            assert ticks > ticks_at_entry
        finally:
            release.set()
        await startup_task
        await ticker_task
        return event_loop_thread

    event_loop_thread = asyncio.run(exercise_startup())

    assert startup_threads and startup_threads[0] != event_loop_thread


def test_camera_discovery_and_connection_probe_run_off_event_loop(monkeypatch):
    worker_threads: list[int] = []
    monkeypatch.setattr(camera_routes, "get_config", lambda: {"cameras": {}})

    def discover(_cidrs, *, timeout_seconds):
        assert timeout_seconds == 1.5
        worker_threads.append(threading.get_ident())
        return {"devices": [], "warnings": [], "cidrs": []}

    def resolve(_cidrs):
        worker_threads.append(threading.get_ident())
        return [], []

    def probe(_payload):
        worker_threads.append(threading.get_ident())
        return {"ok": True, "host": "camera.local"}

    monkeypatch.setattr(camera_routes, "discover_cameras", discover)
    monkeypatch.setattr(camera_routes, "resolve_scan_networks", resolve)
    monkeypatch.setattr(camera_routes, "test_camera_connection", probe)
    monkeypatch.setattr(
        camera_routes,
        "_annotate_duplicate_state",
        lambda result, _cfg: result,
    )

    async def exercise():
        event_loop_thread = threading.get_ident()
        await camera_routes.api_discover_cameras(
            camera_routes.DiscoveryScanRequest(cidrs=[], timeout_seconds=1.5)
        )
        await camera_routes.api_test_discovered_camera(
            camera_routes.DiscoveryTestRequest(host="camera.local")
        )
        return event_loop_thread

    event_loop_thread = asyncio.run(exercise())

    assert len(worker_threads) == 3
    assert all(thread_id != event_loop_thread for thread_id in worker_threads)


def test_bulk_import_revalidates_after_each_slow_connection_test(monkeypatch):
    cfg = {"cameras": {}, "global": {"target_fps": 4}}
    events: list[str] = []
    snapshot_threads: list[int] = []
    licensed_camera_counts: list[int] = []
    monkeypatch.setattr(camera_routes, "get_config", lambda: cfg)
    monkeypatch.setattr(camera_routes, "save_config", lambda _cfg: None)
    monkeypatch.setattr(
        camera_routes.licensing,
        "can_add_camera",
        lambda count: licensed_camera_counts.append(count) or (True, ""),
    )
    monkeypatch.setattr(
        camera_routes,
        "_find_duplicate_camera",
        lambda *_args, **_kwargs: ("none", None),
    )
    monkeypatch.setattr(
        camera_routes,
        "_prepare_camera_submission",
        lambda payload, _cfg: {
            **payload,
            "execution_plan": {
                "required_model_keys": ["coco_primary"],
                "zones_required": False,
            },
        },
    )

    def probe(payload):
        events.append(f"probe:{payload['name']}")
        return {
            "ok": True,
            "host": payload["host"],
            "stream_path": "/live",
            "rtsp_url": f"rtsp://{payload['host']}/live",
        }

    def snapshot(model_keys):
        assert model_keys == ["coco_primary"]
        events.append("snapshot")
        snapshot_threads.append(threading.get_ident())
        return {"coco_primary": True}

    monkeypatch.setattr(camera_routes, "test_camera_connection", probe)
    monkeypatch.setattr(
        camera_routes.model_manager,
        "model_readiness_snapshot",
        snapshot,
    )
    monkeypatch.setattr(camera_routes, "start_camera", lambda _camera_id: True)
    monkeypatch.setattr(camera_routes.audit_store, "log_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        camera_routes.audit_store,
        "build_actor_context",
        lambda _request: {},
    )

    request_body = camera_routes.DiscoveryImportRequest(
        devices=[
            camera_routes.DiscoveryImportCamera(
                name="Camera 1",
                zone="Zone A",
                host="camera-1.local",
            ),
            camera_routes.DiscoveryImportCamera(
                name="Camera 2",
                zone="Zone B",
                host="camera-2.local",
            ),
        ]
    )

    async def exercise():
        event_loop_thread = threading.get_ident()
        response = await camera_routes.api_import_discovered_cameras(
            request_body,
            _request("POST", "/api/cameras/discover/import"),
        )
        return event_loop_thread, response

    event_loop_thread, response = asyncio.run(exercise())

    assert events == [
        "probe:Camera 1",
        "snapshot",
        "probe:Camera 2",
        "snapshot",
    ]
    assert len(response["created"]) == 2
    assert licensed_camera_counts == [0, 1]
    assert all(thread_id != event_loop_thread for thread_id in snapshot_threads)
