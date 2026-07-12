import asyncio
import threading
from types import SimpleNamespace

import pytest

import diagnostics
import state
import video_processing


_REAL_THREAD = threading.Thread


def test_phone_probe_health_reports_runtime_counts_and_ages(monkeypatch):
    monkeypatch.setattr(
        state,
        "camera_schedule_telemetry",
        {
            "cam-phone": {
                "phoneProbe": {
                    "probeCount": 7,
                    "hitProbeCount": 2,
                    "lastProbeAt": "1970-01-01T00:01:30+00:00",
                    "lastHitAt": "1970-01-01T00:01:20+00:00",
                    "lastProbePhoneDetections": 0,
                    "contextSuppressedCount": 7,
                    "lastContextSuppressedAt": "1970-01-01T00:01:38+00:00",
                }
            }
        },
    )

    health = diagnostics._phone_probe_health("cam-phone", 100.0)

    assert health["probeCount"] == 7
    assert health["hitProbeCount"] == 2
    assert health["lastProbeAgeSeconds"] == 10.0
    assert health["lastHitAgeSeconds"] == 20.0
    assert health["contextSuppressedCount"] == 7
    assert health["lastContextSuppressedAgeSeconds"] == 2.0


@pytest.fixture
def lifecycle_state(monkeypatch):
    monkeypatch.setattr(state, "camera_threads", {})
    monkeypatch.setattr(state, "vlm_threads", {})
    monkeypatch.setattr(state, "camera_runtime_status", {})
    monkeypatch.setattr(state, "camera_frames", {})
    monkeypatch.setattr(state, "camera_clean_frames", {})
    monkeypatch.setattr(state, "camera_detections", {})
    monkeypatch.setattr(state, "camera_connection_health", {})
    monkeypatch.setattr(video_processing, "_camera_lifecycle_locks", {})
    monkeypatch.setattr(
        video_processing,
        "_camera_lifecycle_shutdown",
        threading.Event(),
    )
    monkeypatch.setattr(video_processing, "_last_pose_results", {})
    monkeypatch.setattr(
        video_processing,
        "get_config",
        lambda: {
            "cameras": {
                "cam-1": {
                    "enabled": True,
                    "demo": "yolo",
                    "execution_plan": {"required_model_keys": []},
                }
            }
        },
    )
    monkeypatch.setattr(
        video_processing.model_manager,
        "missing_model_keys",
        lambda _model_keys: [],
    )
    monkeypatch.setattr(
        video_processing,
        "get_config_snapshot",
        lambda *_args, **_kwargs: video_processing.get_config(),
    )


def test_concurrent_camera_starts_publish_one_worker_before_start(
    monkeypatch,
    lifecycle_state,
):
    created = []

    class OwnedThread:
        def __init__(self, *, target, args, daemon):
            self.target = target
            self.args = args
            self.daemon = daemon
            self.alive = False
            created.append(self)

        def start(self):
            assert state.camera_threads["cam-1"][0] is self
            self.alive = True

        def is_alive(self):
            return self.alive

    monkeypatch.setattr(video_processing.threading, "Thread", OwnedThread)
    barrier = threading.Barrier(3)
    results = []

    def start_camera():
        barrier.wait()
        results.append(video_processing.start_camera("cam-1"))

    contenders = [_REAL_THREAD(target=start_camera) for _ in range(2)]
    for contender in contenders:
        contender.start()
    barrier.wait()
    for contender in contenders:
        contender.join(timeout=2)

    assert not any(contender.is_alive() for contender in contenders)
    assert results.count(True) == 1
    assert results.count(False) == 1
    assert len(created) == 1
    assert state.camera_threads["cam-1"][0] is created[0]


def test_camera_and_direct_vlm_start_cannot_create_two_vlm_workers(
    monkeypatch,
    lifecycle_state,
):
    config = video_processing.get_config()
    config["cameras"]["cam-1"]["demo"] = "yolo+vlm"
    monkeypatch.setattr(video_processing, "get_config", lambda: config)
    created = []

    class OwnedThread:
        def __init__(self, *, target, args, daemon):
            self.target = target
            self.args = args
            self.daemon = daemon
            self.alive = False
            created.append(self)

        def start(self):
            registry = (
                state.vlm_threads
                if self.target is video_processing.vlm_worker
                else state.camera_threads
            )
            assert registry["cam-1"][0] is self
            self.alive = True

        def is_alive(self):
            return self.alive

    monkeypatch.setattr(video_processing.threading, "Thread", OwnedThread)
    barrier = threading.Barrier(3)

    def camera_start():
        barrier.wait()
        video_processing.start_camera("cam-1")

    def vlm_start():
        barrier.wait()
        video_processing.start_vlm_for_camera("cam-1")

    contenders = [
        _REAL_THREAD(target=camera_start),
        _REAL_THREAD(target=vlm_start),
    ]
    for contender in contenders:
        contender.start()
    barrier.wait()
    for contender in contenders:
        contender.join(timeout=2)

    assert not any(contender.is_alive() for contender in contenders)
    assert sum(worker.target is video_processing.video_processor for worker in created) == 1
    assert sum(worker.target is video_processing.vlm_worker for worker in created) == 1
    assert state.camera_threads["cam-1"][0].target is video_processing.video_processor
    assert state.vlm_threads["cam-1"][0].target is video_processing.vlm_worker


def test_thread_start_failure_rolls_back_exact_camera_ownership(
    monkeypatch,
    lifecycle_state,
):
    failed = []

    class FailingThread:
        def __init__(self, *, target, args, daemon):
            self.args = args
            failed.append(self)

        def start(self):
            assert state.camera_threads["cam-1"][0] is self
            raise RuntimeError("thread quota exhausted")

        def is_alive(self):
            return False

    monkeypatch.setattr(video_processing.threading, "Thread", FailingThread)

    with pytest.raises(RuntimeError, match="thread quota exhausted"):
        video_processing.start_camera("cam-1")

    assert "cam-1" not in state.camera_threads
    assert failed[0].args[1].is_set()
    assert state.camera_runtime_status["cam-1"] == "error"


def test_vlm_thread_start_failure_keeps_camera_and_automatic_healer_repairs_it(
    monkeypatch,
    lifecycle_state,
):
    config = video_processing.get_config()
    config["cameras"]["cam-1"]["demo"] = "yolo+vlm"
    config["cameras"]["cam-2"] = {
        "enabled": True,
        "demo": "yolo",
        "execution_plan": {"required_model_keys": []},
    }
    monkeypatch.setattr(video_processing, "get_config", lambda: config)
    video_owners = {}
    failed_vlm_event = None

    class StartThread:
        def __init__(self, *, target, args, daemon):
            self.target = target
            self.args = args
            self.alive = False

        def start(self):
            nonlocal failed_vlm_event
            camera_id = self.args[0]
            registry = (
                state.vlm_threads
                if self.target is video_processing.vlm_worker
                else state.camera_threads
            )
            assert registry[camera_id][0] is self
            if (
                camera_id == "cam-1"
                and self.target is video_processing.vlm_worker
                and failed_vlm_event is None
            ):
                failed_vlm_event = self.args[1]
                raise RuntimeError("no VLM thread slot")
            self.alive = True
            if self.target is video_processing.video_processor:
                video_owners[camera_id] = self

        def is_alive(self):
            return self.alive

    monkeypatch.setattr(video_processing.threading, "Thread", StartThread)
    import server

    monkeypatch.setattr(server, "start_camera", video_processing.start_camera)
    monkeypatch.setattr(
        server,
        "camera_lifecycle_shutting_down",
        lambda: False,
    )

    assert server._start_configured_cameras(config) == []

    assert state.camera_threads["cam-1"][0] is video_owners["cam-1"]
    assert state.camera_threads["cam-2"][0] is video_owners["cam-2"]
    assert "cam-1" not in state.vlm_threads
    assert failed_vlm_event.is_set()

    sleep_calls = 0

    async def run_one_healing_interval(_seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls > 1:
            raise asyncio.CancelledError

    monkeypatch.setattr(
        video_processing.asyncio,
        "sleep",
        run_one_healing_interval,
    )

    async def inline_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    # This test replaces threading.Thread with a camera-specific fake; keep
    # asyncio's executor implementation out of that fake's scope.
    monkeypatch.setattr(video_processing.asyncio, "to_thread", inline_to_thread)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(video_processing.camera_worker_healing_loop())

    assert state.camera_threads["cam-1"][0] is video_owners["cam-1"]
    assert state.camera_threads["cam-2"][0] is video_owners["cam-2"]
    assert state.vlm_threads["cam-1"][0].is_alive()


def test_deferred_startup_isolates_one_camera_failure(monkeypatch):
    import server

    starts = []

    def start(cam_id):
        starts.append(cam_id)
        if cam_id == "cam-1":
            raise RuntimeError("camera 1 exhausted its thread quota")

    monkeypatch.setattr(server, "start_camera", start)
    monkeypatch.setattr(
        server,
        "camera_lifecycle_shutting_down",
        lambda: False,
    )

    failed = server._start_configured_cameras(
        {"cameras": {"cam-1": {}, "cam-2": {}}}
    )

    assert starts == ["cam-1", "cam-2"]
    assert failed == ["cam-1"]


def test_old_camera_and_vlm_exits_cannot_remove_new_owners(
    monkeypatch,
    lifecycle_state,
):
    old_camera_thread = object()
    old_camera_stop = threading.Event()
    new_camera_ownership = (object(), threading.Event())
    state.camera_threads["cam-1"] = new_camera_ownership
    state.camera_frames["cam-1"] = b"new-owner-frame"
    state.update_camera_connection_health(
        "cam-1",
        outage_active=True,
        outage_started_monotonic=1,
        outage_failure_count=2,
        total_failure_count=2,
        suppressed_failure_count=1,
        last_transition="outage",
        last_transition_monotonic=1,
    )
    monkeypatch.setattr(
        video_processing.threading,
        "current_thread",
        lambda: old_camera_thread,
    )
    monkeypatch.setattr(video_processing, "_video_processor_loop", lambda *_args: None)

    video_processing.video_processor("cam-1", old_camera_stop)

    assert state.camera_threads["cam-1"] is new_camera_ownership
    assert state.camera_frames["cam-1"] == b"new-owner-frame"
    assert state.get_camera_connection_health("cam-1")["outageActive"] is True

    old_vlm_thread = object()
    old_vlm_stop = threading.Event()
    new_vlm_ownership = (object(), threading.Event())
    state.vlm_threads["cam-1"] = new_vlm_ownership
    monkeypatch.setattr(
        video_processing.threading,
        "current_thread",
        lambda: old_vlm_thread,
    )
    monkeypatch.setattr(video_processing, "_vlm_worker_loop", lambda *_args: None)

    video_processing.vlm_worker("cam-1", old_vlm_stop)

    assert state.vlm_threads["cam-1"] is new_vlm_ownership


def test_exact_camera_owner_exit_clears_stale_connection_outage(
    monkeypatch,
    lifecycle_state,
):
    owner = object()
    stop_event = threading.Event()
    state.camera_threads["cam-1"] = (owner, stop_event)
    state.update_camera_connection_health(
        "cam-1",
        outage_active=True,
        outage_started_monotonic=1,
        outage_failure_count=3,
        total_failure_count=3,
        suppressed_failure_count=2,
        last_transition="outage",
        last_transition_monotonic=1,
    )
    monkeypatch.setattr(video_processing.threading, "current_thread", lambda: owner)

    video_processing._finalize_camera_worker_exit("cam-1", stop_event)

    assert "cam-1" not in state.camera_threads
    assert state.get_camera_connection_health("cam-1")["outageActive"] is False
    assert state.camera_runtime_status["cam-1"] == "error"


def test_health_degrades_for_dead_camera_and_vlm_threads(monkeypatch):
    class ThreadState:
        def __init__(self, alive):
            self.alive = alive

        def is_alive(self):
            return self.alive

    cfg = {
        "retention": {},
        "cameras": {
            "dead-camera": {"enabled": True, "demo": "yolo"},
            "dead-vlm": {"enabled": True, "demo": "yolo+vlm"},
        },
    }
    monkeypatch.setattr(diagnostics, "_health_cache", None)
    monkeypatch.setattr(diagnostics, "get_config", lambda: cfg)
    monkeypatch.setattr(diagnostics.db, "check_connection", lambda: True)
    monkeypatch.setattr(
        diagnostics.licensing,
        "get_status",
        lambda: SimpleNamespace(
            state=diagnostics.licensing.LicenseState.VALID,
            to_public_dict=lambda: {"state": "valid"},
        ),
    )
    monkeypatch.setattr(diagnostics, "_path_usage", lambda _path: {})
    monkeypatch.setattr(
        diagnostics.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=1024),
    )
    monkeypatch.setattr(diagnostics.model_manager, "list_models_for_status", lambda: [])
    monkeypatch.setattr(
        diagnostics.stream_fanout,
        "stats",
        lambda _cam_id: {
            "sequence": 0,
            "subscribers": 0,
            "has_frame": False,
            "frame_age_seconds": None,
        },
    )
    monkeypatch.setattr(
        diagnostics.stream_fanout,
        "operational_stats",
        lambda: {},
    )
    monkeypatch.setattr(
        state,
        "camera_threads",
        {
            "dead-camera": (ThreadState(False), threading.Event()),
            "dead-vlm": (ThreadState(True), threading.Event()),
        },
    )
    monkeypatch.setattr(
        state,
        "vlm_threads",
        {"dead-vlm": (ThreadState(False), threading.Event())},
    )
    monkeypatch.setattr(state, "camera_frames", {})
    monkeypatch.setattr(state, "camera_detections", {})
    monkeypatch.setattr(state, "camera_runtime_status", {})

    snapshot = diagnostics.build_health_snapshot()
    by_id = {camera["id"]: camera for camera in snapshot["cameras"]}

    assert snapshot["status"] == "degraded"
    assert "one or more enabled cameras are not running" in snapshot["reasons"]
    assert (
        "one or more enabled VLM camera companions are not running"
        in snapshot["reasons"]
    )
    assert by_id["dead-camera"]["workerRunning"] is False
    assert by_id["dead-vlm"]["workerRunning"] is True
    assert by_id["dead-vlm"]["vlmExpected"] is True
    assert by_id["dead-vlm"]["vlmWorkerRunning"] is False


def test_start_waits_for_stop_transition_and_retains_stuck_owner(
    monkeypatch,
    lifecycle_state,
):
    join_started = threading.Event()
    release_join = threading.Event()

    class StuckThread:
        def join(self, timeout):
            join_started.set()
            assert release_join.wait(2)

        def is_alive(self):
            return True

    owner = StuckThread()
    stop_event = threading.Event()
    ownership = (owner, stop_event)
    state.camera_threads["cam-1"] = ownership
    stop_results = []
    start_results = []
    start_entered = threading.Event()
    start_done = threading.Event()

    stop_thread = _REAL_THREAD(
        target=lambda: stop_results.append(video_processing.stop_camera("cam-1"))
    )

    def attempt_start():
        start_entered.set()
        start_results.append(video_processing.start_camera("cam-1"))
        start_done.set()

    start_thread = _REAL_THREAD(target=attempt_start)
    stop_thread.start()
    assert join_started.wait(1)
    start_thread.start()
    assert start_entered.wait(1)
    assert not start_done.wait(0.05)
    release_join.set()
    stop_thread.join(timeout=2)
    start_thread.join(timeout=2)

    assert stop_results == [False]
    assert start_results == [False]
    assert state.camera_threads["cam-1"] is ownership
    assert stop_event.is_set()
    assert state.camera_runtime_status["cam-1"] == "stopping"


def test_healer_cannot_start_vlm_from_stale_config_generation(
    monkeypatch,
    lifecycle_state,
):
    class LiveThread:
        def is_alive(self):
            return True

    state.camera_threads["cam-1"] = (LiveThread(), threading.Event())
    yolo_vlm = {
        "cameras": {
            "cam-1": {
                "enabled": True,
                "demo": "yolo+vlm",
                "execution_plan": {"required_model_keys": []},
            }
        }
    }
    yolo_only = {
        "cameras": {
            "cam-1": {
                "enabled": True,
                "demo": "yolo",
                "execution_plan": {"required_model_keys": []},
            }
        }
    }
    snapshots = [yolo_vlm, yolo_vlm, yolo_only]
    snapshot_calls = 0

    def config_snapshot(*_args, **_kwargs):
        nonlocal snapshot_calls
        snapshot = snapshots[min(snapshot_calls, len(snapshots) - 1)]
        snapshot_calls += 1
        return snapshot

    created = []

    class UnexpectedThread:
        def __init__(self, **_kwargs):
            created.append(self)

    monkeypatch.setattr(video_processing, "get_config_snapshot", config_snapshot)
    monkeypatch.setattr(video_processing.threading, "Thread", UnexpectedThread)

    video_processing.heal_camera_workers_once()

    assert snapshot_calls >= 3
    assert created == []
    assert "cam-1" not in state.vlm_threads


def test_healer_stops_vlm_that_current_config_no_longer_expects(
    monkeypatch,
    lifecycle_state,
):
    class LiveThread:
        def is_alive(self):
            return True

    class StoppableThread:
        def __init__(self):
            self.alive = True
            self.join_timeout = None

        def is_alive(self):
            return self.alive

        def join(self, timeout):
            self.join_timeout = timeout
            self.alive = False

    state.camera_threads["cam-1"] = (LiveThread(), threading.Event())
    vlm_thread = StoppableThread()
    vlm_stop = threading.Event()
    state.vlm_threads["cam-1"] = (vlm_thread, vlm_stop)

    video_processing.heal_camera_workers_once()

    assert vlm_stop.is_set()
    assert vlm_thread.join_timeout == 5
    assert "cam-1" not in state.vlm_threads
    assert "cam-1" in state.camera_threads


def test_camera_and_vlm_starts_are_rejected_after_shutdown_fence(
    monkeypatch,
    lifecycle_state,
):
    created = []

    class UnexpectedThread:
        def __init__(self, **_kwargs):
            created.append(self)

    monkeypatch.setattr(video_processing.threading, "Thread", UnexpectedThread)
    video_processing.begin_camera_lifecycle_shutdown()

    assert video_processing.start_camera("cam-1") is False
    assert video_processing.start_vlm_for_camera("cam-1") is False
    assert created == []
    assert state.camera_threads == {}
    assert state.vlm_threads == {}


def test_shutdown_fence_cannot_miss_a_start_waiting_to_publish(
    monkeypatch,
    lifecycle_state,
):
    reached_publication = threading.Event()
    start_results = []
    fence_snapshots = []

    class OwnedThread:
        def __init__(self, *, target, args, daemon):
            self.alive = False

        def start(self):
            self.alive = True

        def is_alive(self):
            return self.alive

    def models_ready(_model_keys):
        reached_publication.set()
        return []

    monkeypatch.setattr(video_processing.threading, "Thread", OwnedThread)
    monkeypatch.setattr(
        video_processing.model_manager,
        "missing_model_keys",
        models_ready,
    )

    publication_lock = video_processing._camera_lifecycle_fence_lock
    publication_lock.acquire()
    starter = _REAL_THREAD(
        target=lambda: start_results.append(video_processing.start_camera("cam-1"))
    )

    def set_fence_and_snapshot():
        video_processing.begin_camera_lifecycle_shutdown()
        fence_snapshots.append(dict(state.camera_threads))

    fencer = _REAL_THREAD(target=set_fence_and_snapshot)
    try:
        starter.start()
        assert reached_publication.wait(1)
        fencer.start()
    finally:
        publication_lock.release()
    starter.join(timeout=2)
    fencer.join(timeout=2)

    assert not starter.is_alive()
    assert not fencer.is_alive()
    assert start_results in ([True], [False])
    assert len(fence_snapshots) == 1
    if start_results == [True]:
        assert "cam-1" in fence_snapshots[0]
    else:
        assert fence_snapshots[0] == {}


def test_offloaded_healer_waiting_on_transition_cannot_resurrect_after_fence(
    monkeypatch,
    lifecycle_state,
):
    discovered = threading.Event()
    config = video_processing.get_config()

    def config_snapshot(*_args, **_kwargs):
        discovered.set()
        return config

    monkeypatch.setattr(video_processing, "get_config_snapshot", config_snapshot)
    lifecycle_lock = video_processing._camera_lifecycle_lock("cam-1")
    lifecycle_lock.acquire()
    healer = _REAL_THREAD(target=video_processing.heal_camera_workers_once)
    try:
        healer.start()
        assert discovered.wait(1)
        video_processing.begin_camera_lifecycle_shutdown()
    finally:
        lifecycle_lock.release()
    healer.join(timeout=2)

    assert not healer.is_alive()
    assert state.camera_threads == {}
    assert state.vlm_threads == {}


def test_healing_work_runs_off_the_asyncio_event_loop(
    monkeypatch,
    lifecycle_state,
):
    original_sleep = asyncio.sleep
    sleep_calls = 0
    worker_threads = []

    async def one_interval_then_cancel(_seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls > 1:
            raise asyncio.CancelledError

    def record_heal_thread():
        worker_threads.append(threading.current_thread())

    monkeypatch.setattr(video_processing.asyncio, "sleep", one_interval_then_cancel)
    monkeypatch.setattr(video_processing, "heal_camera_workers_once", record_heal_thread)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(video_processing.camera_worker_healing_loop())

    # Restore is handled by monkeypatch; retain a reference so this assertion
    # also proves the main event-loop thread was not used for the healing pass.
    assert original_sleep is not None
    assert len(worker_threads) == 1
    assert worker_threads[0] is not threading.main_thread()


def test_server_shutdown_stops_camera_producers_before_alert_pipeline(monkeypatch):
    import server

    calls = []
    monkeypatch.setattr(server, "_camera_startup_task", None)
    monkeypatch.setattr(server, "_camera_healing_task", None)
    monkeypatch.setattr(
        server,
        "begin_camera_lifecycle_shutdown",
        lambda: calls.append("fence"),
    )
    monkeypatch.setattr(
        server,
        "stop_all_camera_workers",
        lambda: calls.append("cameras") or True,
    )
    monkeypatch.setattr(
        server,
        "stop_alert_pipeline",
        lambda _timeout: calls.append("pipeline") or True,
    )

    asyncio.run(server.shutdown())

    assert calls == ["fence", "cameras", "pipeline"]
