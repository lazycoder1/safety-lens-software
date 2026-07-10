"""
Rakshak Lens Demo Backend
- Loops videos with YOLO detection, streams annotated frames as MJPEG
- Runs VLM (qwen3-vl) periodically on cameras with demo=yolo+vlm
- Pushes alerts via WebSocket to the React frontend
- Config-driven: all settings from config_manager
"""

import asyncio
import logging
import time
from uuid import uuid4

import jwt
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from config_manager import load_config, get_config
from constants import PUBLIC_PATHS, PUBLIC_PREFIXES, FRONTEND_DIR
from logging_config import setup_logging
from routers import register_routers
from routers.safety_rules import _ensure_safety_rules
from video_processing import (
    begin_camera_lifecycle_shutdown,
    camera_lifecycle_shutting_down,
    camera_worker_healing_loop,
    resume_camera_lifecycle,
    start_alert_delivery_workers,
    start_camera,
    stop_all_camera_workers,
    stop_alert_pipeline,
)
import db
import alert_store
import face_store
import plate_store
import audit_store
import auth_store
import diagnostics
import error_store
import licensing
import report_generator
import state

logger = logging.getLogger("rakshak_lens")

_camera_startup_task: asyncio.Task | None = None
_camera_healing_task: asyncio.Task | None = None

# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="Rakshak Lens Demo Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _authenticate_request(request: Request):
    path = request.url.path
    if request.method == "OPTIONS":
        return None
    if request.method == "POST" and path == "/api/errors":
        return None
    if path in PUBLIC_PATHS or any(path.startswith(p) for p in PUBLIC_PREFIXES):
        return None
    if request.headers.get("upgrade", "").lower() == "websocket":
        return None
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)
    try:
        payload = auth_store.decode_token(auth[7:])
        request.state.user = payload
    except jwt.ExpiredSignatureError:
        return JSONResponse({"detail": "Token expired"}, status_code=401)
    except Exception:
        return JSONResponse({"detail": "Invalid token"}, status_code=401)
    return None


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = str(uuid4())[:8]
    request.state.request_id = request_id
    start = time.perf_counter()
    response = None

    try:
        auth_response = _authenticate_request(request)
        if auth_response is not None:
            response = auth_response
        else:
            response = await call_next(request)
        return response
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        user = getattr(request.state, "user", {}) or {}
        logger.exception(
            "Unhandled request error",
            extra={
                "request_id": request_id,
                "method": request.method,
                "route": request.url.path,
                "duration_ms": duration_ms,
                "user_id": user.get("sub"),
                "username": user.get("username"),
            },
        )
        try:
            import traceback
            error_store.log_error(
                source="backend",
                message=f"Unhandled error: {exc}",
                stack=traceback.format_exc(),
                url=f"{request.method} {request.url.path}",
                request_id=request_id,
                context={"user_id": user.get("sub"), "username": user.get("username")},
            )
        except Exception:
            pass  # Never let error logging break the request
        raise
    finally:
        if response is not None:
            response.headers["X-Request-ID"] = request_id
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            user = getattr(request.state, "user", {}) or {}
            route = request.url.path
            status_code = response.status_code
            if status_code >= 500:
                log_fn = logger.error
                try:
                    error_store.log_error(
                        source="backend",
                        message=f"HTTP {status_code} on {route}",
                        url=f"{request.method} {route}",
                        request_id=request_id,
                        context={"status_code": status_code, "user_id": user.get("sub")},
                    )
                except Exception:
                    pass
            elif status_code >= 400:
                log_fn = logger.warning
            elif duration_ms > 2000:
                # Slow requests are worth logging
                log_fn = logger.info
            elif request.method in ("POST", "PUT", "DELETE", "PATCH"):
                # State-changing operations are worth logging
                log_fn = logger.info
            else:
                # Routine successful GETs — debug only (file won't capture these)
                log_fn = logger.debug
            log_fn(
                "HTTP request complete",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "route": route,
                    "status_code": status_code,
                    "duration_ms": duration_ms,
                    "user_id": user.get("sub"),
                    "username": user.get("username"),
                },
            )


# ── Register routers ────────────────────────────────────────────────────────

register_routers(app)


# ── Startup ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global _camera_startup_task

    setup_logging()
    logger.info("Rakshak Lens backend starting")
    resume_camera_lifecycle()

    # Phase 1: fast init — server becomes responsive for /api/health and /api/auth/login
    db.init_pool()
    load_config()
    auth_store.init_jwt_secret()
    alert_store.init_db()
    face_store.init_db()
    plate_store.init_db()
    audit_store.init_db()
    auth_store.init_auth_db()
    error_store.init_db()
    await start_alert_delivery_workers()

    # License gate. Inference workers always start, but they self-pause
    # whenever the license state is SUSPENDED. The admin UI stays reachable
    # at all times so the customer can upload a fresh license to recover.
    license_status = licensing.init_licensing()
    if license_status.state == licensing.LicenseState.SUSPENDED:
        logger.warning(
            "License is SUSPENDED at startup — inference will not run until a valid license is installed. Reason: %s",
            license_status.reason,
        )
    elif license_status.state != licensing.LicenseState.VALID:
        logger.warning(
            "License state is %s — %s",
            license_status.state.value,
            license_status.reason,
        )

    # Background task that fetches a fresh heartbeat token from License Hub
    # once a day. Failures are logged but never crash the loop — the existing
    # heartbeat keeps working until the grace period runs out.
    asyncio.create_task(licensing.heartbeat_refresh_loop())
    asyncio.create_task(diagnostics.retention_cleanup_loop())
    asyncio.create_task(alert_store.auto_resolve_loop())
    asyncio.create_task(report_generator.scheduled_report_loop())
    asyncio.create_task(camera_start_retry_loop())
    asyncio.create_task(camera_frame_watchdog_loop())

    # Phase 2: model loading + camera startup in background so the server
    # can serve login and health requests immediately.
    _camera_startup_task = asyncio.create_task(
        _deferred_model_startup(),
        name="camera-model-startup",
    )


@app.on_event("shutdown")
async def shutdown():
    global _camera_startup_task, _camera_healing_task

    # Stop lifecycle repair before any shutdown work. Both tasks run on this
    # event loop. The shared fence also stops an already-offloaded healing pass
    # or model-install callback from recreating workers after cancellation.
    begin_camera_lifecycle_shutdown()
    lifecycle_tasks = [
        task
        for task in (_camera_startup_task, _camera_healing_task)
        if task is not None and not task.done()
    ]
    for task in lifecycle_tasks:
        task.cancel()
    if lifecycle_tasks:
        await asyncio.gather(*lifecycle_tasks, return_exceptions=True)
    _camera_startup_task = None
    _camera_healing_task = None

    cameras_stopped = await asyncio.to_thread(stop_all_camera_workers)
    if not cameras_stopped:
        logger.warning("One or more camera workers exceeded the shutdown timeout")

    drained = await asyncio.to_thread(stop_alert_pipeline, 10.0)
    if not drained:
        logger.warning("Alert pipeline did not drain before shutdown timeout")


async def _deferred_model_startup():
    global _camera_healing_task

    logger.info("Loading models (background)...")
    await asyncio.to_thread(state.load_model)
    logger.info("Models loaded, starting cameras")
    cfg = get_config()
    _ensure_safety_rules(cfg)
    failed_cameras = _start_configured_cameras(cfg)
    logger.info(
        "Camera startup pass complete",
        extra={"failed_camera_count": len(failed_cameras)},
    )
    if not camera_lifecycle_shutting_down():
        _camera_healing_task = asyncio.create_task(
            camera_worker_healing_loop(),
            name="camera-worker-healing",
        )


def _start_configured_cameras(cfg: dict) -> list[str]:
    """Start every configured camera without one failure aborting the pass."""
    failed_cameras: list[str] = []
    for cam_id in cfg.get("cameras", {}):
        if camera_lifecycle_shutting_down():
            break
        try:
            start_camera(cam_id)
        except Exception:
            failed_cameras.append(cam_id)
            logger.exception(
                "Camera failed during deferred startup; continuing with later cameras",
                extra={"camera_id": cam_id},
            )
    return failed_cameras


# ── Serve frontend (production build) ──────────────────────────────────────

if FRONTEND_DIR.is_dir():
    from fastapi.staticfiles import StaticFiles

    # Serve static assets (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """SPA catch-all: serve index.html for any non-API route."""
        file_path = FRONTEND_DIR / full_path
        if full_path and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
