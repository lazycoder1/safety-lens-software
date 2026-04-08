"""
SafetyLens Demo Backend
- Loops videos with YOLO detection, streams annotated frames as MJPEG
- Runs VLM (qwen3-vl) periodically on cameras with demo=yolo+vlm
- Pushes alerts via WebSocket to the React frontend
- Config-driven: all settings from config_manager
"""

import logging

import jwt
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from config_manager import load_config, get_config
from constants import PUBLIC_PATHS, PUBLIC_PREFIXES, FRONTEND_DIR
from logging_config import setup_logging
from routers import register_routers
from routers.safety_rules import _ensure_safety_rules
from video_processing import start_camera
import db
import alert_store
import auth_store
import state

logger = logging.getLogger("safetylens")

# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="SafetyLens Demo Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    # Skip CORS preflight requests
    if request.method == "OPTIONS":
        return await call_next(request)
    if path in PUBLIC_PATHS or any(path.startswith(p) for p in PUBLIC_PREFIXES):
        return await call_next(request)
    # Skip auth for WebSocket upgrade (handled in the WS endpoint)
    if request.headers.get("upgrade", "").lower() == "websocket":
        return await call_next(request)
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
    return await call_next(request)


# ── Register routers ────────────────────────────────────────────────────────

register_routers(app)


# ── Startup ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    setup_logging()
    logger.info("SafetyLens backend starting")
    db.init_pool()
    alert_store.init_db()
    auth_store.init_auth_db()
    state.load_model()
    load_config()
    cfg = get_config()
    _ensure_safety_rules(cfg)
    for cam_id in cfg["cameras"]:
        start_camera(cam_id)


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
