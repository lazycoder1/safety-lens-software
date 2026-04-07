"""
SafetyLens API routers.
"""

from fastapi import FastAPI

from routers.auth import router as auth_router
from routers.admin import router as admin_router
from routers.alerts import router as alerts_router
from routers.cameras import router as cameras_router
from routers.zones import router as zones_router
from routers.config import router as config_router
from routers.stream import router as stream_router
from routers.detection_rules import router as detection_rules_router
from routers.safety_rules import router as safety_rules_router
from routers.misc import router as misc_router


def register_routers(app: FastAPI):
    app.include_router(auth_router)
    app.include_router(admin_router)
    app.include_router(alerts_router)
    app.include_router(cameras_router)
    app.include_router(zones_router)
    app.include_router(config_router)
    app.include_router(stream_router)
    app.include_router(detection_rules_router)
    app.include_router(safety_rules_router)
    app.include_router(misc_router)
