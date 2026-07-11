"""
Rakshak Lens API routers.
"""

from fastapi import FastAPI

from routers.auth import router as auth_router
from routers.admin import router as admin_router
from routers.alerts import router as alerts_router
from routers.cameras import router as cameras_router
from routers.zones import router as zones_router
from routers.config import router as config_router
from routers.stream import router as stream_router
from routers.safety_rules import router as safety_rules_router
from routers.automation_rules import router as automation_rules_router
from routers.faces import router as faces_router
from routers.plates import router as plates_router
from routers.license import router as license_router
from routers.models import router as models_router
from routers.misc import router as misc_router
from routers.errors import router as errors_router
from routers.reports import router as reports_router


def register_routers(app: FastAPI):
    app.include_router(auth_router)
    app.include_router(admin_router)
    app.include_router(alerts_router)
    app.include_router(cameras_router)
    app.include_router(zones_router)
    app.include_router(config_router)
    app.include_router(stream_router)
    app.include_router(safety_rules_router)
    app.include_router(automation_rules_router)
    app.include_router(faces_router)
    app.include_router(plates_router)
    app.include_router(license_router)
    app.include_router(models_router)
    app.include_router(misc_router)
    app.include_router(errors_router)
    app.include_router(reports_router)
