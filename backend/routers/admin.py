"""
SafetyLens admin user management endpoints.
"""

from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import audit_store
import auth_store
import diagnostics
from dependencies import require_admin

router = APIRouter(prefix="/api/admin", tags=["admin"])


class UpdateRoleRequest(BaseModel):
    role: str


class ResetPasswordRequest(BaseModel):
    newPassword: str | None = None


@router.get("/users", dependencies=[Depends(require_admin)])
async def api_get_users():
    return auth_store.get_users()


@router.put("/users/{user_id}/approve", dependencies=[Depends(require_admin)])
async def api_approve_user(user_id: str, request: Request):
    result = auth_store.update_user_status(user_id, "active")
    if not result:
        raise HTTPException(status_code=404, detail="User not found")
    audit_store.log_event(
        "user.approve",
        target_type="user",
        target_id=user_id,
        details={"status": "active"},
        **audit_store.build_actor_context(request),
    )
    return result


@router.put("/users/{user_id}/reject", dependencies=[Depends(require_admin)])
async def api_reject_user(user_id: str, request: Request):
    result = auth_store.update_user_status(user_id, "rejected")
    if not result:
        raise HTTPException(status_code=404, detail="User not found")
    audit_store.log_event(
        "user.reject",
        target_type="user",
        target_id=user_id,
        details={"status": "rejected"},
        **audit_store.build_actor_context(request),
    )
    return result


@router.put("/users/{user_id}/role", dependencies=[Depends(require_admin)])
async def api_update_role(user_id: str, body: UpdateRoleRequest, request: Request):
    if body.role not in ("admin", "operator", "viewer"):
        raise HTTPException(status_code=400, detail="Invalid role")
    result = auth_store.update_user_role(user_id, body.role)
    if not result:
        raise HTTPException(status_code=404, detail="User not found")
    audit_store.log_event(
        "user.role_update",
        target_type="user",
        target_id=user_id,
        details={"role": body.role},
        **audit_store.build_actor_context(request),
    )
    return result


@router.delete("/users/{user_id}", dependencies=[Depends(require_admin)])
async def api_delete_user(user_id: str, request: Request):
    if user_id == request.state.user["sub"]:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    if not auth_store.delete_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    audit_store.log_event(
        "user.delete",
        target_type="user",
        target_id=user_id,
        **audit_store.build_actor_context(request),
    )
    return {"message": "User deleted"}


@router.put("/users/{user_id}/reset-password", dependencies=[Depends(require_admin)])
async def api_reset_password(user_id: str, body: ResetPasswordRequest, request: Request):
    if body.newPassword:
        valid, err = auth_store.validate_password(body.newPassword)
        if not valid:
            raise HTTPException(status_code=400, detail=err)
        new_password = body.newPassword
    else:
        new_password = auth_store.generate_strong_password()
    if not auth_store.reset_password(user_id, new_password):
        raise HTTPException(status_code=404, detail="User not found")
    audit_store.log_event(
        "user.reset_password",
        target_type="user",
        target_id=user_id,
        details={"generated": body.newPassword is None},
        **audit_store.build_actor_context(request),
    )
    return {"message": "Password reset", "newPassword": new_password}


@router.get("/audit-log", dependencies=[Depends(require_admin)])
async def api_get_audit_log(limit: int = 200):
    return audit_store.get_recent(limit=min(max(limit, 1), 1000))


@router.get("/diagnostics/bundle", dependencies=[Depends(require_admin)])
async def api_download_diagnostics_bundle(request: Request):
    bundle = diagnostics.create_diagnostics_bundle()
    audit_store.log_event(
        "diagnostics.bundle_download",
        target_type="diagnostics",
        target_id=bundle.name,
        details={"path": str(bundle)},
        **audit_store.build_actor_context(request),
    )
    return FileResponse(bundle, media_type="application/zip", filename=bundle.name)
