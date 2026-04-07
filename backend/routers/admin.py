"""
SafetyLens admin user management endpoints.
"""

from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import auth_store
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
async def api_approve_user(user_id: str):
    result = auth_store.update_user_status(user_id, "active")
    if not result:
        raise HTTPException(status_code=404, detail="User not found")
    return result


@router.put("/users/{user_id}/reject", dependencies=[Depends(require_admin)])
async def api_reject_user(user_id: str):
    result = auth_store.update_user_status(user_id, "rejected")
    if not result:
        raise HTTPException(status_code=404, detail="User not found")
    return result


@router.put("/users/{user_id}/role", dependencies=[Depends(require_admin)])
async def api_update_role(user_id: str, body: UpdateRoleRequest):
    if body.role not in ("admin", "operator", "viewer"):
        raise HTTPException(status_code=400, detail="Invalid role")
    result = auth_store.update_user_role(user_id, body.role)
    if not result:
        raise HTTPException(status_code=404, detail="User not found")
    return result


@router.delete("/users/{user_id}", dependencies=[Depends(require_admin)])
async def api_delete_user(user_id: str, request: Request):
    if user_id == request.state.user["sub"]:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    if not auth_store.delete_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User deleted"}


@router.put("/users/{user_id}/reset-password", dependencies=[Depends(require_admin)])
async def api_reset_password(user_id: str, body: ResetPasswordRequest):
    if body.newPassword:
        valid, err = auth_store.validate_password(body.newPassword)
        if not valid:
            raise HTTPException(status_code=400, detail=err)
        new_password = body.newPassword
    else:
        new_password = auth_store.generate_strong_password()
    if not auth_store.reset_password(user_id, new_password):
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "Password reset", "newPassword": new_password}
