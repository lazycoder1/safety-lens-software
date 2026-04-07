"""
SafetyLens auth endpoints — login, register, me, change-password.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import auth_store

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    currentPassword: str
    newPassword: str


@router.post("/login")
async def api_login(body: LoginRequest):
    user = auth_store.authenticate(body.username, body.password)
    if not user:
        return JSONResponse({"detail": "Invalid credentials or account not active"}, status_code=401)
    token = auth_store.create_token(user["id"], user["username"], user["role"])
    return {"token": token, "user": user}


@router.post("/register")
async def api_register(body: RegisterRequest):
    if len(body.username) < 3:
        return JSONResponse({"detail": "Username must be at least 3 characters"}, status_code=400)
    valid, err = auth_store.validate_password(body.password)
    if not valid:
        return JSONResponse({"detail": err}, status_code=400)
    try:
        auth_store.create_user(body.username, body.password)
    except Exception as e:
        if "duplicate key" in str(e) or "unique" in str(e).lower():
            return JSONResponse({"detail": "Username already taken"}, status_code=409)
        raise
    return {"message": "Account created. Pending admin approval."}


@router.get("/me")
async def api_me(request: Request):
    user = auth_store.get_user(request.state.user["sub"])
    if not user:
        return JSONResponse({"detail": "User not found"}, status_code=404)
    return {"user": user}


@router.post("/change-password")
async def api_change_password(request: Request, body: ChangePasswordRequest):
    valid, err = auth_store.validate_password(body.newPassword)
    if not valid:
        return JSONResponse({"detail": err}, status_code=400)
    user_id = request.state.user["sub"]
    success = auth_store.change_password(user_id, body.currentPassword, body.newPassword)
    if not success:
        return JSONResponse({"detail": "Current password is incorrect"}, status_code=400)
    user = auth_store.get_user(user_id)
    token = auth_store.create_token(user["id"], user["username"], user["role"])
    return {"token": token, "user": user}
