"""Model lifecycle endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import model_manager
from dependencies import require_admin

router = APIRouter(prefix="/api/models", tags=["models"], dependencies=[Depends(require_admin)])


class ModelInstallRequest(BaseModel):
    model_keys: list[str] = Field(default_factory=list)


@router.get("")
async def api_list_models():
    return {
        "models": await asyncio.to_thread(model_manager.list_models),
    }


@router.post("/install")
async def api_install_models(body: ModelInstallRequest):
    try:
        job = await asyncio.to_thread(model_manager.install_models, body.model_keys)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return job


@router.get("/install/{job_id}")
async def api_get_install_job(job_id: str):
    job = await asyncio.to_thread(model_manager.get_install_job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Install job not found")
    return job


@router.post("/install/{job_id}/retry")
async def api_retry_install_job(job_id: str):
    try:
        job = await asyncio.to_thread(model_manager.retry_install_job, job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return job
