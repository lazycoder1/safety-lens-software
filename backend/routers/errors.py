"""
Rakshak Lens error log endpoints — frontend error reporting + admin query.
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

import error_store
from dependencies import require_operator_or_admin

router = APIRouter(prefix="/api", tags=["errors"])


class ErrorReport(BaseModel):
    message: str
    stack: Optional[str] = None
    url: Optional[str] = None
    component: Optional[str] = None
    context: Optional[dict] = None
    request_id: Optional[str] = None


@router.post("/errors")
async def report_error(body: ErrorReport):
    """Accept error reports from the frontend. No auth required so errors
    during login or expired sessions can still be recorded."""
    ctx = body.context or {}
    if body.component:
        ctx["component"] = body.component
    result = error_store.log_error(
        source="frontend",
        message=body.message,
        stack=body.stack,
        url=body.url,
        request_id=body.request_id,
        context=ctx,
    )
    return result


@router.get("/errors", dependencies=[Depends(require_operator_or_admin)])
async def get_errors(
    source: Optional[str] = Query(None),
    since: Optional[str] = Query(None),
    limit: int = Query(100, le=500),
):
    """Query error log. Admin/operator only."""
    return error_store.get_errors(source=source, since=since, limit=limit)
