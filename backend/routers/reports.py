"""
Rakshak Lens report endpoints — PDF compliance report download.
"""

import asyncio
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import Response

import report_generator

router = APIRouter(prefix="/api", tags=["reports"])


@router.get("/reports/pdf")
async def download_pdf_report(
    start: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end: str = Query(..., description="End date (YYYY-MM-DD)"),
    severity: Optional[str] = Query(None),
    cameraId: Optional[str] = Query(None),
):
    """Generate and download a compliance PDF report."""
    pdf_bytes = await asyncio.to_thread(
        report_generator.generate_compliance_pdf, start, end, severity, cameraId
    )
    filename = f"rakshak-lens-report-{start}-to-{end}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
