"""
PDF compliance report generator for Rakshak Lens.
Uses reportlab to produce branded safety reports with KPIs, charts, and violation tables.
"""

import asyncio
import io
import logging
from datetime import datetime, timedelta, timezone

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

import alert_store
from config_manager import get_config

logger = logging.getLogger("rakshak_lens.reports")

PAGE_W, PAGE_H = A4

SEVERITY_COLORS = {
    "P1": colors.HexColor("#dc2626"),
    "P2": colors.HexColor("#ea580c"),
    "P3": colors.HexColor("#ca8a04"),
    "P4": colors.HexColor("#2563eb"),
}

STATUS_COLORS = {
    "active": colors.HexColor("#dc2626"),
    "acknowledged": colors.HexColor("#ea580c"),
    "resolved": colors.HexColor("#16a34a"),
    "snoozed": colors.HexColor("#6b7280"),
}


def generate_compliance_pdf(
    start: str,
    end: str,
    severity: str | None = None,
    camera_id: str | None = None,
    site_name: str = "Rakshak Lens",
) -> bytes:
    """Generate a compliance PDF report and return the bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )
    styles = getSampleStyleSheet()
    story = []

    # Query data
    alerts = _query_alerts(start, end, severity, camera_id)
    stats = _compute_stats(alerts)
    compliance = alert_store.get_compliance_metrics(camera_id=camera_id)

    # --- Cover / Summary ---
    story.append(Paragraph(f"<b>{site_name}</b>", styles["Title"]))
    story.append(Paragraph("Safety Compliance Report", styles["Heading2"]))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(f"<b>Period:</b> {start} to {end}", styles["Normal"]))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]))
    if camera_id:
        story.append(Paragraph(f"<b>Camera:</b> {camera_id}", styles["Normal"]))
    if severity:
        story.append(Paragraph(f"<b>Severity Filter:</b> {severity}", styles["Normal"]))
    story.append(Spacer(1, 10 * mm))

    # KPI table
    safety_pct = compliance.get("safety_compliance_pct")
    ppe_pct = compliance.get("ppe_compliance_pct")
    mtta = compliance.get("mtta_seconds")

    kpi_data = [
        ["Total Violations", "Active", "Acknowledged", "Resolved"],
        [str(stats["total"]), str(stats["active"]), str(stats["acknowledged"]), str(stats["resolved"])],
        ["Safety Compliance", "PPE Compliance", "MTTA", "False Positives"],
        [
            f"{safety_pct:.0f}%" if safety_pct is not None else "N/A",
            f"{ppe_pct:.0f}%" if ppe_pct is not None else "N/A",
            f"{mtta / 60:.1f} min" if mtta is not None else "N/A",
            str(stats["false_positives"]),
        ],
    ]

    kpi_table = Table(kpi_data, colWidths=[40 * mm] * 4)
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
        ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#f3f4f6")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 2), (-1, 2), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("FONTSIZE", (0, 1), (-1, 1), 14),
        ("FONTSIZE", (0, 3), (-1, 3), 14),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 10 * mm))

    # --- Charts ---
    story.append(Paragraph("Violation Analysis", styles["Heading2"]))
    story.append(Spacer(1, 5 * mm))

    # Severity pie chart
    if stats["by_severity"]:
        story.append(Paragraph("Severity Distribution", styles["Heading4"]))
        story.append(_build_severity_pie(stats["by_severity"]))
        story.append(Spacer(1, 8 * mm))

    # By camera bar chart
    if stats["by_camera"]:
        story.append(Paragraph("Violations by Camera", styles["Heading4"]))
        story.append(_build_camera_bar(stats["by_camera"]))
        story.append(Spacer(1, 8 * mm))

    # --- Violation Table ---
    story.append(PageBreak())
    story.append(Paragraph("Violation Details", styles["Heading2"]))
    story.append(Spacer(1, 5 * mm))

    top_alerts = sorted(alerts, key=lambda a: ("P1 P2 P3 P4".index(a.get("severity", "P4")), a.get("timestamp", "")))[:50]

    if top_alerts:
        story.append(_build_violation_table(top_alerts))
    else:
        story.append(Paragraph("No violations found for the selected period.", styles["Normal"]))

    # Build PDF
    doc.build(story, onFirstPage=_add_footer, onLaterPages=_add_footer)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _query_alerts(start: str, end: str, severity: str | None, camera_id: str | None) -> list[dict]:
    """Query alerts within a date range."""
    from db import get_conn
    from psycopg2.extras import RealDictCursor

    clauses = ["timestamp >= %s", "timestamp <= %s"]
    params: list = [start, end]

    if severity:
        clauses.append("severity = %s")
        params.append(severity)
    if camera_id:
        clauses.append("camera_id = %s")
        params.append(camera_id)

    query = f"SELECT * FROM alerts WHERE {' AND '.join(clauses)} ORDER BY timestamp DESC LIMIT 500"

    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

    return [dict(row) for row in rows]


def _compute_stats(alerts: list[dict]) -> dict:
    total = len(alerts)
    active = sum(1 for a in alerts if a.get("status") == "active")
    acknowledged = sum(1 for a in alerts if a.get("status") == "acknowledged")
    resolved = sum(1 for a in alerts if a.get("status") == "resolved")
    false_positives = sum(1 for a in alerts if a.get("false_positive"))

    by_severity: dict[str, int] = {}
    by_rule: dict[str, int] = {}
    by_camera: dict[str, int] = {}

    for a in alerts:
        sev = a.get("severity", "P4")
        by_severity[sev] = by_severity.get(sev, 0) + 1

        rule = a.get("rule", "Unknown")
        by_rule[rule] = by_rule.get(rule, 0) + 1

        cam = a.get("camera_name", "Unknown")
        by_camera[cam] = by_camera.get(cam, 0) + 1

    return {
        "total": total,
        "active": active,
        "acknowledged": acknowledged,
        "resolved": resolved,
        "false_positives": false_positives,
        "by_severity": by_severity,
        "by_rule": by_rule,
        "by_camera": by_camera,
    }


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _build_severity_pie(by_severity: dict[str, int]) -> Drawing:
    d = Drawing(300, 160)
    pie = Pie()
    pie.x = 60
    pie.y = 10
    pie.width = 120
    pie.height = 120

    labels = []
    data = []
    pie_colors = []
    for sev in ["P1", "P2", "P3", "P4"]:
        count = by_severity.get(sev, 0)
        if count > 0:
            labels.append(f"{sev} ({count})")
            data.append(count)
            pie_colors.append(SEVERITY_COLORS.get(sev, colors.gray))

    if not data:
        return d

    pie.data = data
    pie.labels = labels
    for i, c in enumerate(pie_colors):
        pie.slices[i].fillColor = c
        pie.slices[i].strokeColor = colors.white
        pie.slices[i].strokeWidth = 1
    pie.sideLabels = True
    pie.slices.fontName = "Helvetica"
    pie.slices.fontSize = 9

    d.add(pie)
    return d


def _build_camera_bar(by_camera: dict[str, int]) -> Drawing:
    # Take top 8 cameras
    sorted_cams = sorted(by_camera.items(), key=lambda x: -x[1])[:8]
    if not sorted_cams:
        return Drawing(300, 10)

    cam_names = [c[0][:15] for c in sorted_cams]
    cam_values = [c[1] for c in sorted_cams]

    d = Drawing(450, 160)
    chart = VerticalBarChart()
    chart.x = 50
    chart.y = 30
    chart.width = 370
    chart.height = 110
    chart.data = [cam_values]
    chart.categoryAxis.categoryNames = cam_names
    chart.categoryAxis.labels.angle = 30
    chart.categoryAxis.labels.fontSize = 7
    chart.categoryAxis.labels.boxAnchor = "ne"
    chart.valueAxis.valueMin = 0
    chart.valueAxis.labels.fontSize = 8
    chart.bars[0].fillColor = colors.HexColor("#3b82f6")

    d.add(chart)
    return d


# ---------------------------------------------------------------------------
# Violation table
# ---------------------------------------------------------------------------

def _build_violation_table(alerts: list[dict]) -> Table:
    header = ["#", "Timestamp", "Severity", "Camera", "Zone", "Rule", "Conf.", "Status"]
    rows = [header]

    for i, a in enumerate(alerts, 1):
        ts = str(a.get("timestamp", ""))[:19]
        conf = a.get("confidence", 0)
        rows.append([
            str(i),
            ts,
            a.get("severity", ""),
            str(a.get("camera_name", ""))[:18],
            str(a.get("zone", ""))[:15],
            str(a.get("rule", ""))[:20],
            f"{conf:.0%}" if isinstance(conf, (int, float)) else str(conf),
            a.get("status", ""),
        ])

    col_widths = [8 * mm, 35 * mm, 16 * mm, 30 * mm, 25 * mm, 32 * mm, 14 * mm, 20 * mm]
    t = Table(rows, colWidths=col_widths, repeatRows=1)

    style_commands = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (6, 0), (6, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
    ]

    t.setStyle(TableStyle(style_commands))
    return t


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

def _add_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#9ca3af"))
    canvas.drawString(20 * mm, 10 * mm, f"Generated by Rakshak Lens — Page {doc.page}")
    canvas.drawRightString(PAGE_W - 20 * mm, 10 * mm, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Scheduled report loop
# ---------------------------------------------------------------------------

async def scheduled_report_loop() -> None:
    """Background loop that generates and emails reports on schedule."""
    while True:
        await asyncio.sleep(3600)  # Check every hour
        try:
            cfg = get_config()
            sched = cfg.get("scheduled_reports", {})
            if not sched.get("enabled", False):
                continue

            recipients = sched.get("recipients", [])
            if not recipients:
                continue

            now = datetime.now(timezone.utc)
            target_hour = sched.get("hour", 6)
            if now.hour != target_hour:
                continue

            schedule = sched.get("schedule", "weekly")

            if schedule == "daily":
                start = (now - timedelta(days=1)).strftime("%Y-%m-%d")
                end = now.strftime("%Y-%m-%d")
            elif schedule == "weekly":
                day_of_week = sched.get("day_of_week", 1)  # 1=Monday
                if now.isoweekday() != day_of_week:
                    continue
                start = (now - timedelta(days=7)).strftime("%Y-%m-%d")
                end = now.strftime("%Y-%m-%d")
            elif schedule == "monthly":
                if now.day != 1:
                    continue
                start = (now - timedelta(days=30)).strftime("%Y-%m-%d")
                end = now.strftime("%Y-%m-%d")
            else:
                continue

            pdf_bytes = await asyncio.to_thread(generate_compliance_pdf, start, end)
            await asyncio.to_thread(_email_report, pdf_bytes, recipients, start, end)
            logger.info("Scheduled report sent", extra={"schedule": schedule, "recipients": len(recipients)})
        except Exception:
            logger.exception("Scheduled report loop failed")


def _email_report(pdf_bytes: bytes, recipients: list[str], start: str, end: str) -> None:
    """Email a PDF report to recipients using configured SMTP."""
    import smtplib
    from email.mime.application import MIMEApplication
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    cfg = get_config()
    em = cfg.get("email", {})
    smtp_host = em.get("smtp_host", "")
    smtp_port = em.get("smtp_port", 587)
    smtp_user = em.get("smtp_user", "")
    smtp_pass = em.get("smtp_pass", "")
    from_addr = em.get("from_address", "")

    if not smtp_host or not from_addr:
        logger.warning("Cannot send scheduled report: email not configured")
        return

    msg = MIMEMultipart()
    msg["Subject"] = f"[Rakshak Lens] Safety Report {start} to {end}"
    msg["From"] = from_addr
    msg["To"] = ", ".join(recipients)

    body = MIMEText(
        f"<h3>Rakshak Lens Safety Compliance Report</h3>"
        f"<p>Period: {start} to {end}</p>"
        f"<p>Please find the compliance report attached.</p>",
        "html",
    )
    msg.attach(body)

    attachment = MIMEApplication(pdf_bytes, _subtype="pdf")
    attachment.add_header("Content-Disposition", "attachment", filename=f"rakshak-lens-report-{start}-to-{end}.pdf")
    msg.attach(attachment)

    if smtp_port == 465:
        server = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=15)
    else:
        server = smtplib.SMTP(smtp_host, smtp_port, timeout=15)
        server.ehlo()
        server.starttls()
        server.ehlo()

    try:
        if smtp_user and smtp_pass:
            server.login(smtp_user, smtp_pass)
        server.sendmail(from_addr, recipients, msg.as_string())
    finally:
        server.quit()
