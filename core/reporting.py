"""
core/reporting.py - Executive Intelligence Report Generator
Generates publication-ready vector PDF documents (via ReportLab) and
interactive standalone HTML dossiers with Distill enterprise styling.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import io
import pandas as pd
import numpy as np

from core.cleaning import DataHealthReport
from core.eda import EDAReport


@dataclass
class ExecutiveMetric:
    label: str
    value: str
    subtext: str = ""


@dataclass
class ExecutiveReportData:
    title: str
    dataset_name: str
    generated_at: str
    total_rows: int
    total_columns: int
    quality_score: float
    missing_cells: int
    duplicate_rows: int
    metrics: List[ExecutiveMetric] = field(default_factory=list)
    column_summary: List[Dict[str, Any]] = field(default_factory=list)
    key_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


def extract_report_data(
    df: pd.DataFrame,
    filename: str,
    health: Optional[DataHealthReport] = None,
    eda: Optional[EDAReport] = None,
) -> ExecutiveReportData:
    """Extracts executive intelligence metrics from active dataset and diagnostics."""
    n_rows, n_cols = df.shape
    missing = int(df.isna().sum().sum())
    dups = int(df.duplicated().sum())

    score = 100.0
    if n_rows * n_cols > 0:
        missing_pct = (missing / (n_rows * n_cols)) * 100.0
        dup_pct = (dups / n_rows) * 100.0
        score = max(0.0, round(100.0 - (missing_pct * 2.0) - (dup_pct * 1.5), 1))

    metrics = [
        ExecutiveMetric(label="TOTAL OBSERVATIONS", value=f"{n_rows:,}", subtext="Audited records"),
        ExecutiveMetric(label="DIMENSIONS", value=f"{n_cols}", subtext="Feature columns"),
        ExecutiveMetric(label="INTEGRITY SCORE", value=f"{score}%", subtext="Data hygiene index"),
        ExecutiveMetric(label="DUPLICATE ROWS", value=f"{dups:,}", subtext="Redundant records"),
    ]

    col_summary = []
    for col in df.columns[:15]:  # Top 15 for executive summary
        s = df[col]
        col_summary.append({
            "name": str(col),
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "unique": int(s.nunique()),
            "null_pct": f"{round((s.isna().sum() / n_rows) * 100, 1)}%",
        })

    findings = []
    if missing > 0:
        findings.append(f"Identified {missing:,} null/missing data cells across {n_cols} columns.")
    else:
        findings.append("Zero missing cells detected. Dataset has 100% attribute completeness.")

    if dups > 0:
        findings.append(f"Found {dups:,} duplicate rows requiring deduplication.")
    else:
        findings.append("No exact duplicate records detected across the observation space.")

    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        findings.append(f"Contains {len(num_cols)} continuous quantitative features suitable for predictive modeling.")

    recommendations = [
        "Enforce automated schema validation on ingestion to maintain data integrity.",
        "Address outlier distributions in critical numeric metrics prior to production model training.",
        "Establish automated data health monitoring pipelines with anomaly alerts.",
    ]

    return ExecutiveReportData(
        title="Distill Executive Intelligence Dossier",
        dataset_name=filename or "Active Dataset",
        generated_at=datetime.now().strftime("%B %d, %Y - %H:%M UTC"),
        total_rows=n_rows,
        total_columns=n_cols,
        quality_score=score,
        missing_cells=missing,
        duplicate_rows=dups,
        metrics=metrics,
        column_summary=col_summary,
        key_findings=findings,
        recommendations=recommendations,
    )


from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def _get_distill_styles() -> Dict[str, ParagraphStyle]:
    """Defines ReportLab typography styles matching the Distill design system."""
    base_styles = getSampleStyleSheet()
    styles = {}

    styles["Title"] = ParagraphStyle(
        "DistillTitle",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4,
    )
    styles["Subheader"] = ParagraphStyle(
        "DistillSubheader",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=12,
    )
    styles["Heading2"] = ParagraphStyle(
        "DistillH2",
        parent=base_styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=12,
        spaceAfter=6,
    )
    styles["Body"] = ParagraphStyle(
        "DistillBody",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
        spaceAfter=4,
    )
    styles["Bullet"] = ParagraphStyle(
        "DistillBullet",
        parent=styles["Body"],
        leftIndent=12,
        bulletIndent=4,
        spaceAfter=4,
    )
    styles["TableHead"] = ParagraphStyle(
        "DistillTableHead",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#475569"),
    )
    styles["TableCell"] = ParagraphStyle(
        "DistillTableCell",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1E293B"),
    )
    return styles



def build_pdf_report(report_data: ExecutiveReportData) -> bytes:
    """
    Renders an executive-ready vector PDF document using ReportLab.
    Returns the binary content of the PDF.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = _get_distill_styles()
    story = []

    # 1. Header
    story.append(Paragraph(report_data.title, styles["Title"]))
    story.append(Paragraph(
        f"Dataset: <b>{report_data.dataset_name}</b> &nbsp;|&nbsp; Generated: {report_data.generated_at} &nbsp;|&nbsp; Distill Data Lab Enterprise",
        styles["Subheader"]
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E2E8F0"), spaceAfter=12))

    # 2. KPI Scorecards Table
    kpi_data = [
        [Paragraph(f"<b>{m.label}</b>", styles["TableHead"]) for m in report_data.metrics],
        [Paragraph(f"<font size=14 color='#4F46E5'><b>{m.value}</b></font>", styles["TableCell"]) for m in report_data.metrics],
        [Paragraph(f"<font color='#64748B'>{m.subtext}</font>", styles["TableCell"]) for m in report_data.metrics],
    ]
    kpi_table = Table(kpi_data, colWidths=[135, 135, 135, 135])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 14))

    # 3. Key Findings
    story.append(Paragraph("Executive Observations & Integrity Audit", styles["Heading2"]))
    for finding in report_data.key_findings:
        story.append(Paragraph(f"&bull; &nbsp; {finding}", styles["Bullet"]))
    story.append(Spacer(1, 10))

    # 4. Attribute Schema Table
    story.append(Paragraph("Dimension & Quality Summary", styles["Heading2"]))
    table_rows = [
        [
            Paragraph("<b>Column Name</b>", styles["TableHead"]),
            Paragraph("<b>Data Type</b>", styles["TableHead"]),
            Paragraph("<b>Non-Null Count</b>", styles["TableHead"]),
            Paragraph("<b>Unique Values</b>", styles["TableHead"]),
            Paragraph("<b>Null Rate</b>", styles["TableHead"]),
        ]
    ]
    for row in report_data.column_summary:
        table_rows.append([
            Paragraph(f"<b>{row['name']}</b>", styles["TableCell"]),
            Paragraph(row["dtype"], styles["TableCell"]),
            Paragraph(str(row["non_null"]), styles["TableCell"]),
            Paragraph(str(row["unique"]), styles["TableCell"]),
            Paragraph(row["null_pct"], styles["TableCell"]),
        ])

    col_table = Table(table_rows, colWidths=[150, 90, 100, 100, 100])
    col_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF2FF")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#3730A3")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(col_table)
    story.append(Spacer(1, 12))

    # 5. Strategic Recommendations
    story.append(Paragraph("Recommended Action Items", styles["Heading2"]))
    for rec in report_data.recommendations:
        story.append(Paragraph(f"&check; &nbsp; {rec}", styles["Bullet"]))

    doc.build(story)
    return buf.getvalue()
