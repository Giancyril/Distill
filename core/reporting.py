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



def build_html_dossier(
    report_data: ExecutiveReportData,
    df: pd.DataFrame,
    plotly_figs: Optional[List[Any]] = None
) -> str:
    """
    Generates a standalone, self-contained HTML executive report dossier with
    responsive Distill styling and interactive Plotly visual charts.
    """
    kpi_cards_html = ""
    for m in report_data.metrics:
        kpi_cards_html += f"""
        <div class="kpi-card">
            <div class="kpi-label">{m.label}</div>
            <div class="kpi-value">{m.value}</div>
            <div class="kpi-subtext">{m.subtext}</div>
        </div>
        """

    findings_html = "".join(f"<li>{f}</li>" for f in report_data.key_findings)
    recommendations_html = "".join(f"<li>{r}</li>" for r in report_data.recommendations)

    table_rows = ""
    for r in report_data.column_summary:
        table_rows += f"""
        <tr>
            <td style="font-weight:600;">{r['name']}</td>
            <td><code>{r['dtype']}</code></td>
            <td>{r['non_null']}</td>
            <td>{r['unique']}</td>
            <td>{r['null_pct']}</td>
        </tr>
        """

    charts_html = ""
    if plotly_figs:
        for idx, fig in enumerate(plotly_figs):
            try:
                c_div = fig.to_html(full_html=False, include_plotlyjs="cdn" if idx == 0 else False)
                charts_html += f'<div class="chart-container">{c_div}</div>'
            except Exception:
                pass

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_data.title} - {report_data.dataset_name}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Inter', -apple-system, sans-serif; background: #F8FAFC; color: #0F172A; padding: 40px 20px; line-height: 1.5; }}
        .container {{ max-width: 1100px; margin: 0 auto; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px; padding: 40px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); }}
        .header {{ display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 1px solid #E2E8F0; padding-bottom: 24px; margin-bottom: 32px; }}
        .title {{ font-size: 26px; font-weight: 700; color: #0F172A; letter-spacing: -0.02em; }}
        .subtitle {{ font-size: 13px; color: #64748B; margin-top: 4px; }}
        .badge {{ background: #EEF2FF; color: #4F46E5; border: 1px solid #E0E7FF; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600; }}
        .grid-4 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 32px; }}
        .kpi-card {{ background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px; }}
        .kpi-label {{ font-size: 11px; font-weight: 600; color: #64748B; letter-spacing: 0.05em; text-transform: uppercase; }}
        .kpi-value {{ font-size: 24px; font-weight: 700; color: #4F46E5; margin-top: 4px; }}
        .kpi-subtext {{ font-size: 12px; color: #94A3B8; margin-top: 2px; }}
        h2 {{ font-size: 16px; font-weight: 600; color: #1E293B; margin-bottom: 12px; }}
        .section {{ margin-bottom: 32px; }}
        ul {{ list-style-type: square; margin-left: 20px; color: #334155; font-size: 14px; line-height: 1.8; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }}
        th {{ background: #F1F5F9; color: #475569; font-weight: 600; text-align: left; padding: 10px 12px; border: 1px solid #E2E8F0; }}
        td {{ padding: 10px 12px; border: 1px solid #E2E8F0; color: #334155; }}
        tr:nth-child(even) {{ background: #F8FAFC; }}
        code {{ font-family: 'JetBrains Mono', monospace; font-size: 12px; background: #EEF2FF; color: #4338CA; padding: 2px 6px; border-radius: 4px; }}
        .chart-container {{ margin-top: 20px; border: 1px solid #E2E8F0; border-radius: 12px; padding: 16px; background: #FFFFFF; }}
        @media print {{ body {{ background: #FFF; padding: 0; }} .container {{ border: none; box-shadow: none; padding: 0; }} }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <div class="title">{report_data.title}</div>
                <div class="subtitle">Dataset: <strong>{report_data.dataset_name}</strong> &bull; Generated: {report_data.generated_at}</div>
            </div>
            <span class="badge">Distill Intelligence Dossier</span>
        </div>

        <div class="grid-4">
            {kpi_cards_html}
        </div>

        <div class="section">
            <h2>Executive Observations & Integrity Highlights</h2>
            <ul>{findings_html}</ul>
        </div>

        <div class="section">
            <h2>Dimension & Feature Schema Audit</h2>
            <table>
                <thead>
                    <tr>
                        <th>Feature Column</th>
                        <th>Data Type</th>
                        <th>Non-Null</th>
                        <th>Cardinality</th>
                        <th>Missing %</th>
                    </tr>
                </thead>
                <tbody>{table_rows}</tbody>
            </table>
        </div>

        {charts_html}

        <div class="section" style="margin-top: 32px;">
            <h2>Actionable Next Steps</h2>
            <ul>{recommendations_html}</ul>
        </div>
    </div>
</body>
</html>"""
    return html
