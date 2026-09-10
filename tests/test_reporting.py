"""
tests/test_reporting.py - Comprehensive tests for PDF and HTML reporting engines.
"""

import pytest
import pandas as pd
import numpy as np
import plotly.express as px
from core.reporting import (
    extract_report_data,
    build_pdf_report,
    build_html_dossier,
    ExecutiveReportData,
)


@pytest.fixture
def sample_report_df():
    np.random.seed(42)
    return pd.DataFrame({
        "order_id": range(1001, 1051),
        "category": np.random.choice(["Office", "Furniture", "Tech"], size=50),
        "revenue": np.random.uniform(20.0, 500.0, size=50),
        "discount": np.random.choice([0.0, 0.1, 0.2], size=50),
    })


class TestReportingExtraction:
    def test_extract_report_data(self, sample_report_df):
        rep = extract_report_data(sample_report_df, "sample_orders.csv")
        assert isinstance(rep, ExecutiveReportData)
        assert rep.total_rows == 50
        assert rep.total_columns == 4
        assert rep.quality_score > 90.0
        assert len(rep.metrics) == 4
        assert len(rep.column_summary) == 4
        assert len(rep.key_findings) >= 2

    def test_extract_handles_missing_values(self):
        df_missing = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", None]})
        rep = extract_report_data(df_missing, "messy.csv")
        assert rep.missing_cells == 2
        assert rep.quality_score < 100.0


class TestPDFGeneration:
    def test_build_pdf_returns_valid_pdf_bytes(self, sample_report_df):
        rep = extract_report_data(sample_report_df, "sample_orders.csv")
        pdf_bytes = build_pdf_report(rep)
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 1000
        # Valid PDF header magic bytes
        assert pdf_bytes.startswith(b"%PDF-")


class TestHTMLDossierGeneration:
    def test_build_html_returns_valid_html_document(self, sample_report_df):
        rep = extract_report_data(sample_report_df, "sample_orders.csv")
        fig = px.scatter(sample_report_df, x="revenue", y="discount")
        html_str = build_html_dossier(rep, sample_report_df, plotly_figs=[fig])
        assert isinstance(html_str, str)
        assert "<!DOCTYPE html>" in html_str
        assert "sample_orders.csv" in html_str
        assert "TOTAL OBSERVATIONS" in html_str
        assert "plotly" in html_str.lower()
        assert "</html>" in html_str
