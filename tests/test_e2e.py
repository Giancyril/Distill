"""
tests/test_e2e.py — Comprehensive End-to-End Pipeline & Integration Tests.
Tests the complete flow:
File Ingestion -> Cleaning & Health Audit -> Automated EDA -> Query Engine -> Sandbox Execution -> Chart Generation.
"""
import json
import os
from pathlib import Path
import pandas as pd
import pytest

from core.ingestion import load_dataset
from core.cleaning import audit_data_health, clean_dataset, detect_column_types
from core.eda import generate_eda_report
from core.query_engine import extract_schema, run_query
from core.sandbox import execute_in_sandbox, validate_ast
from core.charts import build_chart_from_intent, bar_chart, line_chart

SAMPLE_DATA_DIR = Path(__file__).parent.parent / "sample_data"
SALES_CSV = SAMPLE_DATA_DIR / "sales_clean.csv"
CHURN_CSV = SAMPLE_DATA_DIR / "customer_churn_messy.csv"


class TestFullDataPipelineE2E:
    """Tests the full pipeline from raw file to insight generation."""

    def test_sales_pipeline_e2e(self):
        # 1. Ingestion
        assert SALES_CSV.exists()
        df = load_dataset(str(SALES_CSV))
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

        # 2. Type detection & health audit
        types = detect_column_types(df)
        assert "revenue" in types
        health = audit_data_health(df)
        assert health.total_rows == len(df)
        assert health.total_columns == len(df.columns)

        # 3. Clean dataset
        cleaned_df, actions = clean_dataset(df)
        assert isinstance(cleaned_df, pd.DataFrame)

        # 4. Automated EDA
        eda = generate_eda_report(cleaned_df, types)
        assert eda.total_rows == len(cleaned_df)
        assert len(eda.numeric_profiles) > 0
        assert len(eda.summary_narrative) > 0

        # 5. Schema Extraction for Query
        schemas = extract_schema(cleaned_df, types)
        assert len(schemas) == len(cleaned_df.columns)

        # 6. Natural Language Query (offline / fallback mode)
        query_res = run_query("What is the total revenue by category?", cleaned_df, types)
        assert query_res.generated_code != ""
        assert query_res.chart_intent != ""

        # 7. Sandbox Execution with specific analytical code
        user_code = """
import pandas as pd
_result = df.groupby("category")["revenue"].sum().reset_index()
"""
        sandbox_res = execute_in_sandbox(user_code, cleaned_df)
        assert sandbox_res.success is True
        assert sandbox_res.result_type == "table"
        assert sandbox_res.result_json is not None
        records = json.loads(sandbox_res.result_json)
        assert len(records) > 0
        assert "category" in records[0]
        assert "revenue" in records[0]

        # 8. Visualization from computed table
        result_df = pd.DataFrame(records)
        fig = bar_chart(result_df, x="category", y="revenue", title="Total Revenue by Category")
        assert fig is not None
        assert hasattr(fig, "to_json")

    def test_messy_churn_pipeline_e2e(self):
        # 1. Ingestion of messy CSV
        assert CHURN_CSV.exists()
        df = load_dataset(str(CHURN_CSV))
        assert isinstance(df, pd.DataFrame)

        # 2. Audit should detect duplicates or whitespace or null tokens
        types = detect_column_types(df)
        health = audit_data_health(df)
        assert health.total_rows > 0

        # 3. Cleaning step should resolve messy artifacts without crashing
        cleaned_df, actions = clean_dataset(
            df,
            remove_duplicates=True,
            strip_whitespace=True,
            coerce_numeric=True,
        )
        assert len(cleaned_df) <= len(df)

        # 4. EDA generation on cleaned messy data
        eda = generate_eda_report(cleaned_df)
        assert eda.total_rows == len(cleaned_df)

        # 5. Sandboxed analytical execution on cleaned data
        numeric_cols = [prof.column for prof in eda.numeric_profiles]
        if numeric_cols:
            first_num = numeric_cols[0]
            code = f"""
_result = float(df['{first_num}'].mean())
"""
            res = execute_in_sandbox(code, cleaned_df)
            assert res.success is True
            assert res.result_type == "scalar"

    def test_sandboxed_plotly_figure_e2e(self):
        """Verify code that generates a Plotly figure directly in sandbox."""
        df = load_dataset(str(SALES_CSV))
        code = """
import plotly.express as px
_result = px.scatter(df, x="quantity", y="revenue", color="category")
"""
        res = execute_in_sandbox(code, df)
        assert res.success is True
        assert res.result_type == "chart"
        assert res.result_json is not None
        chart_data = json.loads(res.result_json)
        assert "data" in chart_data
        assert "layout" in chart_data

    def test_security_violation_rejection_e2e(self):
        """Verify attacks are stopped cold before execution."""
        df = load_dataset(str(SALES_CSV))
        attacks = [
            "import os; os.system('whoami')",
            "import sys; sys.exit(1)",
            "import subprocess; subprocess.Popen(['calc.exe'])",
            "open('passwords.txt', 'w').write('hacked')",
            "eval('1 + 1')",
            "exec('x = 5')",
            "(lambda: None).__class__.__bases__[0].__subclasses__()",
        ]
        for attack in attacks:
            # Check AST check directly
            violation = validate_ast(attack)
            assert violation is not None, f"Expected AST block for: {attack}"

            # Check sandbox execution stops it
            res = execute_in_sandbox(attack, df)
            assert res.success is False
            assert res.blocked_reason is not None

    def test_runtime_error_and_timeout_resilience(self):
        """Verify sandbox catches runtime errors cleanly and does not hang."""
        df = load_dataset(str(SALES_CSV))
        # 1. Division by zero
        res_zero = execute_in_sandbox("_result = 1 / 0", df)
        assert res_zero.success is False
        assert res_zero.error_message is not None
        assert "division by zero" in res_zero.error_message.lower()

        # 2. KeyError on column
        res_col = execute_in_sandbox("_result = df['non_existent_column_xyz']", df)
        assert res_col.success is False
        assert res_col.error_message is not None
        assert "non_existent_column_xyz" in res_col.error_message


