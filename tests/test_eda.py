"""
tests/test_eda.py — Unit tests for the EDA profiling engine.
"""
import pandas as pd
import pytest

from core.eda import (
    CategoricalProfile,
    DatetimeProfile,
    EDAReport,
    NumericProfile,
    generate_eda_report,
)


@pytest.fixture()
def mixed_df() -> pd.DataFrame:
    return pd.DataFrame({
        "order_id": ["ORD-001", "ORD-002", "ORD-003", "ORD-004", "ORD-005"],
        "revenue": [100.0, 250.0, 80.0, 400.0, 175.0],
        "units": [1, 5, 2, 8, 3],
        "category": ["Electronics", "Clothing", "Electronics", "Grocery", "Clothing"],
        "churn": ["Yes", "No", "Yes", "No", "No"],
        "signup_date": ["2024-01-15", "2024-02-20", "2024-03-05", "2024-04-18", "2024-05-22"],
    })


@pytest.fixture()
def inferred_types() -> dict:
    return {
        "order_id": "id_text",
        "revenue": "numeric",
        "units": "numeric",
        "category": "categorical",
        "churn": "boolean",
        "signup_date": "datetime",
    }


class TestGenerateEdaReport:
    def test_returns_eda_report(self, mixed_df, inferred_types):
        report = generate_eda_report(mixed_df, inferred_types)
        assert isinstance(report, EDAReport)

    def test_row_col_counts(self, mixed_df, inferred_types):
        report = generate_eda_report(mixed_df, inferred_types)
        assert report.total_rows == 5
        assert report.total_columns == 6

    def test_numeric_profiles_generated(self, mixed_df, inferred_types):
        report = generate_eda_report(mixed_df, inferred_types)
        assert len(report.numeric_profiles) >= 2
        col_names = [p.column for p in report.numeric_profiles]
        assert "revenue" in col_names
        assert "units" in col_names

    def test_categorical_profiles_generated(self, mixed_df, inferred_types):
        report = generate_eda_report(mixed_df, inferred_types)
        cat_names = [p.column for p in report.categorical_profiles]
        assert "category" in cat_names or "churn" in cat_names

    def test_datetime_profiles_generated(self, mixed_df, inferred_types):
        report = generate_eda_report(mixed_df, inferred_types)
        dt_names = [p.column for p in report.datetime_profiles]
        assert "signup_date" in dt_names

    def test_numeric_profile_values(self, mixed_df, inferred_types):
        report = generate_eda_report(mixed_df, inferred_types)
        rev_profile = next(p for p in report.numeric_profiles if p.column == "revenue")
        assert rev_profile.count == 5
        assert rev_profile.mean == pytest.approx(201.0, rel=0.01)
        assert rev_profile.min == 80.0
        assert rev_profile.max == 400.0

    def test_categorical_cardinality(self, mixed_df, inferred_types):
        report = generate_eda_report(mixed_df, inferred_types)
        cat_profile = next((p for p in report.categorical_profiles if p.column == "category"), None)
        if cat_profile:
            assert cat_profile.cardinality == 3  # Electronics, Clothing, Grocery

    def test_datetime_range(self, mixed_df, inferred_types):
        report = generate_eda_report(mixed_df, inferred_types)
        dt_profile = next(p for p in report.datetime_profiles if p.column == "signup_date")
        assert dt_profile.min_date == "2024-01-15"
        assert dt_profile.max_date == "2024-05-22"
        assert dt_profile.date_range_days == 128

    def test_narrative_nonempty(self, mixed_df, inferred_types):
        report = generate_eda_report(mixed_df, inferred_types)
        assert len(report.summary_narrative) > 20

    def test_narrative_mentions_rows(self, mixed_df, inferred_types):
        report = generate_eda_report(mixed_df, inferred_types)
        assert "5" in report.summary_narrative or "rows" in report.summary_narrative

    def test_handles_all_numeric_df(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        types = {"a": "numeric", "b": "numeric"}
        report = generate_eda_report(df, types)
        assert len(report.numeric_profiles) == 2

    def test_handles_empty_inferred_types(self, mixed_df):
        # Should fall back to pandas dtype detection
        report = generate_eda_report(mixed_df)
        assert isinstance(report, EDAReport)

    def test_outlier_count_type(self, mixed_df, inferred_types):
        report = generate_eda_report(mixed_df, inferred_types)
        for p in report.numeric_profiles:
            assert isinstance(p.outlier_count, int)
            assert p.outlier_count >= 0
