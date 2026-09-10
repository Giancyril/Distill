"""
Unit tests for core/cleaning.py:
  - detect_column_types
  - audit_data_health
  - clean_dataset
"""
import numpy as np
import pandas as pd
import pytest

from core.cleaning import audit_data_health, clean_dataset, detect_column_types


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_df() -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": ["C-001", "C-002", "C-003", "C-004", "C-005"],
        "age": [25, 30, None, 45, 28],
        "revenue": [100.5, 200.0, 150.0, None, 300.0],
        "category": ["A", "B", "A", "B", "A"],
        "churn": ["True", "False", "No", "Yes", "True"],
        "signup_date": ["2024-01-01", "2024-02-15", "2024-03-20", "2024-04-01", "2024-05-10"],
        "notes": ["  leading space", "ok", "trailing  ", "fine", "   both   "],
    })


@pytest.fixture()
def dup_df() -> pd.DataFrame:
    base = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    return pd.concat([base, base.iloc[:2]], ignore_index=True)  # 2 duplicates


@pytest.fixture()
def null_token_df() -> pd.DataFrame:
    return pd.DataFrame({
        "val": ["n/a", "N/A", "none", "NA", "null", "valid", "NULL", "N.A."],
    })


# ---------------------------------------------------------------------------
# detect_column_types
# ---------------------------------------------------------------------------

class TestDetectColumnTypes:
    def test_numeric_col(self):
        df = pd.DataFrame({"score": [10.5, 20.3, 30.1]})
        types = detect_column_types(df)
        assert types["score"] == "numeric"

    def test_boolean_numeric_col(self):
        df = pd.DataFrame({"flag": [0, 1, 0, 1, 1]})
        types = detect_column_types(df)
        assert types["flag"] == "boolean"

    def test_boolean_string_col(self):
        df = pd.DataFrame({"active": ["True", "False", "true", "false"]})
        types = detect_column_types(df)
        assert types["active"] == "boolean"

    def test_categorical_col(self):
        df = pd.DataFrame({"tier": ["Gold", "Silver", "Bronze", "Gold", "Silver"]})
        types = detect_column_types(df)
        assert types["tier"] == "categorical"

    def test_id_text_by_name(self):
        df = pd.DataFrame({"customer_id": ["C001", "C002", "C003"]})
        types = detect_column_types(df)
        assert types["customer_id"] == "id_text"

    def test_datetime_col_by_name(self):
        df = pd.DataFrame({"signup_date": ["2024-01-01", "2024-02-01", "2024-03-01"]})
        types = detect_column_types(df)
        assert types["signup_date"] == "datetime"

    def test_all_nulls_col(self):
        df = pd.DataFrame({"empty_col": [None, None, None]})
        types = detect_column_types(df)
        assert types["empty_col"] == "text"

    def test_numeric_as_string(self):
        df = pd.DataFrame({"amount": ["100", "200", "300", "400"]})
        types = detect_column_types(df)
        assert types["amount"] == "numeric"


# ---------------------------------------------------------------------------
# audit_data_health
# ---------------------------------------------------------------------------

class TestAuditDataHealth:
    def test_basic_report_shape(self, simple_df):
        report = audit_data_health(simple_df)
        assert report.total_rows == 5
        assert report.total_columns == 7

    def test_missing_cell_count(self, simple_df):
        report = audit_data_health(simple_df)
        assert report.missing_cells >= 2

    def test_missing_pct_within_range(self, simple_df):
        report = audit_data_health(simple_df)
        assert 0 < report.missing_percentage < 100

    def test_duplicate_detection(self, dup_df):
        report = audit_data_health(dup_df)
        assert report.duplicate_rows == 2

    def test_no_duplicates(self, simple_df):
        report = audit_data_health(simple_df)
        assert report.duplicate_rows == 0

    def test_whitespace_warning(self, simple_df):
        report = audit_data_health(simple_df)
        notes_health = report.columns_health["notes"]
        assert any("whitespace" in w.lower() for w in notes_health.warnings)

    def test_high_missingness_flagged(self):
        df = pd.DataFrame({
            "sparse_col": [None] * 8 + ["value"] * 2,
            "ok_col": list(range(10)),
        })
        report = audit_data_health(df)
        assert any("sparse_col" in a for a in report.anomalies)

    def test_column_types_populated(self, simple_df):
        report = audit_data_health(simple_df)
        assert "customer_id" in report.column_types
        assert "age" in report.column_types

    def test_sample_values_count(self, simple_df):
        report = audit_data_health(simple_df)
        for col_health in report.columns_health.values():
            assert len(col_health.sample_values) <= 3

    def test_cleaning_recommendations_nonempty(self, simple_df):
        report = audit_data_health(simple_df)
        assert len(report.cleaning_recommendations) >= 1

    def test_null_tokens_counted(self, null_token_df):
        report = audit_data_health(null_token_df)
        val_health = report.columns_health["val"]
        assert val_health.missing_count >= 6


# ---------------------------------------------------------------------------
# clean_dataset
# ---------------------------------------------------------------------------

class TestCleanDataset:
    def test_deduplication(self, dup_df):
        cleaned, log = clean_dataset(dup_df, remove_duplicates=True)
        assert len(cleaned) == 3
        assert any("duplicate" in msg.lower() for msg in log)

    def test_whitespace_trimming(self, simple_df):
        cleaned, log = clean_dataset(simple_df, strip_whitespace=True)
        assert not cleaned["notes"].str.startswith(" ").any()
        assert any("whitespace" in msg.lower() or "trimmed" in msg.lower() for msg in log)

    def test_null_token_normalization(self, null_token_df):
        cleaned, log = clean_dataset(null_token_df, normalize_nulls=True)
        assert cleaned["val"].notna().sum() == 1
        assert any("normalized" in msg.lower() for msg in log)

    def test_numeric_coercion(self):
        df = pd.DataFrame({"amount": ["100", "200", "300", "400", "500"]})
        cleaned, log = clean_dataset(df, coerce_numeric=True)
        assert pd.api.types.is_numeric_dtype(cleaned["amount"])
        assert any("coerced" in msg.lower() for msg in log)

    def test_no_side_effects_on_original(self, simple_df):
        original_shape = simple_df.shape
        _ = clean_dataset(simple_df)
        assert simple_df.shape == original_shape

    def test_clean_already_clean_df(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        cleaned, log = clean_dataset(df)
        pd.testing.assert_frame_equal(df, cleaned)
        assert log == []

    def test_disable_dedup(self, dup_df):
        cleaned, log = clean_dataset(dup_df, remove_duplicates=False)
        assert len(cleaned) == len(dup_df)
        assert not any("duplicate" in m.lower() for m in log)

    def test_returns_tuple(self, simple_df):
        result = clean_dataset(simple_df)
        assert isinstance(result, tuple)
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], list)

