"""
tests/test_transform.py - Test suite for feature engineering and dataset time-travel.
"""

import pytest
import numpy as np
import pandas as pd
from core.transform import (
    apply_log_transform,
    apply_sqrt_transform,
    apply_standard_scale,
    apply_minmax_scale,
    apply_binning,
    apply_datetime_decompose,
    apply_interaction,
    apply_drop_column,
    DatasetVersionManager,
    compute_dataset_diff,
)


@pytest.fixture
def sample_feature_df():
    np.random.seed(42)
    return pd.DataFrame({
        "sales": [10.0, 50.0, 200.0, 1000.0, 5000.0],
        "quantity": [1, 2, 5, 10, 20],
        "timestamp": pd.date_range("2025-01-01", periods=5, freq="D"),
        "category": ["A", "B", "A", "C", "B"],
    })


class TestMathematicalTransforms:
    def test_log_transform(self, sample_feature_df):
        df_out, act = apply_log_transform(sample_feature_df, "sales")
        assert "sales_log" in df_out.columns
        assert df_out["sales_log"].iloc[0] == pytest.approx(np.log1p(10.0))
        assert act.column == "sales"

    def test_sqrt_transform(self, sample_feature_df):
        df_out, _ = apply_sqrt_transform(sample_feature_df, "quantity")
        assert "quantity_sqrt" in df_out.columns
        assert df_out["quantity_sqrt"].iloc[1] == pytest.approx(np.sqrt(2.0))

    def test_standard_scale(self, sample_feature_df):
        df_out, _ = apply_standard_scale(sample_feature_df, "sales")
        assert "sales_zscore" in df_out.columns
        assert df_out["sales_zscore"].mean() == pytest.approx(0.0, abs=1e-6)
        assert df_out["sales_zscore"].std(ddof=0) == pytest.approx(1.0, abs=1e-6)

    def test_minmax_scale(self, sample_feature_df):
        df_out, _ = apply_minmax_scale(sample_feature_df, "sales")
        assert "sales_scaled" in df_out.columns
        assert df_out["sales_scaled"].min() == pytest.approx(0.0)
        assert df_out["sales_scaled"].max() == pytest.approx(1.0)

    def test_binning(self, sample_feature_df):
        df_out, _ = apply_binning(sample_feature_df, "sales", n_bins=2, strategy="quantile")
        assert "sales_bin" in df_out.columns
        assert df_out["sales_bin"].nunique() <= 2


class TestFeatureEngineering:
    def test_datetime_decompose(self, sample_feature_df):
        df_out, act = apply_datetime_decompose(sample_feature_df, "timestamp")
        assert "timestamp_month" in df_out.columns
        assert "timestamp_day" in df_out.columns
        assert "timestamp_is_weekend" in df_out.columns

    def test_interaction(self, sample_feature_df):
        df_out, _ = apply_interaction(sample_feature_df, "sales", "quantity", operation="multiply")
        assert "sales_x_quantity" in df_out.columns
        assert df_out["sales_x_quantity"].iloc[0] == 10.0 * 1

    def test_drop_column(self, sample_feature_df):
        df_out, _ = apply_drop_column(sample_feature_df, "category")
        assert "category" not in df_out.columns


class TestTimeTravelManager:
    def test_undo_redo_stack(self, sample_feature_df):
        mgr = DatasetVersionManager(sample_feature_df, initial_name="Raw Data")
        assert mgr.current_index == 0
        assert not mgr.can_undo
        assert not mgr.can_redo

        # Apply transformation 1
        df1, act1 = apply_log_transform(mgr.current_df, "sales")
        mgr.commit_transform(df1, act1)
        assert mgr.current_index == 1
        assert mgr.can_undo
        assert not mgr.can_redo
        assert "sales_log" in mgr.current_df.columns

        # Undo
        mgr.undo()
        assert mgr.current_index == 0
        assert "sales_log" not in mgr.current_df.columns
        assert mgr.can_redo

        # Redo
        mgr.redo()
        assert mgr.current_index == 1
        assert "sales_log" in mgr.current_df.columns

    def test_dataset_diff(self, sample_feature_df):
        df_mod = sample_feature_df.copy()
        df_mod["new_feat"] = 99
        df_mod = df_mod.drop(columns=["category"])
        diff = compute_dataset_diff(sample_feature_df, df_mod)
        assert diff.cols_delta == 0
        assert "new_feat" in diff.columns_added
        assert "category" in diff.columns_removed
