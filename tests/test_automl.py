"""
tests/test_automl.py - Comprehensive test suite for the AutoML & Predictive Studio.
"""

import pytest
import numpy as np
import pandas as pd
from core.automl import (
    TaskType,
    infer_task_type,
    build_preprocessor,
    get_candidate_models,
    run_automl_tournament,
    predict_single,
)


@pytest.fixture
def classification_df():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "age": np.random.randint(20, 70, size=n),
        "income": np.random.normal(50000, 15000, size=n),
        "plan": np.random.choice(["Basic", "Pro", "Enterprise"], size=n),
        "churn": np.random.choice([0, 1], p=[0.7, 0.3], size=n),
    })


@pytest.fixture
def regression_df():
    np.random.seed(42)
    n = 100
    x1 = np.random.uniform(10, 100, size=n)
    x2 = np.random.choice(["North", "South", "East", "West"], size=n)
    y = x1 * 2.5 + np.random.normal(0, 5, size=n)
    return pd.DataFrame({"units": x1, "region": x2, "revenue": y})


class TestTaskInference:
    def test_infer_classification_categorical(self, classification_df):
        tt = infer_task_type(classification_df, "plan")
        assert tt == TaskType.CLASSIFICATION

    def test_infer_classification_binary_numeric(self, classification_df):
        tt = infer_task_type(classification_df, "churn")
        assert tt == TaskType.CLASSIFICATION

    def test_infer_regression_continuous(self, regression_df):
        tt = infer_task_type(regression_df, "revenue")
        assert tt == TaskType.REGRESSION

    def test_infer_missing_target_raises(self, classification_df):
        with pytest.raises(ValueError, match="not found in dataset"):
            infer_task_type(classification_df, "nonexistent")


class TestPreprocessing:
    def test_build_preprocessor(self, classification_df):
        prep, num_cols, cat_cols = build_preprocessor(classification_df, "churn")
        assert "age" in num_cols
        assert "income" in num_cols
        assert "plan" in cat_cols


class TestAutomlTournament:
    def test_classification_tournament(self, classification_df):
        result = run_automl_tournament(classification_df, "churn", task_type=TaskType.CLASSIFICATION)
        assert result.task_type == TaskType.CLASSIFICATION
        assert len(result.candidate_models) >= 3
        assert result.best_pipeline is not None
        assert result.best_model_name != ""
        # Check metrics on top candidate
        top_m = result.candidate_models[0]
        assert "Accuracy" in top_m.metrics
        assert "F1 Score (Weighted)" in top_m.metrics
        assert 0.0 <= top_m.metrics["Accuracy"] <= 1.0

    def test_regression_tournament(self, regression_df):
        result = run_automl_tournament(regression_df, "revenue", task_type=TaskType.REGRESSION)
        assert result.task_type == TaskType.REGRESSION
        assert len(result.candidate_models) >= 3
        top_m = result.candidate_models[0]
        assert "R² Score" in top_m.metrics
        assert "RMSE" in top_m.metrics

    def test_single_row_prediction_classification(self, classification_df):
        result = run_automl_tournament(classification_df, "churn")
        pred_res = predict_single(
            pipeline=result.best_pipeline,
            input_values={"age": 35, "income": 55000, "plan": "Pro"},
            feature_columns=result.feature_columns,
            task_type=result.task_type,
            classes=result.classes_,
        )
        assert pred_res.prediction in ["0", "1", 0, 1]

    def test_single_row_prediction_regression(self, regression_df):
        result = run_automl_tournament(regression_df, "revenue")
        pred_res = predict_single(
            pipeline=result.best_pipeline,
            input_values={"units": 50, "region": "North"},
            feature_columns=result.feature_columns,
            task_type=result.task_type,
        )
        assert isinstance(pred_res.prediction, float)
        assert pred_res.prediction > 0

    def test_insufficient_rows_raises(self):
        tiny_df = pd.DataFrame({"x": [1, 2], "y": [10, 20]})
        with pytest.raises(ValueError, match="fewer than 15 rows"):
            run_automl_tournament(tiny_df, "y")
