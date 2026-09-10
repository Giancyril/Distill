"""
core/automl.py - Automated Machine Learning & Predictive Modeling Studio
Provides task inference, feature preprocessing, model tournaments,
metric evaluations, and interactive inference.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


@dataclass
class ModelEvaluation:
    """Evaluation metrics and metadata for a trained model candidate."""
    model_name: str
    task_type: TaskType
    metrics: Dict[str, float]
    feature_importances: List[Dict[str, Union[str, float]]] = field(default_factory=list)
    confusion_matrix: Optional[List[List[int]]] = None
    classes: Optional[List[str]] = None
    best_params: Dict[str, Any] = field(default_factory=dict)
    fit_time_seconds: float = 0.0


@dataclass
class TournamentResult:
    """Results from an AutoML multi-model tournament."""
    target_column: str
    task_type: TaskType
    feature_columns: List[str]
    candidate_models: List[ModelEvaluation]
    best_model_name: str
    best_pipeline: Any
    classes_: Optional[List[str]] = None
    summary: str = ""


def infer_task_type(df: pd.DataFrame, target_col: str) -> TaskType:
    """
    Infers whether the prediction target is a classification or regression task.
    - If target dtype is object, category, boolean, or string -> classification
    - If target is numeric with very low cardinality (<= 10 unique values and <= 5% unique ratio) -> classification
    - Otherwise -> regression
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    series = df[target_col].dropna()
    if len(series) == 0:
        raise ValueError(f"Target column '{target_col}' contains only null values.")

    dtype_str = str(series.dtype).lower()
    unique_count = series.nunique()
    total_count = len(series)

    if any(k in dtype_str for k in ["object", "category", "bool", "string"]):
        return TaskType.CLASSIFICATION

    # Numeric case
    if unique_count <= 2:
        return TaskType.CLASSIFICATION
    if unique_count <= 10 and (unique_count / total_count < 0.15):
        return TaskType.CLASSIFICATION

    return TaskType.REGRESSION


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_preprocessor(
    df: pd.DataFrame,
    target_col: str,
    max_categories: int = 25
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Constructs an automated ColumnTransformer that handles numeric imputation/scaling
    and categorical imputation/one-hot encoding.
    Returns (preprocessor, numeric_features, categorical_features).
    """
    feature_df = df.drop(columns=[target_col])
    numeric_features: List[str] = []
    categorical_features: List[str] = []

    for col in feature_df.columns:
        # Skip pure id / high-cardinality string columns or timestamp-like
        series = feature_df[col]
        dtype_str = str(series.dtype).lower()
        if any(num_type in dtype_str for num_type in ["int", "float", "double"]):
            numeric_features.append(col)
        else:
            # Categorical if cardinality is reasonable
            n_unique = series.nunique()
            if 1 < n_unique <= max_categories:
                categorical_features.append(col)

    transformers = []
    if numeric_features:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, numeric_features))

    if categorical_features:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipeline, categorical_features))

    if not transformers:
        raise ValueError("No viable numeric or categorical feature columns identified for modeling.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, numeric_features, categorical_features


import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split


def get_candidate_models(task_type: TaskType) -> Dict[str, Any]:
    """Returns candidate estimators tailored for the task type."""
    if task_type == TaskType.CLASSIFICATION:
        return {
            "Random Forest": RandomForestClassifier(n_estimators=75, max_depth=10, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=60, max_depth=4, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        }
    else:
        return {
            "Random Forest": RandomForestRegressor(n_estimators=75, max_depth=10, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=60, max_depth=4, random_state=42),
            "Ridge Regression": Ridge(alpha=1.0, random_state=42),
            "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
            "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
        }
