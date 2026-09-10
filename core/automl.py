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
