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


from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
)


def extract_feature_importances(
    pipeline: Pipeline,
    feature_names: List[str],
    top_n: int = 15
) -> List[Dict[str, Union[str, float]]]:
    """Extracts ranked feature importances or coefficients from a fitted pipeline."""
    model = pipeline.named_steps.get("model")
    if model is None:
        return []

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = np.abs(coef)

    if importances is None or len(importances) == 0:
        return []

    # Match names if possible
    n = min(len(feature_names), len(importances))
    records = []
    for i in range(n):
        val = float(importances[i])
        records.append({"feature": feature_names[i], "importance": round(val, 4)})

    records.sort(key=lambda x: abs(x["importance"]), reverse=True)
    return records[:top_n]


def run_automl_tournament(
    df: pd.DataFrame,
    target_col: str,
    task_type: Optional[TaskType] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> TournamentResult:
    """
    Executes a tournament among 5 diverse ML algorithms, evaluates them on holdout test set,
    and returns comprehensive rankings and the winning pipeline.
    """
    if task_type is None:
        task_type = infer_task_type(df, target_col)

    clean_df = df.dropna(subset=[target_col]).copy()
    if len(clean_df) < 15:
        raise ValueError("Dataset has fewer than 15 rows with target present; insufficient for ML.")

    preprocessor, num_cols, cat_cols = build_preprocessor(clean_df, target_col)
    feature_cols = num_cols + cat_cols

    X = clean_df[feature_cols]
    y = clean_df[target_col]

    classes_list = None
    if task_type == TaskType.CLASSIFICATION:
        classes_list = [str(c) for c in np.unique(y)]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    candidates = get_candidate_models(task_type)
    evaluations: List[ModelEvaluation] = []
    best_pipeline = None
    best_score = -float("inf")
    best_name = ""

    # Fit preprocessor on X_train to get feature names
    preprocessor.fit(X_train)
    try:
        transformed_feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        transformed_feature_names = feature_cols

    for name, estimator in candidates.items():
        start_t = time.time()
        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", estimator),
        ])

        try:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            fit_time = round(time.time() - start_t, 3)

            metrics: Dict[str, float] = {}
            cm = None
            if task_type == TaskType.CLASSIFICATION:
                acc = float(accuracy_score(y_test, y_pred))
                f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
                prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
                rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
                metrics = {
                    "Accuracy": round(acc, 4),
                    "F1 Score (Weighted)": round(f1, 4),
                    "Precision": round(prec, 4),
                    "Recall": round(rec, 4),
                }
                cm = confusion_matrix(y_test, y_pred).tolist()
                primary_score = f1
            else:
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                mae = float(mean_absolute_error(y_test, y_pred))
                r2 = float(r2_score(y_test, y_pred))
                metrics = {
                    "R² Score": round(r2, 4),
                    "RMSE": round(rmse, 4),
                    "MAE": round(mae, 4),
                }
                primary_score = r2

            feat_imp = extract_feature_importances(pipe, transformed_feature_names)

            evaluation = ModelEvaluation(
                model_name=name,
                task_type=task_type,
                metrics=metrics,
                feature_importances=feat_imp,
                confusion_matrix=cm,
                classes=classes_list,
                fit_time_seconds=fit_time,
            )
            evaluations.append(evaluation)

            if primary_score > best_score:
                best_score = primary_score
                best_pipeline = pipe
                best_name = name

        except Exception as err:
            # Model failed on this configuration, record error
            evaluations.append(ModelEvaluation(
                model_name=name,
                task_type=task_type,
                metrics={"Error": 0.0},
                best_params={"error": str(err)},
            ))

    # Rank evaluations
    if task_type == TaskType.CLASSIFICATION:
        evaluations.sort(key=lambda e: e.metrics.get("F1 Score (Weighted)", -1), reverse=True)
    else:
        evaluations.sort(key=lambda e: e.metrics.get("R² Score", -999), reverse=True)

    summary = (
        f"Evaluated {len(candidates)} models on target '{target_col}'. "
        f"Winner: '{best_name}' with primary benchmark score {round(best_score, 4)}."
    )

    return TournamentResult(
        target_column=target_col,
        task_type=task_type,
        feature_columns=feature_cols,
        candidate_models=evaluations,
        best_model_name=best_name,
        best_pipeline=best_pipeline,
        classes_=classes_list,
        summary=summary,
    )



@dataclass
class SinglePredictionResult:
    prediction: Union[str, float, int]
    probabilities: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None
    task_type: TaskType = TaskType.CLASSIFICATION


def predict_single(
    pipeline: Pipeline,
    input_values: Dict[str, Any],
    feature_columns: List[str],
    task_type: TaskType,
    classes: Optional[List[str]] = None
) -> SinglePredictionResult:
    """
    Runs live inference for a single row given as a feature dict.
    Returns predicted label or numeric estimate, alongside probabilities if classification.
    """
    row_data = {col: [input_values.get(col, np.nan)] for col in feature_columns}
    row_df = pd.DataFrame(row_data)

    pred = pipeline.predict(row_df)[0]

    prob_dict = None
    confidence = None

    if task_type == TaskType.CLASSIFICATION and hasattr(pipeline, "predict_proba"):
        try:
            probs = pipeline.predict_proba(row_df)[0]
            model_classes = classes or [str(i) for i in range(len(probs))]
            prob_dict = {str(c): round(float(p), 4) for c, p in zip(model_classes, probs)}
            confidence = round(float(np.max(probs)), 4)
        except Exception:
            prob_dict = None

    if task_type == TaskType.REGRESSION:
        pred_val = round(float(pred), 4)
    else:
        pred_val = str(pred)

    return SinglePredictionResult(
        prediction=pred_val,
        probabilities=prob_dict,
        confidence=confidence,
        task_type=task_type,
    )
