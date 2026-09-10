"""
core/eda.py — Automated Exploratory Data Analysis
Generates statistical profiles and human-readable summaries
immediately upon dataset upload.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class NumericProfile:
    """Descriptive statistics for a numeric column."""
    column: str
    count: int
    mean: float
    median: float
    std: float
    min: float
    max: float
    q1: float
    q3: float
    iqr: float
    outlier_count: int  # values outside [Q1-1.5*IQR, Q3+1.5*IQR]
    skewness: float


@dataclass
class CategoricalProfile:
    """Frequency distribution for a categorical column."""
    column: str
    cardinality: int
    top_categories: List[Dict[str, Any]]  # [{value, count, pct}]


@dataclass
class DatetimeProfile:
    """Time-series metadata for a datetime column."""
    column: str
    min_date: str
    max_date: str
    date_range_days: int
    detected_frequency: str  # "daily", "monthly", "yearly", "irregular"
    null_count: int


@dataclass
class EDAReport:
    """
    Complete automated EDA profile for a DataFrame.
    Returned immediately on upload; drives the Dashboard tab.
    """
    total_rows: int
    total_columns: int
    numeric_profiles: List[NumericProfile] = field(default_factory=list)
    categorical_profiles: List[CategoricalProfile] = field(default_factory=list)
    datetime_profiles: List[DatetimeProfile] = field(default_factory=list)
    summary_narrative: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _outlier_count(series: pd.Series, q1: float, q3: float) -> int:
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum())


def _safe_float(val) -> float:
    """Convert to float, returning 0.0 for NaN/inf."""
    try:
        f = float(val)
        return 0.0 if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except (TypeError, ValueError):
        return 0.0


def _profile_numeric(series: pd.Series) -> NumericProfile:
    clean = series.dropna()
    q1 = _safe_float(clean.quantile(0.25))
    q3 = _safe_float(clean.quantile(0.75))
    return NumericProfile(
        column=str(series.name),
        count=int(clean.count()),
        mean=_safe_float(clean.mean()),
        median=_safe_float(clean.median()),
        std=_safe_float(clean.std()),
        min=_safe_float(clean.min()),
        max=_safe_float(clean.max()),
        q1=q1,
        q3=q3,
        iqr=round(q3 - q1, 4),
        outlier_count=_outlier_count(clean, q1, q3),
        skewness=_safe_float(clean.skew()),
    )


def _profile_categorical(series: pd.Series, top_n: int = 10) -> CategoricalProfile:
    vc = series.dropna().value_counts()
    total = int(vc.sum())
    top = [
        {"value": str(v), "count": int(c), "pct": round(c / max(total, 1) * 100, 1)}
        for v, c in vc.head(top_n).items()
    ]
    return CategoricalProfile(
        column=str(series.name),
        cardinality=int(series.nunique()),
        top_categories=top,
    )


def _infer_frequency(series: pd.Series) -> str:
    """Heuristically infer time-series cadence from a parsed datetime Series."""
    sorted_s = series.dropna().sort_values()
    if len(sorted_s) < 3:
        return "irregular"
    diffs = sorted_s.diff().dropna().dt.days
    median_diff = diffs.median()
    if median_diff <= 1:
        return "daily"
    if median_diff <= 8:
        return "weekly"
    if median_diff <= 35:
        return "monthly"
    if median_diff <= 100:
        return "quarterly"
    return "yearly"


def _profile_datetime(series: pd.Series) -> DatetimeProfile:
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    null_count = int(parsed.isna().sum())
    clean = parsed.dropna()
    if clean.empty:
        return DatetimeProfile(
            column=str(series.name),
            min_date="N/A",
            max_date="N/A",
            date_range_days=0,
            detected_frequency="irregular",
            null_count=null_count,
        )
    min_date = clean.min()
    max_date = clean.max()
    return DatetimeProfile(
        column=str(series.name),
        min_date=str(min_date.date()),
        max_date=str(max_date.date()),
        date_range_days=int((max_date - min_date).days),
        detected_frequency=_infer_frequency(clean),
        null_count=null_count,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_eda_report(
    df: pd.DataFrame,
    inferred_types: Optional[Dict[str, str]] = None,
) -> EDAReport:
    """
    Run automated EDA on a DataFrame and return a structured EDAReport.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to profile.
    inferred_types : dict, optional
        Pre-computed column type map from core.cleaning.detect_column_types.
        If None, types are inferred naively from pandas dtypes.

    Returns
    -------
    EDAReport
    """
    numeric_profiles: List[NumericProfile] = []
    categorical_profiles: List[CategoricalProfile] = []
    datetime_profiles: List[DatetimeProfile] = []

    for col in df.columns:
        series = df[col]
        col_type = (inferred_types or {}).get(col, "")

        if col_type == "numeric" or (not col_type and pd.api.types.is_numeric_dtype(series)):
            try:
                numeric_profiles.append(_profile_numeric(series))
            except Exception:
                pass

        elif col_type == "categorical" or col_type == "boolean":
            try:
                categorical_profiles.append(_profile_categorical(series))
            except Exception:
                pass

        elif col_type == "datetime":
            try:
                datetime_profiles.append(_profile_datetime(series))
            except Exception:
                pass
        else:
            # For text/id_text: skip heavy profiling, no-op
            pass

    narrative = _build_narrative(df, numeric_profiles, categorical_profiles, datetime_profiles)

    return EDAReport(
        total_rows=len(df),
        total_columns=len(df.columns),
        numeric_profiles=numeric_profiles,
        categorical_profiles=categorical_profiles,
        datetime_profiles=datetime_profiles,
        summary_narrative=narrative,
    )


def _build_narrative(
    df: pd.DataFrame,
    numeric: List[NumericProfile],
    categorical: List[CategoricalProfile],
    datetime_profiles: List[DatetimeProfile],
) -> str:
    """Compose a short human-readable EDA summary."""
    lines: List[str] = []
    lines.append(f"Dataset contains **{len(df):,} rows** and **{len(df.columns)} columns**.")

    if numeric:
        top_col = max(numeric, key=lambda p: p.std)
        lines.append(
            f"The most variable numeric feature is **{top_col.column}** "
            f"(std = {top_col.std:,.2f}, range [{top_col.min:,.2f} – {top_col.max:,.2f}])."
        )
        outlier_cols = [p.column for p in numeric if p.outlier_count > 0]
        if outlier_cols:
            lines.append(
                f"Potential outliers were detected in: **{', '.join(outlier_cols)}**."
            )

    if categorical:
        top_cat = max(categorical, key=lambda p: p.cardinality)
        if top_cat.top_categories:
            top_val = top_cat.top_categories[0]
            lines.append(
                f"The most common **{top_cat.column}** value is "
                f"'{top_val['value']}' ({top_val['pct']}% of records)."
            )

    if datetime_profiles:
        dt = datetime_profiles[0]
        lines.append(
            f"Time range spans **{dt.date_range_days} days** "
            f"({dt.min_date} → {dt.max_date}), detected frequency: **{dt.detected_frequency}**."
        )

    return " ".join(lines)
