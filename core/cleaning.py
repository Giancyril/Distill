"""
Data Cleaning & Profiling Module.
Audits dataset health, detects column semantics, and executes transparent cleaning.
"""
from __future__ import annotations

import re
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


# Standard representations of missing values in text data
NULL_TOKENS = {
    "nan", "null", "none", "n/a", "na", "n.a.", "-", "--", "?", "missing", "nil", "empty"
}


class ColumnHealth(BaseModel):
    name: str
    inferred_type: str
    missing_count: int
    missing_pct: float
    unique_count: int
    sample_values: List[Any]
    warnings: List[str] = Field(default_factory=list)


class DataHealthReport(BaseModel):
    total_rows: int
    total_columns: int
    duplicate_rows: int
    missing_cells: int
    missing_percentage: float
    column_types: Dict[str, str]
    columns_health: Dict[str, ColumnHealth]
    anomalies: List[str] = Field(default_factory=list)
    cleaning_recommendations: List[str] = Field(default_factory=list)


def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Infers semantic types for all columns:
    - 'numeric': continuous or integer numbers
    - 'datetime': parseable calendar dates or timestamps
    - 'categorical': low-cardinality discrete categories
    - 'boolean': binary flags (True/False, 0/1, Yes/No)
    - 'id_text': unique identifier keys or names
    - 'text': freeform descriptive strings
    """
    inferred = {}
    row_count = len(df)

    for col in df.columns:
        series = df[col].dropna()

        if len(series) == 0:
            inferred[col] = "text"
            continue

        # 1. Native or parseable numeric
        if pd.api.types.is_numeric_dtype(series):
            # Check if it behaves as a boolean (only 0 and 1)
            unique_vals = set(series.unique())
            if unique_vals.issubset({0, 1, 0.0, 1.0}):
                inferred[col] = "boolean"
            else:
                inferred[col] = "numeric"
            continue

        # 2. Boolean check
        str_series = series.astype(str).str.strip().str.lower()
        unique_lower = set(str_series.unique())
        if unique_lower.issubset({"true", "false", "yes", "no", "y", "n", "0", "1"}):
            inferred[col] = "boolean"
            continue

        # 3. Check if text column can be parsed as numbers (e.g. " 45 ", "1068.00")
        try:
            # Exclude strings with pure alphabetical characters
            sample_strs = str_series.head(50)
            cleaned_sample = sample_strs.replace(list(NULL_TOKENS), np.nan).dropna()
            if len(cleaned_sample) > 0:
                pd.to_numeric(cleaned_sample, errors="raise")
                inferred[col] = "numeric"
                continue
        except (ValueError, TypeError):
            pass

        # 4. Datetime detection
        # Check column name or sample format
        col_lower = str(col).lower()
        if any(dt_word in col_lower for dt_word in ["date", "time", "timestamp", "year", "month", "day"]):
            try:
                sample_dates = series.head(50).astype(str)
                pd.to_datetime(sample_dates, errors="raise", format="mixed")
                inferred[col] = "datetime"
                continue
            except Exception:
                pass

        # 5. Identifier detection (e.g., 'CUST-001', 'order_id')
        num_uniques = series.nunique()
        if any(id_word in col_lower for id_word in ["id", "code", "uuid", "key", "sku"]):
            inferred[col] = "id_text"
            continue

        if num_uniques == row_count and num_uniques > 20:
            inferred[col] = "id_text"
            continue

        # 6. Categorical vs General Text
        if num_uniques <= 30 or (row_count > 0 and (num_uniques / row_count) < 0.15):
            inferred[col] = "categorical"
        else:
            inferred[col] = "text"

    return inferred


def audit_data_health(df: pd.DataFrame) -> DataHealthReport:
    """
    Conducts a comprehensive health assessment of the DataFrame:
    reports missingness, duplicate rows, whitespace issues, and anomalies.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    column_types = detect_column_types(df)

    # Duplicate count
    duplicate_rows = int(df.duplicated().sum())

    # Column-level health
    columns_health = {}
    missing_cells = 0
    anomalies = []
    recommendations = []

    if duplicate_rows > 0:
        anomalies.append(f"Found {duplicate_rows} duplicate row(s) ({duplicate_rows / max(total_rows, 1) * 100:.1f}%).")
        recommendations.append(f"Remove {duplicate_rows} duplicate record(s) to prevent skewed aggregations.")

    for col in df.columns:
        series = df[col]
        col_type = column_types[col]
        warnings = []

        # Count NaN and token missingness
        direct_nulls = int(series.isna().sum())
        token_nulls = 0

        if pd.api.types.is_string_dtype(series):
            str_vals = series.dropna().astype(str).str.strip().str.lower()
            token_nulls = int(str_vals.isin(NULL_TOKENS).sum())
            
            # Check for leading/trailing whitespace
            has_whitespace = bool(series.dropna().astype(str).str.contains(r"^\s+|\s+$", regex=True).any())
            if has_whitespace:
                warnings.append("Contains unstripped leading/trailing whitespaces.")
                recommendations.append(f"Trim leading/trailing whitespace in column '{col}'.")

        total_missing_col = direct_nulls + token_nulls
        missing_cells += total_missing_col
        missing_pct = (total_missing_col / max(total_rows, 1)) * 100

        if missing_pct > 0:
            warnings.append(f"{total_missing_col} missing value(s) ({missing_pct:.1f}%).")
            if missing_pct > 40:
                anomalies.append(f"Column '{col}' has high missingness ({missing_pct:.1f}%).")

        unique_count = int(series.nunique(dropna=True))
        sample_vals = [v for v in series.dropna().head(3).tolist()]

        columns_health[col] = ColumnHealth(
            name=col,
            inferred_type=col_type,
            missing_count=total_missing_col,
            missing_pct=round(missing_pct, 2),
            unique_count=unique_count,
            sample_values=sample_vals,
            warnings=warnings
        )

    overall_missing_pct = (missing_cells / max(total_rows * total_cols, 1)) * 100

    return DataHealthReport(
        total_rows=total_rows,
        total_columns=total_cols,
        duplicate_rows=duplicate_rows,
        missing_cells=missing_cells,
        missing_percentage=round(overall_missing_pct, 2),
        column_types=column_types,
        columns_health=columns_health,
        anomalies=anomalies,
        cleaning_recommendations=list(dict.fromkeys(recommendations))
    )


def clean_dataset(
    df: pd.DataFrame,
    remove_duplicates: bool = True,
    strip_whitespace: bool = True,
    normalize_nulls: bool = True,
    coerce_numeric: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Executes transparent data cleaning operations and returns:
    (cleaned_dataframe, audit_log_of_changes)
    """
    cleaned = df.copy()
    changes: List[str] = []

    # 1. Deduplication
    if remove_duplicates:
        dups = int(cleaned.duplicated().sum())
        if dups > 0:
            cleaned = cleaned.drop_duplicates().reset_index(drop=True)
            changes.append(f"Removed {dups} duplicate row(s).")

    # 2. String whitespace trimming
    if strip_whitespace:
        for col in cleaned.select_dtypes(include=["object", "string"]):
            original = cleaned[col]
            trimmed = original.map(lambda x: x.strip() if isinstance(x, str) else x)
            if not original.equals(trimmed):
                cleaned[col] = trimmed
                changes.append(f"Trimmed leading/trailing whitespace in column '{col}'.")

    # 3. Normalize string null tokens to actual np.nan
    if normalize_nulls:
        for col in cleaned.select_dtypes(include=["object", "string"]):
            mask = cleaned[col].astype(str).str.strip().str.lower().isin(NULL_TOKENS)
            replaced_count = int(mask.sum())
            if replaced_count > 0:
                cleaned.loc[mask, col] = np.nan
                changes.append(f"Normalized {replaced_count} placeholder null token(s) to NaN in column '{col}'.")

    # 4. Coerce numbers stored as strings
    if coerce_numeric:
        for col in cleaned.select_dtypes(include=["object", "string"]):
            non_null = cleaned[col].dropna()
            if len(non_null) == 0:
                continue
            # Try numeric conversion
            converted = pd.to_numeric(cleaned[col], errors="coerce")
            # If at least 80% of non-null values convert to valid numbers, coerce
            valid_ratio = converted.notna().sum() / len(non_null)
            if valid_ratio >= 0.80 and not cleaned[col].equals(converted):
                cleaned[col] = converted
                changes.append(f"Coerced column '{col}' from text to numeric values.")

    return cleaned, changes


