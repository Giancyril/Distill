"""
core/transform.py - Feature Engineering & Immutable Dataset Time-Travel Manager
Provides mathematical transforms, scaling, binning, datetime decomposition,
interaction features, automated dataset diffing, and an undo/redo time-travel stack.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class TransformType(str, Enum):
    LOG_TRANSFORM = "log_transform"
    SQRT_TRANSFORM = "sqrt_transform"
    STANDARD_SCALE = "standard_scale"
    MINMAX_SCALE = "minmax_scale"
    QUANTILE_BINNING = "quantile_binning"
    DATETIME_DECOMPOSE = "datetime_decompose"
    INTERACTION = "interaction"
    DROP_COLUMN = "drop_column"


@dataclass
class TransformAction:
    action_type: TransformType
    column: str
    target_column: Optional[str] = None  # if creating new column
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


@dataclass
class DatasetSnapshot:
    version_id: int
    name: str
    df: pd.DataFrame
    action_applied: Optional[TransformAction] = None
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    row_count: int = 0
    column_count: int = 0

    def __post_init__(self):
        if self.row_count == 0 and self.df is not None:
            self.row_count = len(self.df)
            self.column_count = len(self.df.columns)



def apply_log_transform(
    df: pd.DataFrame,
    col: str,
    new_col: Optional[str] = None
) -> Tuple[pd.DataFrame, TransformAction]:
    """Applies log1p transformation to reduce right-skewness."""
    df_out = df.copy()
    out_name = new_col or f"{col}_log"
    series = pd.to_numeric(df_out[col], errors="coerce").fillna(0)
    # Clip negative values to 0 for log1p stability
    series_clipped = np.clip(series, 0, None)
    df_out[out_name] = np.log1p(series_clipped)

    action = TransformAction(
        action_type=TransformType.LOG_TRANSFORM,
        column=col,
        target_column=out_name,
        description=f"Applied log1p transformation to '{col}' -> '{out_name}'",
    )
    return df_out, action


def apply_sqrt_transform(
    df: pd.DataFrame,
    col: str,
    new_col: Optional[str] = None
) -> Tuple[pd.DataFrame, TransformAction]:
    """Applies square root transformation to compress moderate right-skew."""
    df_out = df.copy()
    out_name = new_col or f"{col}_sqrt"
    series = pd.to_numeric(df_out[col], errors="coerce").fillna(0)
    series_clipped = np.clip(series, 0, None)
    df_out[out_name] = np.sqrt(series_clipped)

    action = TransformAction(
        action_type=TransformType.SQRT_TRANSFORM,
        column=col,
        target_column=out_name,
        description=f"Applied square root transform to '{col}' -> '{out_name}'",
    )
    return df_out, action


def apply_standard_scale(
    df: pd.DataFrame,
    col: str,
    new_col: Optional[str] = None
) -> Tuple[pd.DataFrame, TransformAction]:
    """Standardizes feature to zero mean and unit variance (Z-Score)."""
    df_out = df.copy()
    out_name = new_col or f"{col}_zscore"
    series = pd.to_numeric(df_out[col], errors="coerce")
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        df_out[out_name] = 0.0
    else:
        df_out[out_name] = (series - mean) / std

    action = TransformAction(
        action_type=TransformType.STANDARD_SCALE,
        column=col,
        target_column=out_name,
        description=f"Standardized '{col}' (mean={mean:.2f}, std={std:.2f}) -> '{out_name}'",
    )
    return df_out, action


def apply_minmax_scale(
    df: pd.DataFrame,
    col: str,
    new_col: Optional[str] = None
) -> Tuple[pd.DataFrame, TransformAction]:
    """Rescales feature strictly into the [0, 1] range."""
    df_out = df.copy()
    out_name = new_col or f"{col}_scaled"
    series = pd.to_numeric(df_out[col], errors="coerce")
    min_v = series.min()
    max_v = series.max()
    denom = max_v - min_v
    if denom == 0 or np.isnan(denom):
        df_out[out_name] = 0.0
    else:
        df_out[out_name] = (series - min_v) / denom

    action = TransformAction(
        action_type=TransformType.MINMAX_SCALE,
        column=col,
        target_column=out_name,
        description=f"Min-Max scaled '{col}' [min={min_v:.2f}, max={max_v:.2f}] -> '{out_name}'",
    )
    return df_out, action


def apply_binning(
    df: pd.DataFrame,
    col: str,
    n_bins: int = 4,
    strategy: str = "quantile",
    new_col: Optional[str] = None
) -> Tuple[pd.DataFrame, TransformAction]:
    """Discretizes continuous metric into binned intervals."""
    df_out = df.copy()
    out_name = new_col or f"{col}_bin"
    series = pd.to_numeric(df_out[col], errors="coerce")

    if strategy == "quantile":
        df_out[out_name] = pd.qcut(series, q=n_bins, duplicates="drop").astype(str)
    else:
        df_out[out_name] = pd.cut(series, bins=n_bins, duplicates="drop").astype(str)

    action = TransformAction(
        action_type=TransformType.QUANTILE_BINNING,
        column=col,
        target_column=out_name,
        parameters={"n_bins": n_bins, "strategy": strategy},
        description=f"Binned '{col}' into {n_bins} intervals ({strategy}) -> '{out_name}'",
    )
    return df_out, action
