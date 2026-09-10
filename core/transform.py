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



def apply_datetime_decompose(
    df: pd.DataFrame,
    col: str,
    components: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, TransformAction]:
    """Extracts temporal components (month, day, dayofweek, is_weekend, quarter) from datetime."""
    df_out = df.copy()
    if components is None:
        components = ["month", "day", "dayofweek", "is_weekend"]

    dt_series = pd.to_datetime(df_out[col], errors="coerce")
    created_cols = []

    if "year" in components:
        c_name = f"{col}_year"
        df_out[c_name] = dt_series.dt.year
        created_cols.append(c_name)

    if "month" in components:
        c_name = f"{col}_month"
        df_out[c_name] = dt_series.dt.month
        created_cols.append(c_name)

    if "day" in components:
        c_name = f"{col}_day"
        df_out[c_name] = dt_series.dt.day
        created_cols.append(c_name)

    if "dayofweek" in components:
        c_name = f"{col}_dayofweek"
        df_out[c_name] = dt_series.dt.dayofweek
        created_cols.append(c_name)

    if "is_weekend" in components:
        c_name = f"{col}_is_weekend"
        df_out[c_name] = (dt_series.dt.dayofweek >= 5).astype(int)
        created_cols.append(c_name)

    action = TransformAction(
        action_type=TransformType.DATETIME_DECOMPOSE,
        column=col,
        parameters={"created_columns": created_cols},
        description=f"Extracted temporal features from '{col}': {', '.join(created_cols)}",
    )
    return df_out, action


def apply_interaction(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    operation: str = "multiply",
    new_col: Optional[str] = None
) -> Tuple[pd.DataFrame, TransformAction]:
    """Generates multiplicative or ratio feature interactions."""
    df_out = df.copy()
    s1 = pd.to_numeric(df_out[col1], errors="coerce").fillna(0)
    s2 = pd.to_numeric(df_out[col2], errors="coerce").fillna(0)

    if operation == "multiply":
        out_name = new_col or f"{col1}_x_{col2}"
        df_out[out_name] = s1 * s2
        desc = f"Created interaction product '{col1}' * '{col2}' -> '{out_name}'"
    elif operation == "ratio":
        out_name = new_col or f"{col1}_per_{col2}"
        df_out[out_name] = s1 / (s2 + 1e-6)
        desc = f"Created ratio '{col1}' / '{col2}' -> '{out_name}'"
    else:
        raise ValueError(f"Unsupported interaction operation: {operation}")

    action = TransformAction(
        action_type=TransformType.INTERACTION,
        column=col1,
        target_column=out_name,
        parameters={"col2": col2, "operation": operation},
        description=desc,
    )
    return df_out, action


def apply_drop_column(df: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, TransformAction]:
    """Safely drops a redundant or leakage column."""
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found.")
    df_out = df.drop(columns=[col])
    action = TransformAction(
        action_type=TransformType.DROP_COLUMN,
        column=col,
        description=f"Dropped column '{col}' from dataset",
    )
    return df_out, action



class DatasetVersionManager:
    """
    Maintains an immutable time-travel history of dataset states.
    Supports linear and branching undo, redo, snapshot rollbacks, and audit logging.
    """
    def __init__(self, initial_df: pd.DataFrame, initial_name: str = "Raw Ingested Dataset"):
        initial_snap = DatasetSnapshot(
            version_id=0,
            name=initial_name,
            df=initial_df.copy(),
            action_applied=None,
        )
        self.history: List[DatasetSnapshot] = [initial_snap]
        self.current_index: int = 0

    @property
    def current_snapshot(self) -> DatasetSnapshot:
        return self.history[self.current_index]

    @property
    def current_df(self) -> pd.DataFrame:
        return self.current_snapshot.df

    @property
    def can_undo(self) -> bool:
        return self.current_index > 0

    @property
    def can_redo(self) -> bool:
        return self.current_index < len(self.history) - 1

    def commit_transform(
        self,
        new_df: pd.DataFrame,
        action: TransformAction,
        name: Optional[str] = None
    ) -> DatasetSnapshot:
        """Applies a transformation and commits a new immutable snapshot to history."""
        # Truncate any forward redo history on new action
        self.history = self.history[:self.current_index + 1]

        new_version_id = len(self.history)
        snap_name = name or action.description or f"Snapshot v{new_version_id}"
        new_snap = DatasetSnapshot(
            version_id=new_version_id,
            name=snap_name,
            df=new_df.copy(),
            action_applied=action,
        )
        self.history.append(new_snap)
        self.current_index = len(self.history) - 1
        return new_snap

    def undo(self) -> Optional[DatasetSnapshot]:
        """Steps back to the previous dataset state."""
        if self.can_undo:
            self.current_index -= 1
            return self.current_snapshot
        return None

    def redo(self) -> Optional[DatasetSnapshot]:
        """Steps forward to the previously undone dataset state."""
        if self.can_redo:
            self.current_index += 1
            return self.current_snapshot
        return None

    def jump_to(self, version_id: int) -> Optional[DatasetSnapshot]:
        """Jumps directly to any snapshot in the version history."""
        if 0 <= version_id < len(self.history):
            self.current_index = version_id
            return self.current_snapshot
        return None

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Returns structured chronological list of all transformations."""
        logs = []
        for snap in self.history:
            logs.append({
                "version": snap.version_id,
                "name": snap.name,
                "timestamp": snap.created_at,
                "rows": snap.row_count,
                "columns": snap.column_count,
                "is_current": snap.version_id == self.current_snapshot.version_id,
                "action": snap.action_applied.description if snap.action_applied else "Baseline Ingestion",
            })
        return logs
