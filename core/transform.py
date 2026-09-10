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
