"""
core/insights.py - Autonomous AI Insights & Anomaly Narrative Generator
Discovers high-leverage business drivers, data risks, Pareto concentrations,
and synthesizes actionable executive intelligence narratives.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class InsightCategory(str, Enum):
    DRIVER = "Key Driver"
    RISK = "Quality Risk"
    OPPORTUNITY = "Growth Signal"
    ANOMALY = "Anomaly Alert"


class InsightSeverity(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    INFO = "Info"


@dataclass
class ActionableInsight:
    id: str
    headline: str
    narrative: str
    category: InsightCategory
    severity: InsightSeverity
    impact_score: float  # 0.0 to 10.0
    suggested_query: str
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InsightReport:
    total_insights: int
    critical_count: int
    insights: List[ActionableInsight]
    executive_summary: str = ""
