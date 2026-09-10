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



def find_pareto_drivers(
    df: pd.DataFrame,
    category_col: str,
    metric_col: str,
    threshold: float = 0.80
) -> Optional[ActionableInsight]:
    """
    Checks if a Pareto principle (80/20 rule) exists where a small fraction of categories
    accounts for the overwhelming share of a key metric.
    """
    if category_col not in df.columns or metric_col not in df.columns:
        return None

    clean = df[[category_col, metric_col]].dropna()
    clean[metric_col] = pd.to_numeric(clean[metric_col], errors="coerce")
    clean = clean.dropna()

    if len(clean) < 10 or clean[metric_col].sum() <= 0:
        return None

    grouped = clean.groupby(category_col)[metric_col].sum().sort_values(ascending=False)
    total_val = grouped.sum()
    cum_share = grouped.cumsum() / total_val

    n_cats = len(grouped)
    if n_cats < 3:
        return None

    # Find how many categories reach threshold
    top_n = int((cum_share <= threshold).sum()) + 1
    top_ratio = top_n / n_cats

    if top_ratio <= 0.30:  # 30% or fewer categories account for >=80% of the metric
        top_names = list(grouped.index[:top_n])
        pct_covered = round(float(cum_share.iloc[top_n - 1]) * 100, 1)
        headline = f"High Concentration: Top {top_n} {category_col}s Drive {pct_covered}% of {metric_col}"
        narrative = (
            f"The top {top_n} categories ({', '.join(str(x) for x in top_names[:3])}"
            f"{'...' if top_n > 3 else ''}) account for {pct_covered}% of total {metric_col}, "
            f"representing a high revenue/activity concentration risk."
        )
        return ActionableInsight(
            id=f"pareto_{category_col}_{metric_col}",
            headline=headline,
            narrative=narrative,
            category=InsightCategory.DRIVER,
            severity=InsightSeverity.HIGH,
            impact_score=8.5,
            suggested_query=f"Show top {category_col} by total {metric_col} as a bar chart",
            metrics={"top_count": top_n, "category_count": n_cats, "share_percentage": pct_covered},
        )
    return None


def find_subgroup_disparities(
    df: pd.DataFrame,
    category_col: str,
    metric_col: str
) -> Optional[ActionableInsight]:
    """
    Detects if a specific subgroup has performance/churn/values diverging by >2x from global mean.
    """
    if category_col not in df.columns or metric_col not in df.columns:
        return None

    clean = df[[category_col, metric_col]].dropna()
    clean[metric_col] = pd.to_numeric(clean[metric_col], errors="coerce").dropna()

    if len(clean) < 20:
        return None

    global_mean = clean[metric_col].mean()
    if global_mean == 0 or np.isnan(global_mean):
        return None

    group_means = clean.groupby(category_col)[metric_col].agg(["mean", "count"])
    # Only consider groups with reasonable sample size
    group_means = group_means[group_means["count"] >= 5]
    if len(group_means) < 2:
        return None

    # Check for extreme group
    group_means["ratio"] = group_means["mean"] / global_mean
    highest = group_means.sort_values(by="ratio", ascending=False).iloc[0]
    top_group = group_means.sort_values(by="ratio", ascending=False).index[0]

    if highest["ratio"] >= 1.8:
        headline = f"Subgroup Outlier: '{top_group}' is {highest['ratio']:.1f}x higher in {metric_col}"
        narrative = (
            f"Cohort '{top_group}' demonstrates an average {metric_col} of {highest['mean']:.2f}, "
            f"which is {highest['ratio']:.1f}x higher than the dataset baseline average ({global_mean:.2f})."
        )
        return ActionableInsight(
            id=f"disparity_{category_col}_{metric_col}",
            headline=headline,
            narrative=narrative,
            category=InsightCategory.OPPORTUNITY if highest["ratio"] > 1 else InsightCategory.RISK,
            severity=InsightSeverity.HIGH if highest["ratio"] > 2.5 else InsightSeverity.MEDIUM,
            impact_score=7.8,
            suggested_query=f"Compare average {metric_col} across {category_col}",
            metrics={"group": str(top_group), "group_mean": round(float(highest["mean"]), 2), "baseline": round(float(global_mean), 2)},
        )
    return None
