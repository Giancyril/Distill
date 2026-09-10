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



def generate_dataset_insights(
    df: pd.DataFrame,
    health: Optional[Any] = None,
    eda: Optional[Any] = None,
    max_insights: int = 6
) -> InsightReport:
    """
    Automated scan of the entire dataset to extract prioritized, actionable intelligence insights.
    Works completely offline without requiring an external LLM.
    """
    insights: List[ActionableInsight] = []
    n_rows, n_cols = df.shape

    # 1. Check Data Health / Completeness Risks
    null_counts = df.isna().sum()
    high_null_cols = null_counts[null_counts / n_rows > 0.15]
    if len(high_null_cols) > 0:
        worst_col = high_null_cols.sort_values(ascending=False).index[0]
        worst_pct = round((null_counts[worst_col] / n_rows) * 100, 1)
        insights.append(ActionableInsight(
            id="risk_high_missingness",
            headline=f"Data Completeness Deficit: '{worst_col}' Has {worst_pct}% Missing Values",
            narrative=(
                f"{len(high_null_cols)} columns exceed a 15% missing data threshold. "
                f"Column '{worst_col}' lacks {worst_pct}% of expected values, which can introduce bias into statistical inferences."
            ),
            category=InsightCategory.RISK,
            severity=InsightSeverity.CRITICAL if worst_pct > 40 else InsightSeverity.HIGH,
            impact_score=9.0 if worst_pct > 40 else 7.5,
            suggested_query=f"Analyze missing value distribution in {worst_col}",
            metrics={"missing_columns": len(high_null_cols), "worst_column": worst_col, "worst_null_pct": worst_pct},
        ))

    # 2. Check Duplication
    n_dups = int(df.duplicated().sum())
    if n_dups > 0:
        dup_pct = round((n_dups / n_rows) * 100, 1)
        insights.append(ActionableInsight(
            id="risk_duplicates",
            headline=f"Redundant Observations: {n_dups:,} Duplicate Rows Detected ({dup_pct}%)",
            narrative=(
                f"{n_dups:,} identical records were discovered. Duplicate rows artificially inflate sample sizes and distort standard errors."
            ),
            category=InsightCategory.RISK,
            severity=InsightSeverity.HIGH if dup_pct > 5 else InsightSeverity.MEDIUM,
            impact_score=7.0,
            suggested_query="How many duplicate rows are in this dataset and which columns cause them?",
            metrics={"duplicates": n_dups, "duplicate_rate": dup_pct},
        ))

    # 3. Check Pareto Driver candidates across categorical and numeric features
    cat_cols = list(df.select_dtypes(exclude=[np.number]).columns)
    num_cols = list(df.select_dtypes(include=[np.number]).columns)

    for c_col in cat_cols[:4]:
        for n_col in num_cols[:4]:
            pareto = find_pareto_drivers(df, c_col, n_col)
            if pareto:
                insights.append(pareto)
                break  # one pareto per cat_col is sufficient

    # 4. Check Subgroup Disparities
    for c_col in cat_cols[:3]:
        for n_col in num_cols[:3]:
            disparity = find_subgroup_disparities(df, c_col, n_col)
            if disparity:
                insights.append(disparity)
                break

    # 5. Check Dominance / Skewness in Numeric Columns
    for n_col in num_cols[:5]:
        s = df[n_col].dropna()
        if len(s) >= 10:
            skew = s.skew()
            if abs(skew) > 2.5:
                direction = "Right" if skew > 0 else "Left"
                insights.append(ActionableInsight(
                    id=f"skew_{n_col}",
                    headline=f"Extreme {direction}-Skew Detected in '{n_col}' (Skewness = {skew:.2f})",
                    narrative=(
                        f"The distribution of '{n_col}' is heavily skewed ({direction}). "
                        f"Consider applying a log1p or Yeo-Johnson transform in the Feature Studio prior to linear modeling."
                    ),
                    category=InsightCategory.ANOMALY,
                    severity=InsightSeverity.MEDIUM,
                    impact_score=6.5,
                    suggested_query=f"Show a histogram and boxplot for {n_col}",
                    metrics={"column": n_col, "skewness": round(float(skew), 2)},
                ))
                break

    # Sort insights by impact score descending
    insights.sort(key=lambda x: x.impact_score, reverse=True)
    selected_insights = insights[:max_insights]

    crit_count = sum(1 for i in selected_insights if i.severity == InsightSeverity.CRITICAL)
    summary = (
        f"Autonomous discovery identified {len(selected_insights)} priority insights "
        f"({crit_count} critical) across {n_rows:,} records and {n_cols} attributes."
    )

    return InsightReport(
        total_insights=len(selected_insights),
        critical_count=crit_count,
        insights=selected_insights,
        executive_summary=summary,
    )
