"""
core/analytics_advanced.py - Advanced Statistical Testing & Multidimensional Anomaly Detection
Provides rigorous hypothesis testing (t-test, ANOVA, Mann-Whitney, Kruskal-Wallis),
normality verification, significance-annotated correlation, and Isolation Forest anomaly discovery.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


class HypothesisTestType(str, Enum):
    TWO_SAMPLE_TTEST = "two_sample_ttest"
    WELCHS_TTEST = "welchs_ttest"
    MANN_WHITNEY_U = "mann_whitney_u"
    ANOVA_ONE_WAY = "anova_one_way"
    KRUSKAL_WALLIS = "kruskal_wallis"


@dataclass
class TestResult:
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool  # p < 0.05
    interpretation: str
    group_names: List[str]
    sample_sizes: List[int]
    group_means: Optional[List[float]] = None
    group_medians: Optional[List[float]] = None
    confidence_level: float = 0.95


@dataclass
class NormalityResult:
    column: str
    statistic: float
    p_value: float
    is_normal: bool
    skewness: float
    kurtosis: float
    interpretation: str


@dataclass
class AnomalyDetectionResult:
    total_records: int
    outlier_count: int
    outlier_percentage: float
    contamination_rate: float
    outlier_indices: List[int]
    anomaly_scores: List[float]
    top_anomalous_records: pd.DataFrame
    feature_columns: List[str]


def recommend_hypothesis_test(
    series_a: pd.Series,
    series_b: pd.Series
) -> Tuple[HypothesisTestType, str]:
    """
    Recommends parametric vs non-parametric test based on sample size and normality.
    """
    clean_a = series_a.dropna()
    clean_b = series_b.dropna()

    if len(clean_a) < 5 or len(clean_b) < 5:
        raise ValueError("Samples must have at least 5 observations for hypothesis testing.")

    # Check normality using Shapiro-Wilk or normaltest
    norm_a = True
    norm_b = True
    if len(clean_a) >= 8:
        try:
            _, p_a = stats.shapiro(clean_a[:500])
            norm_a = p_a > 0.05
        except Exception:
            norm_a = True

    if len(clean_b) >= 8:
        try:
            _, p_b = stats.shapiro(clean_b[:500])
            norm_b = p_b > 0.05
        except Exception:
            norm_b = True

    # Check variance equality using Levene's test
    equal_var = True
    try:
        _, p_var = stats.levene(clean_a, clean_b)
        equal_var = p_var > 0.05
    except Exception:
        equal_var = True

    if norm_a and norm_b:
        if equal_var:
            return HypothesisTestType.TWO_SAMPLE_TTEST, "Student's t-test (Normality and equal variances met)"
        else:
            return HypothesisTestType.WELCHS_TTEST, "Welch's t-test (Normality met, unequal variances)"
    else:
        return HypothesisTestType.MANN_WHITNEY_U, "Mann-Whitney U test (Non-parametric: distributions non-normal)"



def run_two_sample_test(
    group_a: pd.Series,
    group_b: pd.Series,
    name_a: str = "Group A",
    name_b: str = "Group B",
    test_type: Optional[HypothesisTestType] = None,
    alpha: float = 0.05
) -> TestResult:
    """
    Executes a two-sample hypothesis test comparing group_a and group_b.
    """
    clean_a = pd.to_numeric(group_a, errors="coerce").dropna()
    clean_b = pd.to_numeric(group_b, errors="coerce").dropna()

    if len(clean_a) < 3 or len(clean_b) < 3:
        raise ValueError("Both groups must contain at least 3 valid numeric values.")

    if test_type is None:
        test_type, _ = recommend_hypothesis_test(clean_a, clean_b)

    mean_a = float(clean_a.mean())
    mean_b = float(clean_b.mean())
    med_a = float(clean_a.median())
    med_b = float(clean_b.median())

    if test_type == HypothesisTestType.TWO_SAMPLE_TTEST:
        stat, p_val = stats.ttest_ind(clean_a, clean_b, equal_var=True)
        t_name = "Independent Samples Two-Sample Student's t-Test"
    elif test_type == HypothesisTestType.WELCHS_TTEST:
        stat, p_val = stats.ttest_ind(clean_a, clean_b, equal_var=False)
        t_name = "Welch's t-Test (Unequal Variances)"
    elif test_type == HypothesisTestType.MANN_WHITNEY_U:
        stat, p_val = stats.mannwhitneyu(clean_a, clean_b, alternative="two-sided")
        t_name = "Mann-Whitney U Test (Non-Parametric)"
    else:
        raise ValueError(f"Unsupported two-sample test type: {test_type}")

    stat = float(stat)
    p_val = float(p_val)
    is_sig = bool(p_val < alpha)

    if is_sig:
        interp = (
            f"Statistically significant difference detected between {name_a} and {name_b} "
            f"(p = {p_val:.4e} < {alpha}). The null hypothesis of identical distributions is rejected."
        )
    else:
        interp = (
            f"No statistically significant difference detected between {name_a} and {name_b} "
            f"(p = {p_val:.4f} >= {alpha}). Insufficient evidence to reject the null hypothesis."
        )

    return TestResult(
        test_name=t_name,
        statistic=round(stat, 4),
        p_value=round(p_val, 6),
        is_significant=is_sig,
        interpretation=interp,
        group_names=[name_a, name_b],
        sample_sizes=[len(clean_a), len(clean_b)],
        group_means=[round(mean_a, 4), round(mean_b, 4)],
        group_medians=[round(med_a, 4), round(med_b, 4)],
        confidence_level=1.0 - alpha,
    )
