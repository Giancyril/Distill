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



def test_normality(series: pd.Series, column_name: str) -> NormalityResult:
    """
    Tests whether a continuous variable follows a normal distribution.
    Uses Shapiro-Wilk for n < 5000, and D'Agostino-Pearson for n >= 5000.
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 8:
        raise ValueError(f"Column '{column_name}' needs at least 8 non-null observations for normality testing.")

    skew = float(stats.skew(clean))
    kurt = float(stats.kurtosis(clean))

    if len(clean) < 5000:
        stat, p_val = stats.shapiro(clean)
    else:
        stat, p_val = stats.normaltest(clean)

    stat = float(stat)
    p_val = float(p_val)
    is_norm = bool(p_val > 0.05)

    if is_norm:
        interp = (
            f"'{column_name}' is consistent with a Gaussian distribution (p = {p_val:.4f} > 0.05). "
            f"Skewness = {skew:.2f}, Kurtosis = {kurt:.2f}."
        )
    else:
        interp = (
            f"'{column_name}' departs significantly from normality (p = {p_val:.4e} <= 0.05). "
            f"Skewness = {skew:.2f}, Kurtosis = {kurt:.2f}. Non-parametric methods recommended."
        )

    return NormalityResult(
        column=column_name,
        statistic=round(stat, 4),
        p_value=round(p_val, 6),
        is_normal=is_norm,
        skewness=round(skew, 2),
        kurtosis=round(kurt, 2),
        interpretation=interp,
    )


def run_multi_group_test(
    groups: List[pd.Series],
    group_names: List[str],
    test_type: Optional[HypothesisTestType] = None,
    alpha: float = 0.05
) -> TestResult:
    """
    Compares 3 or more independent groups using One-Way ANOVA or Kruskal-Wallis.
    """
    if len(groups) < 2:
        raise ValueError("At least 2 groups required for multi-group comparison.")

    cleaned_groups = [pd.to_numeric(g, errors="coerce").dropna() for g in groups]
    for idx, g in enumerate(cleaned_groups):
        if len(g) < 3:
            raise ValueError(f"Group '{group_names[idx]}' has fewer than 3 valid values.")

    # Infer ANOVA vs Kruskal-Wallis if not specified
    if test_type is None:
        all_normal = all(
            (stats.shapiro(g[:500])[1] > 0.05) if len(g) >= 8 else True
            for g in cleaned_groups
        )
        test_type = HypothesisTestType.ANOVA_ONE_WAY if all_normal else HypothesisTestType.KRUSKAL_WALLIS

    means = [round(float(g.mean()), 4) for g in cleaned_groups]
    medians = [round(float(g.median()), 4) for g in cleaned_groups]
    sizes = [len(g) for g in cleaned_groups]

    if test_type == HypothesisTestType.ANOVA_ONE_WAY:
        stat, p_val = stats.f_oneway(*cleaned_groups)
        t_name = "One-Way Analysis of Variance (ANOVA F-Test)"
    elif test_type == HypothesisTestType.KRUSKAL_WALLIS:
        stat, p_val = stats.kruskal(*cleaned_groups)
        t_name = "Kruskal-Wallis H-Test (Non-Parametric Multi-Group)"
    else:
        raise ValueError(f"Unsupported multi-group test: {test_type}")

    stat = float(stat)
    p_val = float(p_val)
    is_sig = bool(p_val < alpha)

    if is_sig:
        interp = (
            f"Statistically significant variance across groups detected by {t_name} "
            f"(p = {p_val:.4e} < {alpha}). At least one group differs significantly."
        )
    else:
        interp = (
            f"No statistically significant difference detected across groups "
            f"(p = {p_val:.4f} >= {alpha}). Insufficient evidence to reject homogeneity."
        )

    return TestResult(
        test_name=t_name,
        statistic=round(stat, 4),
        p_value=round(p_val, 6),
        is_significant=is_sig,
        interpretation=interp,
        group_names=group_names,
        sample_sizes=sizes,
        group_means=means,
        group_medians=medians,
        confidence_level=1.0 - alpha,
    )



@dataclass
class CorrelationPair:
    var1: str
    var2: str
    coefficient: float
    p_value: float
    is_significant: bool
    strength: str
    significance_symbol: str


def compute_correlation_significance(
    df: pd.DataFrame,
    method: str = "pearson",
    alpha: float = 0.05
) -> Tuple[pd.DataFrame, pd.DataFrame, List[CorrelationPair]]:
    """
    Computes both the correlation matrix and the corresponding p-value matrix for all
    numeric columns, alongside ranked significant relationship pairs.
    """
    num_df = df.select_dtypes(include=[np.number]).dropna(how="all", axis=1)
    cols = list(num_df.columns)
    n = len(cols)

    if n < 2:
        empty = pd.DataFrame()
        return empty, empty, []

    corr_mat = pd.DataFrame(np.ones((n, n)), index=cols, columns=cols)
    pval_mat = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)
    pairs: List[CorrelationPair] = []

    for i in range(n):
        for j in range(i + 1, n):
            c1, c2 = cols[i], cols[j]
            valid = num_df[[c1, c2]].dropna()
            if len(valid) >= 4:
                if method == "spearman":
                    r, p = stats.spearmanr(valid[c1], valid[c2])
                else:
                    r, p = stats.pearsonr(valid[c1], valid[c2])
            else:
                r, p = 0.0, 1.0

            r = float(r)
            p = float(p)

            corr_mat.loc[c1, c2] = round(r, 4)
            corr_mat.loc[c2, c1] = round(r, 4)
            pval_mat.loc[c1, c2] = round(p, 6)
            pval_mat.loc[c2, c1] = round(p, 6)

            is_sig = p < alpha
            abs_r = abs(r)
            if abs_r >= 0.7:
                strength = "Strong"
            elif abs_r >= 0.4:
                strength = "Moderate"
            elif abs_r >= 0.2:
                strength = "Weak"
            else:
                strength = "Negligible"

            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            else:
                stars = "ns"

            pairs.append(CorrelationPair(
                var1=c1,
                var2=c2,
                coefficient=round(r, 4),
                p_value=round(p, 6),
                is_significant=is_sig,
                strength=strength,
                significance_symbol=stars,
            ))

    pairs.sort(key=lambda x: abs(x.coefficient), reverse=True)
    return corr_mat, pval_mat, pairs


from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


def detect_multivariate_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42
) -> AnomalyDetectionResult:
    """
    Detects complex multivariate outliers across numeric dimensions using Isolation Forest.
    Unlike univariate IQR, this flags records where feature interactions are statistically abnormal.
    """
    num_df = df.select_dtypes(include=[np.number]).dropna(how="all", axis=1)
    feature_cols = list(num_df.columns)

    if len(feature_cols) < 2:
        raise ValueError("Multivariate anomaly detection requires at least 2 numeric feature columns.")

    if len(num_df) < 10:
        raise ValueError("At least 10 rows are required for multivariate anomaly detection.")

    # Impute missing values with median for anomaly detection
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X_imputed = imputer.fit_transform(num_df)
    X_scaled = scaler.fit_transform(X_imputed)

    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100
    )
    preds = iso.fit_predict(X_scaled)  # -1 for anomaly, 1 for inlier
    raw_scores = iso.score_samples(X_scaled)  # higher = more normal, lower = more anomalous

    # Transform scores: 0 (most normal) to 1 (most anomalous)
    min_s = float(np.min(raw_scores))
    max_s = float(np.max(raw_scores))
    if max_s > min_s:
        normalized_anomaly_scores = [round(float(1.0 - (s - min_s) / (max_s - min_s)), 4) for s in raw_scores]
    else:
        normalized_anomaly_scores = [0.0] * len(raw_scores)

    outlier_mask = preds == -1
    outlier_indices = list(num_df.index[outlier_mask])
    outlier_count = int(np.sum(outlier_mask))
    outlier_pct = round((outlier_count / len(num_df)) * 100, 2)

    # Attach anomaly score and sort top anomalies
    df_copy = df.copy()
    df_copy["_anomaly_score"] = normalized_anomaly_scores
    top_anomalous = df_copy.loc[outlier_indices].sort_values(by="_anomaly_score", ascending=False).head(20)

    return AnomalyDetectionResult(
        total_records=len(df),
        outlier_count=outlier_count,
        outlier_percentage=outlier_pct,
        contamination_rate=contamination,
        outlier_indices=outlier_indices,
        anomaly_scores=normalized_anomaly_scores,
        top_anomalous_records=top_anomalous,
        feature_columns=feature_cols,
    )
