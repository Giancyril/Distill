"""
tests/test_analytics_advanced.py - Rigorous test suite for statistical tests and anomaly detection.
"""

import pytest
import numpy as np
import pandas as pd
from core.analytics_advanced import (
    HypothesisTestType,
    recommend_hypothesis_test,
    run_two_sample_test,
    check_normality,
    run_multi_group_test,
    compute_correlation_significance,
    detect_multivariate_anomalies,
)


@pytest.fixture
def synth_stats_data():
    np.random.seed(42)
    group_a = np.random.normal(loc=100, scale=15, size=60)
    group_b = np.random.normal(loc=115, scale=15, size=60)  # genuinely higher mean
    group_c = np.random.normal(loc=101, scale=15, size=60)  # similar to A
    return pd.DataFrame({"A": group_a, "B": group_b, "C": group_c})


class TestHypothesisTests:
    def test_recommend_hypothesis_test(self, synth_stats_data):
        tt, reason = recommend_hypothesis_test(synth_stats_data["A"], synth_stats_data["B"])
        assert tt in [HypothesisTestType.TWO_SAMPLE_TTEST, HypothesisTestType.WELCHS_TTEST, HypothesisTestType.MANN_WHITNEY_U]
        assert len(reason) > 0

    def test_two_sample_ttest_significance(self, synth_stats_data):
        res = run_two_sample_test(
            synth_stats_data["A"],
            synth_stats_data["B"],
            name_a="Group A",
            name_b="Group B",
            test_type=HypothesisTestType.TWO_SAMPLE_TTEST,
        )
        assert res.is_significant is True
        assert res.p_value < 0.05
        assert len(res.group_means) == 2
        assert "Statistically significant" in res.interpretation

    def test_two_sample_no_difference(self, synth_stats_data):
        res = run_two_sample_test(
            synth_stats_data["A"],
            synth_stats_data["C"],
            name_a="Group A",
            name_b="Group C",
            test_type=HypothesisTestType.TWO_SAMPLE_TTEST,
        )
        assert res.is_significant is False
        assert res.p_value >= 0.05

    def test_mann_whitney_u(self, synth_stats_data):
        res = run_two_sample_test(
            synth_stats_data["A"],
            synth_stats_data["B"],
            test_type=HypothesisTestType.MANN_WHITNEY_U,
        )
        assert res.is_significant is True
        assert "Mann-Whitney" in res.test_name


class TestNormalityAndMultiGroup:
    def test_normality_gaussian(self):
        np.random.seed(42)
        normal_data = pd.Series(np.random.normal(50, 5, size=100))
        norm_res = check_normality(normal_data, "normal_var")
        assert norm_res.is_normal is True
        assert norm_res.p_value > 0.05

    def test_normality_exponential(self):
        np.random.seed(42)
        exp_data = pd.Series(np.random.exponential(scale=2.0, size=200))
        norm_res = check_normality(exp_data, "exp_var")
        assert norm_res.is_normal is False
        assert norm_res.p_value <= 0.05

    def test_anova_multi_group(self, synth_stats_data):
        res = run_multi_group_test(
            [synth_stats_data["A"], synth_stats_data["B"], synth_stats_data["C"]],
            ["A", "B", "C"],
            test_type=HypothesisTestType.ANOVA_ONE_WAY,
        )
        assert res.is_significant is True
        assert len(res.group_means) == 3


class TestCorrelationAndAnomalies:
    def test_correlation_significance(self):
        np.random.seed(42)
        x = np.linspace(1, 100, 50)
        y = 2.0 * x + np.random.normal(0, 5, 50)
        z = np.random.uniform(0, 10, 50)
        df = pd.DataFrame({"x": x, "y": y, "z": z})

        c_mat, p_mat, pairs = compute_correlation_significance(df)
        assert c_mat.shape == (3, 3)
        assert p_mat.shape == (3, 3)
        assert len(pairs) == 3

        # x and y should have near 1.0 correlation with high significance
        xy_pair = next(p for p in pairs if (p.var1 == "x" and p.var2 == "y") or (p.var1 == "y" and p.var2 == "x"))
        assert xy_pair.coefficient > 0.9
        assert xy_pair.is_significant is True
        assert xy_pair.significance_symbol in ["***", "**"]

    def test_multivariate_anomalies(self):
        np.random.seed(42)
        n = 80
        x = np.random.normal(10, 2, n)
        y = np.random.normal(20, 3, n)
        df = pd.DataFrame({"feat1": x, "feat2": y})

        # Inject extreme multivariate outlier
        df.loc[n] = [100.0, -50.0]

        res = detect_multivariate_anomalies(df, contamination=0.05)
        assert res.outlier_count >= 1
        assert n in res.outlier_indices
        assert len(res.top_anomalous_records) > 0
