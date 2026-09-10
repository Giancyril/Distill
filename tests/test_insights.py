"""
tests/test_insights.py - Unit test suite for autonomous insights and narrative synthesis.
"""

import pytest
import numpy as np
import pandas as pd
from core.insights import (
    InsightCategory,
    InsightSeverity,
    find_pareto_drivers,
    find_subgroup_disparities,
    generate_dataset_insights,
    synthesize_executive_narrative,
)


@pytest.fixture
def pareto_sample_df():
    # 10 categories, category 'A' has 85% of the total revenue
    cats = ["A"] * 85 + ["B"] * 5 + ["C"] * 2 + ["D"] * 2 + ["E"] * 2 + ["F"] * 1 + ["G"] * 1 + ["H"] * 1 + ["I"] * 1
    rev = [100.0] * 85 + [10.0] * 5 + [5.0] * 2 + [5.0] * 2 + [5.0] * 2 + [5.0] * 1 + [5.0] * 1 + [5.0] * 1 + [5.0] * 1
    return pd.DataFrame({"category": cats, "revenue": rev})


@pytest.fixture
def disparity_sample_df():
    # Group 'Enterprise' has 3x higher churn/usage
    np.random.seed(42)
    n = 60
    groups = ["Standard"] * 40 + ["Enterprise"] * 20
    metric = [10.0 + np.random.normal(0, 1) for _ in range(40)] + [35.0 + np.random.normal(0, 1) for _ in range(20)]
    return pd.DataFrame({"tier": groups, "value": metric})


class TestDriverDiscovery:
    def test_find_pareto_drivers_detected(self, pareto_sample_df):
        insight = find_pareto_drivers(pareto_sample_df, "category", "revenue")
        assert insight is not None
        assert insight.category == InsightCategory.DRIVER
        assert insight.severity in (InsightSeverity.HIGH, InsightSeverity.CRITICAL)
        assert "Concentration" in insight.headline or "Top" in insight.headline

    def test_find_subgroup_disparities_detected(self, disparity_sample_df):
        insight = find_subgroup_disparities(disparity_sample_df, "tier", "value")
        assert insight is not None
        assert "Enterprise" in insight.headline or "Enterprise" in insight.narrative
        assert insight.impact_score > 7.0


class TestFullInsightScan:
    def test_generate_dataset_insights_detects_missingness_and_dups(self):
        df = pd.DataFrame({
            "col_a": [1, 2, 3, None, None, None, None, None, None, None], # 70% missing
            "col_b": ["x", "x", "x", "y", "y", "y", "z", "z", "z", "z"],
            "val": [10, 10, 10, 20, 20, 20, 30, 30, 30, 30], # duplicates
        })
        rep = generate_dataset_insights(df)
        assert rep.total_insights >= 1
        assert rep.critical_count >= 1
        has_risk = any(i.category == InsightCategory.RISK for i in rep.insights)
        assert has_risk is True

    def test_synthesize_narrative_offline_fallback(self, pareto_sample_df):
        rep = generate_dataset_insights(pareto_sample_df)
        narrative = synthesize_executive_narrative(pareto_sample_df, rep, api_key=None)
        assert isinstance(narrative, str)
        assert len(narrative) > 50
        assert "Executive Overview" in narrative
