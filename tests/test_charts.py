"""
tests/test_charts.py — Tests for core/charts.py intent-aware chart generation.
"""
import pandas as pd
import pytest

from core.charts import (
    area_chart,
    bar_chart,
    box_chart,
    build_chart_from_intent,
    correlation_heatmap,
    histogram_chart,
    line_chart,
    pie_chart,
    scatter_chart,
)


@pytest.fixture()
def sales_df() -> pd.DataFrame:
    return pd.DataFrame({
        "month": ["Jan", "Feb", "Mar", "Apr", "May"],
        "revenue": [1200, 1500, 980, 2100, 1750],
        "units": [12, 18, 9, 25, 20],
        "category": ["A", "B", "A", "C", "B"],
    })


@pytest.fixture()
def numeric_df() -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "x": rng.normal(0, 1, 50),
        "y": rng.normal(5, 2, 50),
        "group": (rng.integers(0, 3, 50)).astype(str),
    })


class TestBarChart:
    def test_returns_figure(self, sales_df):
        from plotly.graph_objects import Figure
        fig = bar_chart(sales_df, x="month", y="revenue")
        assert isinstance(fig, Figure)

    def test_has_data(self, sales_df):
        fig = bar_chart(sales_df, x="month", y="revenue")
        assert len(fig.data) > 0

    def test_title_applied(self, sales_df):
        fig = bar_chart(sales_df, x="month", y="revenue", title="My Chart")
        assert "My Chart" in fig.layout.title.text

    def test_horizontal(self, sales_df):
        fig = bar_chart(sales_df, x="category", y="revenue", horizontal=True)
        assert fig.data[0].orientation == "h"


class TestLineChart:
    def test_returns_figure(self, sales_df):
        from plotly.graph_objects import Figure
        fig = line_chart(sales_df, x="month", y="revenue")
        assert isinstance(fig, Figure)

    def test_markers_enabled(self, sales_df):
        fig = line_chart(sales_df, x="month", y="revenue")
        assert fig.data[0].mode is not None


class TestScatterChart:
    def test_returns_figure(self, numeric_df):
        from plotly.graph_objects import Figure
        fig = scatter_chart(numeric_df, x="x", y="y")
        assert isinstance(fig, Figure)


class TestHistogramChart:
    def test_returns_figure(self, numeric_df):
        from plotly.graph_objects import Figure
        fig = histogram_chart(numeric_df, x="x")
        assert isinstance(fig, Figure)


class TestPieChart:
    def test_returns_figure(self, sales_df):
        from plotly.graph_objects import Figure
        agg = sales_df.groupby("category")["revenue"].sum().reset_index()
        fig = pie_chart(agg, names="category", values="revenue")
        assert isinstance(fig, Figure)

    def test_donut_hole(self, sales_df):
        agg = sales_df.groupby("category")["revenue"].sum().reset_index()
        fig = pie_chart(agg, names="category", values="revenue", donut=True)
        assert fig.data[0].hole > 0


class TestAreaChart:
    def test_returns_figure(self, sales_df):
        from plotly.graph_objects import Figure
        fig = area_chart(sales_df, x="month", y="revenue")
        assert isinstance(fig, Figure)


class TestBoxChart:
    def test_returns_figure(self, sales_df):
        from plotly.graph_objects import Figure
        fig = box_chart(sales_df, x="category", y="revenue")
        assert isinstance(fig, Figure)


class TestCorrelationHeatmap:
    def test_returns_figure(self, sales_df):
        from plotly.graph_objects import Figure
        fig = correlation_heatmap(sales_df)
        assert isinstance(fig, Figure)


class TestBuildChartFromIntent:
    def test_bar_chart_intent(self, sales_df):
        from plotly.graph_objects import Figure
        fig = build_chart_from_intent("bar_chart", sales_df, x="month", y="revenue")
        assert isinstance(fig, Figure)

    def test_line_chart_intent(self, sales_df):
        from plotly.graph_objects import Figure
        fig = build_chart_from_intent("line_chart", sales_df, x="month", y="revenue")
        assert isinstance(fig, Figure)

    def test_histogram_intent(self, numeric_df):
        from plotly.graph_objects import Figure
        fig = build_chart_from_intent("histogram", numeric_df, x="x")
        assert isinstance(fig, Figure)

    def test_table_intent_returns_none(self, sales_df):
        result = build_chart_from_intent("table", sales_df, x="month", y="revenue")
        assert result is None

    def test_unknown_intent_returns_none(self, sales_df):
        result = build_chart_from_intent("unknown_type", sales_df)
        assert result is None

    def test_all_chart_intents_supported(self, sales_df, numeric_df):
        intents_to_test = [
            ("bar_chart", sales_df, "month", "revenue"),
            ("line_chart", sales_df, "month", "revenue"),
            ("area_chart", sales_df, "month", "revenue"),
            ("scatter", numeric_df, "x", "y"),
        ]
        from plotly.graph_objects import Figure
        for intent, df, x, y in intents_to_test:
            fig = build_chart_from_intent(intent, df, x=x, y=y)
            assert isinstance(fig, Figure), f"Failed for intent: {intent}"

    def test_layout_applied(self, sales_df):
        """Verify design system layout is applied (transparent background)."""
        fig = build_chart_from_intent("bar_chart", sales_df, x="month", y="revenue")
        assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
