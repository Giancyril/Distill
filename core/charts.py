"""
core/charts.py — Intent-aware Plotly chart generation with unified design system.
Maps analytical intents to the optimal chart type with a curated color palette.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure

# ---------------------------------------------------------------------------
# Design System Tokens
# ---------------------------------------------------------------------------

PALETTE = {
    "indigo":  "#6366F1",
    "violet":  "#8B5CF6",
    "emerald": "#10B981",
    "amber":   "#F59E0B",
    "rose":    "#F43F5E",
    "sky":     "#0EA5E9",
    "slate":   "#64748B",
}

SEQUENTIAL_COLORS = [
    PALETTE["indigo"],
    PALETTE["violet"],
    PALETTE["emerald"],
    PALETTE["amber"],
    PALETTE["rose"],
    PALETTE["sky"],
    PALETTE["slate"],
]

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", size=13, color="#0F172A"),
    margin=dict(l=16, r=16, t=48, b=16),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
    hoverlabel=dict(
        bgcolor="white",
        bordercolor="#E2E8F0",
        font_size=12,
        font_color="#0F172A",
    ),
    xaxis=dict(
        showgrid=True,
        gridcolor="#F1F5F9",
        gridwidth=1,
        linecolor="#E2E8F0",
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="#F1F5F9",
        gridwidth=1,
        linecolor="#E2E8F0",
    ),
)


def _apply_layout(fig: Figure, title: str = "") -> Figure:
    """Apply unified design system layout to any Plotly figure."""
    layout = dict(**CHART_LAYOUT)
    if title:
        layout["title"] = dict(text=title, font=dict(size=16, color="#0F172A", weight=700))
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Chart Generators
# ---------------------------------------------------------------------------

def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    color: Optional[str] = None,
    horizontal: bool = False,
) -> Figure:
    """Category comparison — vertical or horizontal bar chart."""
    orientation = "h" if horizontal else "v"
    _x, _y = (y, x) if horizontal else (x, y)
    fig = px.bar(
        df,
        x=_x,
        y=_y,
        color=color,
        orientation=orientation,
        color_discrete_sequence=SEQUENTIAL_COLORS,
        template="none",
    )
    fig.update_traces(marker_line_width=0)
    return _apply_layout(fig, title or f"{y} by {x}")


def line_chart(
    df: pd.DataFrame,
    x: str,
    y: str | List[str],
    title: str = "",
    color: Optional[str] = None,
) -> Figure:
    """Temporal / trend analysis — smooth line chart."""
    y_cols = [y] if isinstance(y, str) else y
    fig = px.line(
        df,
        x=x,
        y=y_cols,
        color=color,
        color_discrete_sequence=SEQUENTIAL_COLORS,
        template="none",
        markers=True,
    )
    fig.update_traces(line_width=2.5)
    return _apply_layout(fig, title or f"{y} over {x}")


def scatter_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    color: Optional[str] = None,
    size: Optional[str] = None,
) -> Figure:
    """Correlation / relationship — scatter or bubble plot."""
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        size=size,
        color_discrete_sequence=SEQUENTIAL_COLORS,
        template="none",
        opacity=0.75,
    )
    return _apply_layout(fig, title or f"{y} vs {x}")


def histogram_chart(
    df: pd.DataFrame,
    x: str,
    title: str = "",
    nbins: int = 30,
    color: Optional[str] = None,
) -> Figure:
    """Distribution — histogram."""
    fig = px.histogram(
        df,
        x=x,
        nbins=nbins,
        color=color,
        color_discrete_sequence=SEQUENTIAL_COLORS,
        template="none",
    )
    fig.update_traces(marker_line_width=0.5, marker_line_color="white")
    return _apply_layout(fig, title or f"Distribution of {x}")


def box_chart(
    df: pd.DataFrame,
    x: Optional[str],
    y: str,
    title: str = "",
    color: Optional[str] = None,
) -> Figure:
    """Distribution by group — box plot."""
    fig = px.box(
        df,
        x=x,
        y=y,
        color=color or x,
        color_discrete_sequence=SEQUENTIAL_COLORS,
        template="none",
    )
    return _apply_layout(fig, title or f"Distribution of {y}")


def pie_chart(
    df: pd.DataFrame,
    names: str,
    values: str,
    title: str = "",
    donut: bool = True,
) -> Figure:
    """Composition / share — donut or pie chart."""
    hole = 0.45 if donut else 0
    fig = px.pie(
        df,
        names=names,
        values=values,
        hole=hole,
        color_discrete_sequence=SEQUENTIAL_COLORS,
        template="none",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return _apply_layout(fig, title or f"{values} by {names}")


def area_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    color: Optional[str] = None,
) -> Figure:
    """Temporal cumulative area chart."""
    fig = px.area(
        df,
        x=x,
        y=y,
        color=color,
        color_discrete_sequence=SEQUENTIAL_COLORS,
        template="none",
    )
    return _apply_layout(fig, title or f"{y} over {x}")


def heatmap_chart(
    pivot_df: pd.DataFrame,
    title: str = "",
) -> Figure:
    """Correlation matrix or pivot heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=list(pivot_df.columns),
        y=list(pivot_df.index),
        colorscale=[[0, "#EEF2FF"], [0.5, PALETTE["indigo"]], [1, PALETTE["violet"]]],
        showscale=True,
    ))
    return _apply_layout(fig, title)


# ---------------------------------------------------------------------------
# Intent Router
# ---------------------------------------------------------------------------

INTENT_MAP = {
    "bar_chart":   bar_chart,
    "line_chart":  line_chart,
    "area_chart":  area_chart,
    "scatter":     scatter_chart,
    "pie_chart":   pie_chart,
    "histogram":   histogram_chart,
    "box_plot":    box_chart,
    "heatmap":     heatmap_chart,
}


def build_chart_from_intent(
    intent: str,
    df: pd.DataFrame,
    x: Optional[str] = None,
    y: Optional[str] = None,
    title: str = "",
    **kwargs,
) -> Optional[Figure]:
    """
    Route an analytical intent to the optimal chart generator.

    Parameters
    ----------
    intent : str
        One of: bar_chart, line_chart, area_chart, scatter, pie_chart,
        histogram, box_plot, heatmap, table, scalar, none.
    df : pd.DataFrame
        Data to plot.
    x, y : str, optional
        Column names for axes.
    title : str
        Chart title.
    **kwargs
        Extra keyword args forwarded to the chart function.

    Returns
    -------
    Figure or None (when intent is table/scalar/none)
    """
    if intent not in INTENT_MAP:
        return None

    fn = INTENT_MAP[intent]
    try:
        if intent == "histogram":
            return fn(df, x=x or df.columns[0], title=title, **kwargs)
        elif intent == "heatmap":
            return fn(df, title=title)
        elif intent == "pie_chart":
            names = x or df.columns[0]
            values = y or df.columns[1]
            return fn(df, names=names, values=values, title=title, **kwargs)
        elif intent == "box_plot":
            return fn(df, x=x, y=y or df.columns[0], title=title, **kwargs)
        else:
            return fn(df, x=x or df.columns[0], y=y or df.columns[1], title=title, **kwargs)
    except Exception:
        return None


def correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Matrix") -> Figure:
    """
    Convenience function: generate a correlation matrix heatmap
    from the numeric columns of a DataFrame.
    """
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()
    return heatmap_chart(corr, title=title)
