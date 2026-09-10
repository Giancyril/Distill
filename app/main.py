"""
app/main.py — AI Data Analysis: Professional Streamlit Dashboard
Turn raw data into actionable insights in seconds.
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Add project root to path for core imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cleaning import audit_data_health, clean_dataset, detect_column_types
from core.eda import generate_eda_report
from core.ingestion import IngestionError, load_dataset
from core.query_engine import run_query
from core.sandbox import execute_in_sandbox

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com",
        "About": "# AI Data Analysis\nTurn raw data into actionable insights in seconds.",
    },
)

# ---------------------------------------------------------------------------
# CSS Injection — Premium Design System
# ---------------------------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
/* === Base Typography === */
html, body, .stApp, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif !important;
}

/* === Main Container === */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* === Hero Header === */
.hero-banner {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 60%, #6D28D9 100%);
    color: white;
    padding: 1.75rem 2rem;
    border-radius: 1.25rem;
    margin-bottom: 1.75rem;
    box-shadow: 0 10px 25px -5px rgba(79,70,229,0.35), 0 4px 6px -2px rgba(79,70,229,0.15);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: rgba(255,255,255,0.06);
    border-radius: 50%;
}
.hero-title { font-size: 1.85rem; font-weight: 800; margin-bottom: 0.35rem; }
.hero-subtitle { font-size: 0.95rem; opacity: 0.88; max-width: 560px; }

/* === KPI Cards === */
.kpi-grid { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.kpi-card {
    background: #fff;
    border: 1px solid #EEF2FF;
    border-radius: 1rem;
    padding: 1.1rem 1.3rem;
    flex: 1;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    position: relative; overflow: hidden;
}
.kpi-card::after {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #6366F1, #8B5CF6);
}
.kpi-icon { font-size: 1.4rem; margin-bottom: 0.4rem; }
.kpi-label { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; color: #64748B; }
.kpi-value { font-size: 1.75rem; font-weight: 700; color: #0F172A; margin-top: 0.15rem; }
.kpi-sub { font-size: 0.78rem; color: #94A3B8; margin-top: 0.15rem; }

/* === Section Headers === */
.section-header {
    font-size: 1rem;
    font-weight: 700;
    color: #1E293B;
    padding: 0.5rem 0;
    border-bottom: 2px solid #EEF2FF;
    margin-bottom: 1rem;
}

/* === Chat / Query UI === */
.query-box {
    background: linear-gradient(135deg, #F8FAFF 0%, #F0F4FF 100%);
    border: 1px solid #C7D2FE;
    border-radius: 1rem;
    padding: 1.25rem;
    margin-bottom: 1rem;
}

/* === Result cards === */
.result-card {
    background: #fff;
    border: 1px solid #E2E8F0;
    border-radius: 0.875rem;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.result-question {
    font-size: 0.875rem;
    font-weight: 600;
    color: #4F46E5;
    margin-bottom: 0.5rem;
}
.result-answer {
    font-size: 0.9rem;
    color: #374151;
    line-height: 1.6;
    margin-bottom: 0.75rem;
    background: #F8FAFF;
    padding: 0.75rem;
    border-radius: 0.5rem;
    border-left: 3px solid #6366F1;
}

/* === Health badges === */
.badge-ok { color: #065F46; background: #D1FAE5; padding: 2px 8px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-warn { color: #92400E; background: #FEF3C7; padding: 2px 8px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-error { color: #7F1D1D; background: #FEE2E2; padding: 2px 8px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-type { color: #1E40AF; background: #DBEAFE; padding: 2px 8px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; }

/* === Code block === */
.code-inspector {
    background: #0F172A;
    color: #A5F3FC;
    border-radius: 0.75rem;
    padding: 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    overflow-x: auto;
    white-space: pre-wrap;
}

/* === Sidebar === */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F8FAFF 0%, #F0F4FF 100%);
    border-right: 1px solid #E0E7FF;
}

/* === Tabs override === */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem;
    background: #F1F5F9;
    border-radius: 0.875rem;
    padding: 0.25rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 0.625rem;
    padding: 0.5rem 1.25rem;
    font-weight: 600;
    color: #64748B;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #4F46E5 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

/* === Expander === */
.streamlit-expanderHeader {
    font-weight: 600;
    color: #374151;
}

/* === Scrollbar === */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #F1F5F9; }
::-webkit-scrollbar-thumb { background: #C7D2FE; border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: #6366F1; }

/* Remove default streamlit top padding */
#root > div:first-child { padding-top: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "dataset": None,
        "filename": None,
        "health_report": None,
        "eda_report": None,
        "inferred_types": None,
        "cleaned_df": None,
        "clean_log": [],
        "query_history": [],   # list of {question, answer, code, result_json, result_type, time_ms}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ---------------------------------------------------------------------------
# Data Loading Helper
# ---------------------------------------------------------------------------
def _load_file(source):
    """Load file, run cleaning pipeline, and cache results in session state."""
    try:
        df = load_dataset(source)
    except IngestionError as e:
        st.error(f"❌ **Ingestion Error:** {e}")
        return

    types = detect_column_types(df)
    health = audit_data_health(df)
    eda = generate_eda_report(df, types)

    st.session_state.dataset = df
    st.session_state.filename = getattr(source, "name", str(source))
    st.session_state.inferred_types = types
    st.session_state.health_report = health
    st.session_state.eda_report = eda
    st.session_state.cleaned_df = None
    st.session_state.clean_log = []
    st.session_state.query_history = []


# ---------------------------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">📊 AI Data Analysis</div>
    <div class="hero-subtitle">
        Upload your CSV or Excel dataset to instantly profile trends, surface anomalies,
        and query your data in plain English — powered by GPT-4o.
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — Data Upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📁 Data Source")
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
        help="Upload tabular data up to 50MB.",
    )
    if uploaded_file and st.session_state.filename != uploaded_file.name:
        with st.spinner("Analyzing dataset…"):
            _load_file(uploaded_file)
        st.success(f"Loaded **{uploaded_file.name}**")

    st.markdown("---")
    st.markdown("**Try Sample Datasets**")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📈 Sales", use_container_width=True, key="btn_sales"):
            with st.spinner("Loading…"):
                _load_file("sample_data/sales_clean.csv")
            st.rerun()
    with col_b:
        if st.button("📉 Churn", use_container_width=True, key="btn_churn"):
            with st.spinner("Loading…"):
                _load_file("sample_data/customer_churn_messy.csv")
            st.rerun()

    if st.session_state.dataset is not None:
        st.markdown("---")
        st.markdown("**API Configuration**")
        has_key = bool(os.getenv("OPENAI_API_KEY"))
        if has_key:
            st.success("✅ OpenAI API key detected")
        else:
            api_key_input = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-...",
                key="api_key_input",
            )
            if api_key_input:
                os.environ["OPENAI_API_KEY"] = api_key_input
                st.success("API key set for this session.")

        st.markdown("---")
        if st.button("🔄 Reset", type="secondary", use_container_width=True, key="btn_reset"):
            for k in ["dataset", "filename", "health_report", "eda_report",
                      "inferred_types", "cleaned_df", "clean_log", "query_history"]:
                st.session_state[k] = None if k not in ["clean_log", "query_history"] else []
            st.rerun()

    if st.session_state.eda_report:
        st.markdown("---")
        eda = st.session_state.eda_report
        st.markdown(f"**{st.session_state.filename}**")
        st.markdown(f"🗂️ `{eda.total_rows:,}` rows × `{eda.total_columns}` cols")


# ---------------------------------------------------------------------------
# No Dataset State
# ---------------------------------------------------------------------------
if st.session_state.dataset is None:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 0; color: #94A3B8;">
            <div style="font-size:4rem">📂</div>
            <h3 style="color:#1E293B; margin:1rem 0 0.5rem">No dataset loaded</h3>
            <p>Upload a CSV or Excel file using the sidebar,<br>or try one of the sample datasets.</p>
        </div>
        """, unsafe_allow_html=True)
    st.stop()


# ---------------------------------------------------------------------------
# Main Content — Tabs
# ---------------------------------------------------------------------------
df = st.session_state.dataset
health = st.session_state.health_report
eda = st.session_state.eda_report
types = st.session_state.inferred_types

tab_dashboard, tab_query, tab_health = st.tabs([
    "📊 Dashboard",
    "🤖 Query Workspace",
    "🩺 Data Health",
])


# ============================================================
# TAB 1 — Dashboard
# ============================================================
with tab_dashboard:
    # KPI Row
    missing_pct = health.missing_percentage if health else 0.0
    dups = health.duplicate_rows if health else 0
    numeric_profiles = eda.numeric_profiles if eda else []

    kpi_html = f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-icon">🗂️</div>
            <div class="kpi-label">Total Rows</div>
            <div class="kpi-value">{len(df):,}</div>
            <div class="kpi-sub">records</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">📋</div>
            <div class="kpi-label">Columns</div>
            <div class="kpi-value">{len(df.columns)}</div>
            <div class="kpi-sub">features</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">❓</div>
            <div class="kpi-label">Missing Cells</div>
            <div class="kpi-value">{health.missing_cells if health else 0:,}</div>
            <div class="kpi-sub">{missing_pct:.1f}% of data</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">🔢</div>
            <div class="kpi-label">Numeric Cols</div>
            <div class="kpi-value">{len(numeric_profiles)}</div>
            <div class="kpi-sub">profiled</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">{'⚠️' if dups > 0 else '✅'}</div>
            <div class="kpi-label">Duplicates</div>
            <div class="kpi-value">{dups}</div>
            <div class="kpi-sub">{'need review' if dups > 0 else 'clean'}</div>
        </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)

    # EDA Narrative
    if eda and eda.summary_narrative:
        st.info(f"**📝 Auto-Summary:** {eda.summary_narrative}")

    # Data Preview
    with st.expander("🔎 Preview Raw Data (first 15 rows)", expanded=True):
        st.dataframe(df.head(15), use_container_width=True, height=360)

    # Numeric Distributions
    if numeric_profiles:
        st.markdown('<div class="section-header">📈 Numeric Column Profiles</div>', unsafe_allow_html=True)
        cols_per_row = 3
        for i in range(0, len(numeric_profiles), cols_per_row):
            chunk = numeric_profiles[i : i + cols_per_row]
            row_cols = st.columns(len(chunk))
            for col_widget, profile in zip(row_cols, chunk):
                with col_widget:
                    fig = go.Figure(go.Indicator(
                        mode="number+delta+gauge",
                        value=profile.mean,
                        number={"suffix": "", "font": {"size": 22}},
                        delta={"reference": profile.median, "relative": False, "position": "bottom",
                               "valueformat": ".2f"},
                        gauge={
                            "axis": {"range": [profile.min, profile.max], "tickfont": {"size": 9}},
                            "bar": {"color": "#6366F1"},
                            "bgcolor": "#F8FAFF",
                            "borderwidth": 0,
                            "steps": [
                                {"range": [profile.min, profile.q1], "color": "#EEF2FF"},
                                {"range": [profile.q1, profile.q3], "color": "#C7D2FE"},
                            ],
                        },
                        title={"text": f"<b>{profile.column}</b><br><span style='font-size:10px'>mean vs median</span>",
                               "font": {"size": 13}},
                    ))
                    fig.update_layout(
                        height=200,
                        margin=dict(l=10, r=10, t=40, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"gauge_{profile.column}")
                    if profile.outlier_count > 0:
                        st.caption(f"⚠️ {profile.outlier_count} outlier(s) detected")

    # Categorical Top Values
    if eda and eda.categorical_profiles:
        st.markdown('<div class="section-header">🏷️ Categorical Distributions</div>', unsafe_allow_html=True)
        cat_cols = st.columns(min(len(eda.categorical_profiles), 3))
        for idx, cat_p in enumerate(eda.categorical_profiles[:3]):
            with cat_cols[idx]:
                if cat_p.top_categories:
                    cat_df = pd.DataFrame(cat_p.top_categories)
                    import plotly.express as px
                    fig = px.bar(
                        cat_df.head(8),
                        x="count", y="value", orientation="h",
                        title=f"{cat_p.column} ({cat_p.cardinality} unique)",
                        color_discrete_sequence=["#6366F1"],
                        template="none",
                    )
                    fig.update_layout(
                        height=250,
                        margin=dict(l=8, r=8, t=36, b=8),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        yaxis=dict(title="", autorange="reversed"),
                        xaxis=dict(title="count", showgrid=True, gridcolor="#F1F5F9"),
                        font=dict(size=11),
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"cat_{cat_p.column}")

    # Datetime Range
    if eda and eda.datetime_profiles:
        st.markdown('<div class="section-header">🗓️ Time Dimensions</div>', unsafe_allow_html=True)
        for dt_p in eda.datetime_profiles:
            st.markdown(f"""
            <div class="result-card">
                <strong>{dt_p.column}</strong> &nbsp;
                <span class="badge-type">{dt_p.detected_frequency}</span>
                &nbsp; {dt_p.min_date} → {dt_p.max_date}
                &nbsp; <span style="color:#64748B;">({dt_p.date_range_days} days)</span>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# TAB 2 — Query Workspace
# ============================================================
with tab_query:
    st.markdown('<div class="section-header">🤖 Ask a Question About Your Data</div>', unsafe_allow_html=True)

    # Example prompts
    examples = [
        "What is the total revenue by category?",
        "Show me the distribution of sales amounts",
        "Which product had the highest revenue in 2024?",
        "Show monthly trend of sales",
        "What is the correlation between revenue and units sold?",
    ]
    st.markdown("**💡 Example questions:**")
    ex_cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        with ex_cols[i]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state["current_question"] = ex

    # Query input
    question = st.text_input(
        "Ask a question",
        placeholder="e.g. What was our best-selling product by revenue?",
        key="query_input",
        label_visibility="collapsed",
        value=st.session_state.get("current_question", ""),
    )
    if "current_question" in st.session_state:
        del st.session_state["current_question"]

    col_run, col_clear = st.columns([4, 1])
    with col_run:
        run_btn = st.button("▶ Run Analysis", type="primary", use_container_width=True, key="btn_run")
    with col_clear:
        if st.button("Clear History", use_container_width=True, key="btn_clear_hist"):
            st.session_state.query_history = []
            st.rerun()

    if run_btn and question.strip():
        with st.spinner("🤖 Generating analysis…"):
            qr = run_query(question, df, types)

        if qr.error:
            st.error(f"Query error: {qr.error}")
        else:
            if qr.is_mock:
                sandbox_result = execute_in_sandbox(qr.generated_code, df, timeout_seconds=10)
            else:
                sandbox_result = execute_in_sandbox(qr.generated_code, df, timeout_seconds=10)

            entry = {
                "question": question,
                "answer": qr.natural_language_answer,
                "code": qr.generated_code,
                "result_json": sandbox_result.result_json if sandbox_result.success else None,
                "result_type": sandbox_result.result_type if sandbox_result.success else "error",
                "time_ms": sandbox_result.execution_time_ms,
                "chart_intent": qr.chart_intent,
                "error": sandbox_result.error_message if not sandbox_result.success else None,
            }
            st.session_state.query_history.insert(0, entry)

    # Render history
    if st.session_state.query_history:
        st.markdown("---")
        st.markdown('<div class="section-header">📜 Query History</div>', unsafe_allow_html=True)
        for idx, entry in enumerate(st.session_state.query_history):
            with st.container():
                st.markdown(f"""
                <div class="result-question">❓ {entry['question']}</div>
                <div class="result-answer">{entry['answer']}</div>
                """, unsafe_allow_html=True)

                if entry["result_type"] == "chart" and entry["result_json"]:
                    try:
                        fig_dict = json.loads(entry["result_json"])
                        import plotly.io as pio
                        fig = pio.from_json(entry["result_json"])
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=0, r=0, t=40, b=0),
                            height=420,
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"hist_chart_{idx}")
                    except Exception as e:
                        st.warning(f"Could not render chart: {e}")

                elif entry["result_type"] == "table" and entry["result_json"]:
                    try:
                        table_df = pd.read_json(entry["result_json"], orient="records")
                        st.dataframe(table_df, use_container_width=True, height=300, key=f"hist_table_{idx}")
                    except Exception as e:
                        st.warning(f"Could not render table: {e}")

                elif entry["result_type"] == "scalar" and entry["result_json"]:
                    st.metric("Result", entry["result_json"])

                elif entry["result_type"] == "error":
                    st.error(f"Execution error: {entry.get('error', 'Unknown')}")

                if entry["code"]:
                    with st.expander("🔍 View Generated Code", expanded=False):
                        st.code(entry["code"], language="python")
                        st.caption(f"⏱ Executed in {entry['time_ms']}ms")

                st.markdown("<hr style='margin: 0.5rem 0; border-color: #F1F5F9;'>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 1rem; color: #94A3B8;">
            <div style="font-size:3rem">🤖</div>
            <p>Ask a question above to analyze your data with natural language.</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# TAB 3 — Data Health
# ============================================================
with tab_health:
    if not health:
        st.info("Load a dataset first.")
        st.stop()

    st.markdown('<div class="section-header">🩺 Data Health Report</div>', unsafe_allow_html=True)

    # Summary badges
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        badge = "badge-error" if health.duplicate_rows > 0 else "badge-ok"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Duplicate Rows</div>
            <div class="kpi-value">{health.duplicate_rows}</div>
            <div><span class="{badge}">{'needs cleanup' if health.duplicate_rows > 0 else 'clean'}</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        badge = "badge-warn" if health.missing_percentage > 5 else "badge-ok"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Missing Data</div>
            <div class="kpi-value">{health.missing_percentage:.1f}%</div>
            <div><span class="{badge}">{'review recommended' if health.missing_percentage > 5 else 'acceptable'}</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Anomalies</div>
            <div class="kpi-value">{len(health.anomalies)}</div>
            <div><span class="{'badge-warn' if health.anomalies else 'badge-ok'}">
            {'issues found' if health.anomalies else 'no issues'}</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Anomaly alerts
    if health.anomalies:
        for anomaly in health.anomalies:
            st.warning(f"⚠️ {anomaly}")

    # Recommendations
    if health.cleaning_recommendations:
        st.markdown('<div class="section-header">💡 Cleaning Recommendations</div>', unsafe_allow_html=True)
        for rec in health.cleaning_recommendations:
            st.markdown(f"→ {rec}")

    # Column Health Table
    st.markdown('<div class="section-header">📋 Column-Level Health</div>', unsafe_allow_html=True)

    health_rows = []
    for col_name, col_h in health.columns_health.items():
        warnings_str = " | ".join(col_h.warnings) if col_h.warnings else "✅ OK"
        health_rows.append({
            "Column": col_name,
            "Type": col_h.inferred_type,
            "Missing": f"{col_h.missing_count} ({col_h.missing_pct:.1f}%)",
            "Unique": col_h.unique_count,
            "Sample Values": ", ".join(str(v) for v in col_h.sample_values[:3]),
            "Warnings": warnings_str,
        })
    health_df = pd.DataFrame(health_rows)
    st.dataframe(health_df, use_container_width=True, height=450)

    # Apply cleaning
    st.markdown('<div class="section-header">🧹 Apply Cleaning Operations</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        do_dedup = st.checkbox("Remove duplicates", value=True, key="chk_dedup")
    with c2:
        do_ws = st.checkbox("Trim whitespace", value=True, key="chk_ws")
    with c3:
        do_null = st.checkbox("Normalize null tokens", value=True, key="chk_null")
    with c4:
        do_num = st.checkbox("Coerce numeric strings", value=True, key="chk_num")

    if st.button("🧹 Apply Cleaning", type="primary", key="btn_clean"):
        with st.spinner("Cleaning dataset…"):
            cleaned, log = clean_dataset(
                df,
                remove_duplicates=do_dedup,
                strip_whitespace=do_ws,
                normalize_nulls=do_null,
                coerce_numeric=do_num,
            )
        st.session_state.cleaned_df = cleaned
        st.session_state.clean_log = log
        if log:
            for change in log:
                st.success(f"✅ {change}")
        else:
            st.info("No changes were necessary — dataset is already clean.")

    if st.session_state.cleaned_df is not None and st.button(
        "🔄 Replace Active Dataset with Cleaned Version",
        key="btn_apply_clean",
        type="secondary",
    ):
        with st.spinner("Re-profiling…"):
            _load_file_from_df(st.session_state.cleaned_df, st.session_state.filename)
        st.success("Active dataset replaced with cleaned version.")
        st.rerun()


def _load_file_from_df(df_in: pd.DataFrame, name: str):
    types_ = detect_column_types(df_in)
    health_ = audit_data_health(df_in)
    eda_ = generate_eda_report(df_in, types_)
    st.session_state.dataset = df_in
    st.session_state.filename = name
    st.session_state.inferred_types = types_
    st.session_state.health_report = health_
    st.session_state.eda_report = eda_
