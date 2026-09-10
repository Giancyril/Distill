"""
app/main.py — Threadline Data Lab: Autonomous Tabular Intelligence Dashboard.
State-of-the-art UI inspired by Threadline design system:
Crisp white aesthetic, Plus Jakarta Sans typography, AST Sandboxed execution,
and conversational natural language query workspace.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.cleaning import audit_data_health, clean_dataset, detect_column_types
from core.eda import generate_eda_report
from core.ingestion import IngestionError, load_dataset
from core.query_engine import run_query
from core.sandbox import execute_in_sandbox

st.set_page_config(
    page_title="Threadline Data Lab — AI Data Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com",
        "About": "# Threadline Data Lab\nAutonomous AI-augmented data analytics & sandboxed execution.",
    },
)
THREADLINE_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
    background-color: #F8FAFC !important;
    color: #0F172A !important;
}

code, pre, .font-mono {
    font-family: 'JetBrains Mono', monospace !important;
}

.main .block-container {
    padding-top: 1rem !important;
    padding-bottom: 3rem !important;
    max-width: 1440px !important;
}

.threadline-header {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 1rem;
    padding: 0.875rem 1.5rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}
.header-left {
    display: flex;
    align-items: center;
    gap: 0.875rem;
}
.brand-icon-box {
    width: 2.5rem;
    height: 2.5rem;
    border-radius: 0.75rem;
    background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #FFFFFF;
    font-size: 1.25rem;
    font-weight: 800;
    box-shadow: 0 4px 10px rgba(79, 70, 229, 0.25);
}
.brand-title {
    font-size: 1.1rem;
    font-weight: 800;
    color: #0F172A;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.2;
}
.brand-subtitle {
    font-size: 0.75rem;
    color: #64748B;
    font-weight: 500;
    margin: 0;
}
.header-right {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 9999px;
    padding: 0.35rem 0.875rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: #334155;
}
.pulse-dot {
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 9999px;
    background: #10B981;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2);
}

[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
    padding-left: 1.25rem !important;
    padding-right: 1.25rem !important;
}
.sidebar-brand-card {
    background: #FFFFFF;
    border: 1px solid #F1F5F9;
    border-radius: 0.875rem;
    padding: 1rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.sidebar-section-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94A3B8;
    margin-top: 1.25rem;
    margin-bottom: 0.5rem;
}
.kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.kpi-card-threadline {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 1rem;
    padding: 1.25rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    position: relative;
    overflow: hidden;
}
.kpi-card-threadline:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px -2px rgba(15, 23, 42, 0.08);
    border-color: #CBD5E1;
}
.kpi-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.5rem;
}
.kpi-icon-pill {
    width: 2.25rem;
    height: 2.25rem;
    border-radius: 0.625rem;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
}
.icon-indigo { background: #EEF2FF; color: #4F46E5; }
.icon-emerald { background: #ECFDF5; color: #059669; }
.icon-violet { background: #F5F3FF; color: #7C3AED; }
.icon-amber { background: #FFFBEB; color: #D97706; }
.icon-rose { background: #FFF1F2; color: #E11D48; }

.kpi-badge {
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.2rem 0.5rem;
    border-radius: 9999px;
}
.badge-indigo { background: #EEF2FF; color: #4338CA; border: 1px solid #E0E7FF; }
.badge-emerald { background: #ECFDF5; color: #047857; border: 1px solid #D1FAE5; }
.badge-amber { background: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }
.badge-rose { background: #FFF1F2; color: #BE123C; border: 1px solid #FECDD3; }
.badge-slate { background: #F1F5F9; color: #475569; border: 1px solid #E2E8F0; }

.kpi-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
}
.kpi-val {
    font-size: 1.75rem;
    font-weight: 800;
    color: #0F172A;
    line-height: 1.2;
    letter-spacing: -0.02em;
}
.kpi-subtext {
    font-size: 0.75rem;
    font-weight: 500;
    color: #94A3B8;
    margin-top: 0.25rem;
}

.threadline-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 1rem;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}
.threadline-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #F1F5F9;
}
.card-header-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #0F172A;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
    margin-top: 1rem;
}
.chat-turn {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}
.chat-user-bubble {
    align-self: flex-end;
    background: #4F46E5;
    color: #FFFFFF;
    padding: 0.875rem 1.25rem;
    border-radius: 1.25rem 1.25rem 0.25rem 1.25rem;
    max-width: 80%;
    font-size: 0.9rem;
    font-weight: 500;
    line-height: 1.5;
    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.2);
}
.chat-assistant-bubble {
    align-self: flex-start;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 1.25rem 1.25rem 1.25rem 0.25rem;
    padding: 1.25rem;
    width: 100%;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
}
.chat-meta-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    background: #EEF2FF;
    border: 1px solid #E0E7FF;
    color: #4338CA;
    font-size: 0.72rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
}
.chat-answer-text {
    font-size: 0.92rem;
    color: #1E293B;
    line-height: 1.6;
    margin-bottom: 1rem;
}

.code-box {
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 0.75rem;
    padding: 1rem;
    color: #E2E8F0;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem;
    line-height: 1.6;
    overflow-x: auto;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.35rem;
    background: #F1F5F9 !important;
    border: 1px solid #E2E8F0;
    border-radius: 0.875rem !important;
    padding: 0.3rem !important;
    margin-bottom: 1.25rem !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 0.625rem !important;
    padding: 0.5rem 1.25rem !important;
    font-size: 0.825rem !important;
    font-weight: 600 !important;
    color: #64748B !important;
    border: none !important;
    transition: all 0.15s ease !important;
}
.stTabs [aria-selected="true"] {
    background: #FFFFFF !important;
    color: #4F46E5 !important;
    font-weight: 700 !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08) !important;
}

.stButton > button[kind="primary"] {
    background: #4F46E5 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 0.625rem !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px rgba(79, 70, 229, 0.3) !important;
    transition: background 0.15s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background: #4338CA !important;
}
.stButton > button[kind="secondary"] {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    color: #334155 !important;
    border-radius: 0.625rem !important;
    font-weight: 600 !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #F8FAFC !important;
    border-color: #94A3B8 !important;
    color: #0F172A !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #F8FAFC; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 9999px; }
::-webkit-scrollbar-thumb:hover { background: #94A3B8; }
</style>
"""
st.markdown(THREADLINE_CSS, unsafe_allow_html=True)
def _init_session():
    defaults = {
        "dataset": None,
        "filename": None,
        "health_report": None,
        "eda_report": None,
        "inferred_types": None,
        "cleaned_df": None,
        "clean_log": [],
        "query_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()

def _load_data_source(source, name: Optional[str] = None):
    try:
        df = load_dataset(source)
    except IngestionError as e:
        st.error(f"Ingestion Error: {e}")
        return

    detected_name = name or (source if isinstance(source, str) else getattr(source, "name", "dataset.csv"))
    types = detect_column_types(df)
    health = audit_data_health(df)
    eda = generate_eda_report(df, types)

    st.session_state.dataset = df
    st.session_state.filename = os.path.basename(detected_name)
    st.session_state.inferred_types = types
    st.session_state.health_report = health
    st.session_state.eda_report = eda
    st.session_state.cleaned_df = None
    st.session_state.clean_log = []

has_dataset = st.session_state.dataset is not None
active_file = st.session_state.filename or "No dataset loaded"
row_count_str = f"{len(st.session_state.dataset):,} records" if has_dataset else "Awaiting CSV/Excel"
openai_ready = bool(os.getenv("OPENAI_API_KEY"))

header_html = f"""
<div class="threadline-header">
    <div class="header-left">
        <div class="brand-icon-box">⚡</div>
        <div>
            <h1 class="brand-title">Threadline <span style="color:#4F46E5;">Data Lab</span></h1>
            <p class="brand-subtitle">Autonomous Tabular Analytics • AST Sandboxed Intelligence</p>
        </div>
    </div>
    <div class="header-right">
        <div class="status-pill">
            <span class="pulse-dot"></span>
            <span>{'AST Sandbox Active' if has_dataset else 'System Idle'}</span>
        </div>
        <div class="status-pill" style="background:#FFFFFF;">
            <span style="color:#64748B; font-weight:500;">Active:</span>
            <strong style="color:#0F172A;">{active_file}</strong>
            <span style="color:#94A3B8;">({row_count_str})</span>
        </div>
        <div class="status-pill" style="{'background:#ECFDF5; color:#065F46; border-color:#D1FAE5;' if openai_ready else 'background:#F8FAFC; color:#64748B;'}">
            <span>{'● OpenAI GPT-4o Online' if openai_ready else '○ Offline Sandbox Fallback'}</span>
        </div>
    </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand-card">
        <div style="width:2rem; height:2rem; border-radius:0.5rem; background:#4F46E5; display:flex; align-items:center; justify-content:center; color:#fff; font-size:1rem; font-weight:bold;">
            T
        </div>
        <div>
            <div style="font-size:0.875rem; font-weight:700; color:#0F172A;">Threadline Data</div>
            <div style="font-size:0.7rem; color:#64748B; font-weight:500;">Enterprise Edition v1.4</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Ingest Dataset</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
        help="Max 50MB file size safeguard",
    )
    if uploaded and st.session_state.filename != uploaded.name:
        with st.spinner("Analyzing and profiling dataset..."):
            _load_data_source(uploaded, uploaded.name)
        st.rerun()

    st.markdown('<div class="sidebar-section-title">Benchmark Samples</div>', unsafe_allow_html=True)
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        if st.button("📈 Retail Sales", use_container_width=True, key="btn_sample_sales"):
            with st.spinner("Loading sales..."):
                _load_data_source("sample_data/sales_clean.csv", "sales_clean.csv")
            st.rerun()
    with b_col2:
        if st.button("👥 Churn Messy", use_container_width=True, key="btn_sample_churn"):
            with st.spinner("Loading churn..."):
                _load_data_source("sample_data/customer_churn_messy.csv", "customer_churn_messy.csv")
            st.rerun()

    if has_dataset:
        st.markdown('<div class="sidebar-section-title">Active Dataset Meta</div>', unsafe_allow_html=True)
        eda_side = st.session_state.eda_report
        health_side = st.session_state.health_report
        dup_count = health_side.duplicate_rows if health_side else 0
        quality_badge = "badge-emerald" if dup_count == 0 else "badge-amber"
        quality_text = "Clean" if dup_count == 0 else f"{dup_count} Dups"

        st.markdown(f"""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:0.75rem; padding:0.875rem; margin-bottom:1rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                <span style="font-size:0.75rem; font-weight:700; color:#0F172A;">{st.session_state.filename}</span>
                <span class="kpi-badge {quality_badge}">{quality_text}</span>
            </div>
            <div style="font-size:0.75rem; color:#64748B;">
                <strong>{eda_side.total_rows:,}</strong> rows × <strong>{eda_side.total_columns}</strong> cols
            </div>
            <div style="font-size:0.7rem; color:#94A3B8; margin-top:0.25rem;">
                Missing: {health_side.missing_percentage:.1f}% • Profiles: {len(eda_side.numeric_profiles)} num
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section-title">OpenAI Engine Config</div>', unsafe_allow_html=True)
        if openai_ready:
            st.markdown("""
            <div style="font-size:0.75rem; color:#059669; background:#ECFDF5; border:1px solid #D1FAE5; padding:0.5rem 0.75rem; border-radius:0.5rem; font-weight:600;">
                ✓ OpenAI API Key Connected
            </div>
            """, unsafe_allow_html=True)
        else:
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-proj-...",
                key="input_api_key",
                help="Stored only in local session memory.",
            )
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("↺ Reset Session", use_container_width=True, type="secondary", key="btn_reset_all"):
            for k in ["dataset", "filename", "health_report", "eda_report",
                      "inferred_types", "cleaned_df", "clean_log", "query_history"]:
                st.session_state[k] = None if k not in ["clean_log", "query_history"] else []
            st.rerun()

if not has_dataset:
    st.markdown("""
    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:1.25rem; padding:4rem 2rem; text-align:center; max-width:680px; margin:2rem auto; box-shadow:0 1px 3px rgba(15,23,42,0.04);">
        <div style="width:4rem; height:4rem; border-radius:1rem; background:#EEF2FF; color:#4F46E5; display:inline-flex; align-items:center; justify-content:center; font-size:2rem; margin-bottom:1.25rem;">
            📊
        </div>
        <h2 style="font-size:1.35rem; font-weight:800; color:#0F172A; margin-bottom:0.5rem;">Upload a Dataset to Begin</h2>
        <p style="font-size:0.875rem; color:#64748B; max-width:440px; margin:0 auto 1.75rem; line-height:1.6;">
            Drop your CSV or Excel file into the sidebar or click a benchmark sample to activate the statistical profiling and conversational analysis lab.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("""
        <div class="threadline-card" style="text-align:center;">
            <div style="font-size:1.5rem; margin-bottom:0.5rem;">📈</div>
            <strong style="color:#0F172A;">Sample 1: Retail Sales</strong>
            <p style="font-size:0.78rem; color:#64748B; margin-top:0.25rem;">Clean e-commerce dataset with 1,000 transactions, categories, and profits.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Load Retail Sales Sample", use_container_width=True, type="primary", key="btn_empty_sales"):
            _load_data_source("sample_data/sales_clean.csv", "sales_clean.csv")
            st.rerun()
    with col_c2:
        st.markdown("""
        <div class="threadline-card" style="text-align:center;">
            <div style="font-size:1.5rem; margin-bottom:0.5rem;">👥</div>
            <strong style="color:#0F172A;">Sample 2: Customer Churn</strong>
            <p style="font-size:0.78rem; color:#64748B; margin-top:0.25rem;">Messy real-world customer records with missing tokens, whitespace, and duplicates.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Load Churn Sample", use_container_width=True, type="secondary", key="btn_empty_churn"):
            _load_data_source("sample_data/customer_churn_messy.csv", "customer_churn_messy.csv")
            st.rerun()
    st.stop()
df = st.session_state.dataset
health = st.session_state.health_report
eda = st.session_state.eda_report
types = st.session_state.inferred_types

tab_overview, tab_chat, tab_health_page, tab_explorer = st.tabs([
    "📊 Overview Dashboard",
    "💬 Conversational Workspace",
    "🛡️ Data Health & Diagnostics",
    "🔍 Raw Data Explorer",
])

with tab_overview:
    missing_pct = health.missing_percentage if health else 0.0
    dup_rows = health.duplicate_rows if health else 0
    num_profs = len(eda.numeric_profiles) if eda else 0
    cat_profs = len(eda.categorical_profiles) if eda else 0
    quality_score = max(0, int(100 - (missing_pct * 1.5) - (dup_rows * 2)))

    kpi_html = f"""
    <div class="kpi-row">
        <div class="kpi-card-threadline">
            <div class="kpi-top">
                <div class="kpi-icon-pill icon-indigo">🗂️</div>
                <span class="kpi-badge badge-indigo">100% Parsed</span>
            </div>
            <div class="kpi-label">Total Records</div>
            <div class="kpi-val">{len(df):,}</div>
            <div class="kpi-subtext">Rows loaded in memory</div>
        </div>

        <div class="kpi-card-threadline">
            <div class="kpi-top">
                <div class="kpi-icon-pill icon-violet">📊</div>
                <span class="kpi-badge badge-slate">{num_profs} Num • {cat_profs} Cat</span>
            </div>
            <div class="kpi-label">Total Columns</div>
            <div class="kpi-val">{len(df.columns)}</div>
            <div class="kpi-subtext">Distinct features inferred</div>
        </div>

        <div class="kpi-card-threadline">
            <div class="kpi-top">
                <div class="kpi-icon-pill {'icon-emerald' if missing_pct < 5 else 'icon-amber'}">⚠️</div>
                <span class="kpi-badge {'badge-emerald' if missing_pct < 5 else 'badge-amber'}">
                    {'Optimal' if missing_pct < 5 else 'Missing Found'}
                </span>
            </div>
            <div class="kpi-label">Missing Cells</div>
            <div class="kpi-val">{health.missing_cells if health else 0:,}</div>
            <div class="kpi-subtext">{missing_pct:.1f}% overall missingness</div>
        </div>

        <div class="kpi-card-threadline">
            <div class="kpi-top">
                <div class="kpi-icon-pill {'icon-emerald' if dup_rows == 0 else 'icon-rose'}">👥</div>
                <span class="kpi-badge {'badge-emerald' if dup_rows == 0 else 'badge-rose'}">
                    {'Zero Dups' if dup_rows == 0 else f'{dup_rows} Dups'}
                </span>
            </div>
            <div class="kpi-label">Duplicate Rows</div>
            <div class="kpi-val">{dup_rows}</div>
            <div class="kpi-subtext">{'No redundancy detected' if dup_rows == 0 else 'Deduplication ready'}</div>
        </div>

        <div class="kpi-card-threadline">
            <div class="kpi-top">
                <div class="kpi-icon-pill icon-emerald">✨</div>
                <span class="kpi-badge badge-emerald">Data Health</span>
            </div>
            <div class="kpi-label">Quality Index</div>
            <div class="kpi-val">{quality_score}/100</div>
            <div class="kpi-subtext">Composite health score</div>
        </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)

    if eda and eda.summary_narrative:
        st.markdown(f"""
        <div class="threadline-card" style="border-left: 4px solid #4F46E5;">
            <div class="threadline-card-header">
                <span class="card-header-title">
                    <span>⚡</span> AI Executive Synthesis & Distribution Insights
                </span>
                <span class="kpi-badge badge-indigo">Automated Profiling</span>
            </div>
            <p style="font-size:0.92rem; color:#334155; line-height:1.65; margin:0;">
                {eda.summary_narrative}
            </p>
        </div>
        """, unsafe_allow_html=True)

    if eda and eda.numeric_profiles:
        st.markdown('<div class="card-header-title" style="margin: 1.5rem 0 1rem;">📈 Numeric Dimension Benchmarks</div>', unsafe_allow_html=True)
        chunk_size = 3
        for i in range(0, len(eda.numeric_profiles), chunk_size):
            chunk = eda.numeric_profiles[i : i + chunk_size]
            row_cols = st.columns(len(chunk))
            for idx, prof in enumerate(chunk):
                with row_cols[idx]:
                    fig = go.Figure(go.Indicator(
                        mode="number+delta+gauge",
                        value=prof.mean,
                        number={"font": {"size": 22, "family": "Plus Jakarta Sans", "color": "#0F172A"}},
                        delta={"reference": prof.median, "relative": False, "position": "bottom",
                               "valueformat": ".2f"},
                        gauge={
                            "axis": {"range": [prof.min, prof.max], "tickfont": {"size": 9, "color": "#64748B"}},
                            "bar": {"color": "#4F46E5"},
                            "bgcolor": "#F8FAFC",
                            "borderwidth": 0,
                            "steps": [
                                {"range": [prof.min, prof.q1], "color": "#EEF2FF"},
                                {"range": [prof.q1, prof.q3], "color": "#E0E7FF"},
                            ],
                        },
                        title={"text": f"<b>{prof.column}</b><br><span style='font-size:10px; color:#64748B;'>Mean vs Median</span>",
                               "font": {"size": 13, "family": "Plus Jakarta Sans"}},
                    ))
                    fig.update_layout(
                        height=210,
                        margin=dict(l=15, r=15, t=45, b=15),
                        paper_bgcolor="#FFFFFF",
                        plot_bgcolor="#FFFFFF",
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"gauge_{prof.column}")
                    if prof.outlier_count > 0:
                        st.caption(f"⚠️ **{prof.outlier_count}** statistical outlier(s) detected via IQR.")

    if eda and eda.categorical_profiles:
        st.markdown('<div class="card-header-title" style="margin: 1.5rem 0 1rem;">🏷️ Categorical Distributions & Cardinality</div>', unsafe_allow_html=True)
        cat_cols = st.columns(min(len(eda.categorical_profiles), 3))
        for idx, cat_p in enumerate(eda.categorical_profiles[:3]):
            with cat_cols[idx]:
                if cat_p.top_categories:
                    cat_df = pd.DataFrame(cat_p.top_categories)
                    fig = px.bar(
                        cat_df.head(7),
                        x="count", y="value", orientation="h",
                        title=f"{cat_p.column} ({cat_p.cardinality} unique)",
                        color_discrete_sequence=["#4F46E5"],
                        template="none",
                    )
                    fig.update_layout(
                        height=260,
                        margin=dict(l=8, r=8, t=36, b=8),
                        paper_bgcolor="#FFFFFF",
                        plot_bgcolor="#FFFFFF",
                        yaxis=dict(title="", autorange="reversed", tickfont=dict(color="#334155", size=10)),
                        xaxis=dict(title="Count", showgrid=True, gridcolor="#F1F5F9", tickfont=dict(color="#64748B")),
                        font=dict(family="Plus Jakarta Sans", size=11),
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"cat_{cat_p.column}")

    if eda and eda.datetime_profiles:
        st.markdown('<div class="card-header-title" style="margin: 1.5rem 0 1rem;">📅 Temporal Dimensions</div>', unsafe_allow_html=True)
        for dt_p in eda.datetime_profiles:
            st.markdown(f"""
            <div class="threadline-card" style="display:flex; justify-content:space-between; align-items:center; padding:1rem 1.25rem;">
                <div>
                    <strong style="color:#0F172A; font-size:0.9rem;">{dt_p.column}</strong>
                    <span style="color:#64748B; font-size:0.75rem; margin-left:0.5rem;">Detected Frequency: <strong>{dt_p.detected_frequency}</strong></span>
                </div>
                <div style="font-size:0.8rem; color:#475569;">
                    <span class="kpi-badge badge-indigo">{dt_p.min_date} → {dt_p.max_date}</span>
                    <span style="color:#94A3B8; margin-left:0.5rem;">({dt_p.date_range_days} days)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
with tab_chat:
    st.markdown("""
    <div class="threadline-card">
        <div class="threadline-card-header">
            <span class="card-header-title">
                <span>💬</span> Natural Language Analysis Assistant
            </span>
            <span class="kpi-badge badge-indigo">Zero Data Leakage • Schema-Only Prompts</span>
        </div>
        <p style="font-size:0.875rem; color:#64748B; margin:0;">
            Ask any analytical question in plain English. The query engine generates sandboxed Python Pandas & Plotly code and executes it within an isolated subprocess in under 5 seconds.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<span style='font-size:0.75rem; font-weight:700; color:#64748B; text-transform:uppercase;'>Suggested Prompts:</span>", unsafe_allow_html=True)
    examples = [
        "What is the total revenue by category?",
        "Show monthly sales trend",
        "Which product generated the highest profit?",
        "Show distribution of order quantities",
        "What is the correlation between revenue and quantity?",
    ]
    pill_cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        with pill_cols[i]:
            if st.button(ex, key=f"pill_{i}", use_container_width=True):
                st.session_state["pending_question"] = ex

    col_input, col_run, col_clear = st.columns([6, 1.5, 1])
    with col_input:
        user_input = st.text_input(
            "Ask a question about the dataset",
            value=st.session_state.get("pending_question", ""),
            placeholder="e.g. Compare total sales by region across quarters...",
            key="main_query_input",
            label_visibility="collapsed",
        )
    with col_run:
        execute_click = st.button("⚡ Run Analysis", type="primary", use_container_width=True, key="btn_exec_query")
    with col_clear:
        if st.button("Clear History", type="secondary", use_container_width=True, key="btn_clear_thread"):
            st.session_state.query_history = []
            st.rerun()

    if "pending_question" in st.session_state:
        del st.session_state["pending_question"]

    if execute_click and user_input.strip():
        with st.spinner("Generating analytical plan & sandboxed execution..."):
            query_res = run_query(user_input.strip(), df, types)
            if query_res.error:
                st.error(f"Query Engine Error: {query_res.error}")
            else:
                sandbox_res = execute_in_sandbox(query_res.generated_code, df, timeout_seconds=10)
                history_record = {
                    "question": user_input.strip(),
                    "answer": query_res.natural_language_answer,
                    "code": query_res.generated_code,
                    "result_json": sandbox_res.result_json if sandbox_res.success else None,
                    "result_type": sandbox_res.result_type if sandbox_res.success else "error",
                    "error": sandbox_res.error_message if not sandbox_res.success else None,
                    "time_ms": sandbox_res.execution_time_ms,
                    "chart_intent": query_res.chart_intent,
                    "is_mock": query_res.is_mock,
                }
                st.session_state.query_history.insert(0, history_record)

    if st.session_state.query_history:
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        for idx, item in enumerate(st.session_state.query_history):
            st.markdown(f"""
            <div class="chat-turn">
                <div class="chat-user-bubble">
                    {item['question']}
                </div>
                <div class="chat-assistant-bubble">
                    <div class="chat-meta-pill">
                        ⚡ In-Memory AST Sandbox • {item['time_ms']}ms execution
                        {'• Offline Mock Mode' if item.get('is_mock') else '• GPT-4o Synthesis'}
                    </div>
                    <div class="chat-answer-text">
                        {item['answer']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if item["result_type"] == "chart" and item["result_json"]:
                try:
                    fig = pio.from_json(item["result_json"])
                    fig.update_layout(
                        paper_bgcolor="#FFFFFF",
                        plot_bgcolor="#FFFFFF",
                        margin=dict(l=10, r=10, t=35, b=10),
                        height=420,
                        font=dict(family="Plus Jakarta Sans"),
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"chat_fig_{idx}")
                except Exception as err:
                    st.warning(f"Chart render issue: {err}")

            elif item["result_type"] == "table" and item["result_json"]:
                try:
                    tbl = pd.read_json(item["result_json"], orient="records")
                    st.dataframe(tbl, use_container_width=True, height=280, key=f"chat_tbl_{idx}")
                except Exception as err:
                    st.warning(f"Table render issue: {err}")

            elif item["result_type"] == "scalar" and item["result_json"]:
                st.metric("Computed Value", item["result_json"])

            elif item["result_type"] == "error":
                st.error(f"Sandbox Runtime Error: {item.get('error', 'Execution interrupted')}")

            if item.get("code"):
                with st.expander("🔍 Inspect Sandboxed Python Code (Explainability)", expanded=False):
                    st.markdown(f'<div class="code-box">{item["code"]}</div>', unsafe_allow_html=True)
                    st.caption(f"Code verified via AST allowlist • Completed in {item['time_ms']}ms")

            st.markdown("<hr style='border:0; border-top:1px solid #E2E8F0; margin:1.5rem 0;'>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:1rem; padding:3rem 1.5rem; text-align:center; color:#94A3B8; margin-top:1.5rem;">
            <div style="font-size:2.5rem; margin-bottom:0.5rem;">💬</div>
            <strong style="color:#475569;">No queries in this session yet</strong>
            <p style="font-size:0.8rem; color:#94A3B8; margin-top:0.25rem;">Select a suggested prompt above or ask any question to inspect results.</p>
        </div>
        """, unsafe_allow_html=True)
with tab_health_page:
    st.markdown("""
    <div class="threadline-card">
        <div class="threadline-card-header">
            <span class="card-header-title">
                <span>🛡️</span> Data Health & Quality Assurance Audit
            </span>
            <span class="kpi-badge badge-emerald">Non-Destructive Operations</span>
        </div>
        <p style="font-size:0.875rem; color:#64748B; margin:0;">
            Comprehensive audit inspecting null token normalizations, trailing whitespaces, duplicate rows, and column type fidelity.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if health.anomalies:
        for anomaly in health.anomalies:
            st.warning(f"⚠️ **Anomaly Identified:** {anomaly}")

    if health.cleaning_recommendations:
        st.markdown("**Recommended Remediations:**")
        for rec in health.cleaning_recommendations:
            st.markdown(f"- {rec}")

    st.markdown('<div class="card-header-title" style="margin: 1.5rem 0 0.75rem;">📋 Column Health Breakdown</div>', unsafe_allow_html=True)
    c_rows = []
    for col_n, col_h in health.columns_health.items():
        warns = " | ".join(col_h.warnings) if col_h.warnings else "✓ Healthy"
        c_rows.append({
            "Column": col_n,
            "Detected Type": col_h.inferred_type,
            "Missing Cells": f"{col_h.missing_count} ({col_h.missing_pct:.1f}%)",
            "Unique Values": col_h.unique_count,
            "Sample Values": ", ".join(str(v) for v in col_h.sample_values[:3]),
            "Diagnostic Status": warns,
        })
    st.dataframe(pd.DataFrame(c_rows), use_container_width=True, height=360)

    st.markdown('<div class="card-header-title" style="margin: 1.5rem 0 0.75rem;">⚙️ 1-Click Non-Destructive Cleaning Pipeline</div>', unsafe_allow_html=True)
    cl_col1, cl_col2, cl_col3, cl_col4 = st.columns(4)
    with cl_col1:
        opt_dedup = st.checkbox("Remove duplicate rows", value=True, key="c_dedup")
    with cl_col2:
        opt_ws = st.checkbox("Strip whitespace", value=True, key="c_ws")
    with cl_col3:
        opt_null = st.checkbox("Normalize null tokens (NA/null)", value=True, key="c_null")
    with cl_col4:
        opt_num = st.checkbox("Coerce numeric text fields", value=True, key="c_num")

    if st.button("⚡ Clean & Remediate Dataset", type="primary", key="btn_clean_exec"):
        with st.spinner("Executing non-destructive cleaning pipeline..."):
            cleaned, log_actions = clean_dataset(
                df,
                remove_duplicates=opt_dedup,
                strip_whitespace=opt_ws,
                normalize_nulls=opt_null,
                coerce_numeric=opt_num,
            )
        st.session_state.cleaned_df = cleaned
        st.session_state.clean_log = log_actions
        if log_actions:
            for action in log_actions:
                st.success(f"✓ {action}")
        else:
            st.info("No modifications needed — dataset satisfies all quality thresholds.")

    if st.session_state.cleaned_df is not None:
        if st.button("💾 Replace Active Session Dataset with Cleaned Copy", type="secondary", key="btn_replace_cleaned"):
            _load_data_source(st.session_state.cleaned_df, f"Cleaned_{st.session_state.filename}")
            st.success("Active dataset updated to cleaned version.")
            st.rerun()

with tab_explorer:
    st.markdown(f"""
    <div class="threadline-card">
        <div class="threadline-card-header">
            <span class="card-header-title">
                <span>🔍</span> Interactive Raw Data Grid
            </span>
            <span class="kpi-badge badge-slate">Viewing {len(df):,} records</span>
        </div>
        <p style="font-size:0.875rem; color:#64748B; margin:0;">
            Filter, sort, and inspect tabular records. Export cleaned records directly to CSV.
        </p>
    </div>
    """, unsafe_allow_html=True)

    f_col1, f_col2 = st.columns([3, 1])
    with f_col1:
        col_selector = st.multiselect("Columns to display", options=list(df.columns), default=list(df.columns), key="grid_cols")
    with f_col2:
        row_limit = st.selectbox("Rows limit", [50, 100, 500, 1000, len(df)], index=1, key="grid_limit")

    filtered_df = df[col_selector].head(row_limit) if col_selector else df.head(row_limit)
    st.dataframe(filtered_df, use_container_width=True, height=450)

    csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Current View to CSV",
        data=csv_bytes,
        file_name=f"export_{st.session_state.filename}",
        mime="text/csv",
        type="secondary",
        key="btn_download_csv",
    )
