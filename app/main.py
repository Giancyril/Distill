"""
app/main.py \u2013 Distill Data Lab: Autonomous Tabular Intelligence Dashboard.
Refined UI: Pass 2 \u2014 depth, iconography, typography system.
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
    page_title="Distill Data Lab",
    page_icon="\u26a1",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com",
        "About": "# Distill Data Lab\nAutonomous AI-augmented data analytics & sandboxed execution.",
    },
)

# ---------------------------------------------------------------------------
# SVG ICON REGISTRY  (Lucide-style, stroke-width 2, all 18px unless noted)
# ---------------------------------------------------------------------------
def _svg(path_d: str, size: int = 18, color: str = "currentColor", extra_attrs: str = "") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'style="display:inline-block;vertical-align:middle;flex-shrink:0;" {extra_attrs}>'
        f'{path_d}</svg>'
    )

ICONS: dict = {
    "zap": _svg('<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>', 16, "#FFFFFF"),
    "zap_indigo": _svg('<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>', 14, "#4F46E5"),
    "trending_up": _svg(
        '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/>'
        '<polyline points="17 6 23 6 23 12"/>',
        18, "#4F46E5"
    ),
    "users": _svg(
        '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>'
        '<circle cx="9" cy="7" r="4"/>'
        '<path d="M23 21v-2a4 4 0 0 0-3-3.87"/>'
        '<path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
        18, "#0891B2"
    ),
    "upload_cloud": _svg(
        '<polyline points="16 16 12 12 8 16"/>'
        '<line x1="12" y1="12" x2="12" y2="21"/>'
        '<path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>',
        32, "#6366F1"
    ),
    "bar_chart_2": _svg(
        '<line x1="18" y1="20" x2="18" y2="10"/>'
        '<line x1="12" y1="20" x2="12" y2="4"/>'
        '<line x1="6" y1="20" x2="6" y2="14"/>',
        14, "#4F46E5"
    ),
    "folder": _svg('<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>', 13, "#4F46E5"),
    "columns": _svg('<rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="12" y1="3" x2="12" y2="21"/>', 13, "#7C3AED"),
    "alert_triangle": _svg('<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>', 13, "#D97706"),
    "copy": _svg('<rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>', 13, "#E11D48"),
    "star": _svg('<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>', 13, "#059669"),
    "message_square": _svg('<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>', 14, "#64748B"),
    "shield": _svg('<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>', 14, "#64748B"),
    "table": _svg('<rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/>', 14, "#64748B"),
    "check": _svg('<polyline points="20 6 9 17 4 12"/>', 14, "#059669"),
    "code_2": _svg('<path d="m18 16 4-4-4-4"/><path d="m6 8-4 4 4 4"/><path d="m14.5 4-5 16"/>', 13, "#4338CA"),
    "sparkles": _svg('<path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/><path d="M5 3v4"/><path d="M3 5h4"/><path d="M19 17v4"/><path d="M17 19h4"/>', 13, "#4338CA"),
    "download": _svg('<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>', 13, "#475569"),
}


DISTILL_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
/* === Base Typography \u2014 Explicit Inter === */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    background-color: #F8FAFC !important;
    color: #0F172A !important;
    font-size: 13px !important;
    line-height: 1.5 !important;
}
code, pre, .font-mono {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}
.main .block-container {
    padding-top: 0.75rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1400px !important;
}

/* === Header \u2014 lifted elevation === */
.distill-header {
    background: #FAFAFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 12px 20px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 4px 12px rgba(0,0,0,0.04);
}
.header-left { display: flex; align-items: center; gap: 12px; }
.brand-title {
    font-size: 26px; font-weight: 700; color: #0F172A;
    letter-spacing: -0.02em; margin: 0; line-height: 1.2;
    font-family: 'Inter', sans-serif;
}
/* 500-weight subtitle \u2014 bridges title and body weight */
.brand-subtitle {
    font-size: 13px; color: #64748B; font-weight: 500; margin: 0;
    font-family: 'Inter', sans-serif;
}
.header-right { display: flex; align-items: center; gap: 8px; }

/* === Status pills \u2014 tinted per state === */
.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    border: 1px solid transparent; border-radius: 9999px;
    padding: 4px 12px; font-size: 12px; font-weight: 500;
    height: 28px; white-space: nowrap;
}
.status-pill-neutral { background: #F4F4F6; border-color: #E4E4E7; color: #52525B; }
.status-pill-amber   { background: #FFF8EB; border-color: #FDE68A; color: #92400E; }
.status-pill-active  {
    background: #FFFFFF; border-color: #E2E8F0; color: #334155;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}
.status-dot { width: 7px; height: 7px; border-radius: 9999px; flex-shrink: 0; }
.dot-emerald { background: #10B981; box-shadow: 0 0 0 2px rgba(16,185,129,0.2); }
.dot-slate   { background: #94A3B8; }
.dot-indigo  { background: #4F46E5; }
.dot-amber   { background: #F59E0B; box-shadow: 0 0 0 2px rgba(245,158,11,0.2); }

/* === Sidebar === */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 16px !important; padding-left: 16px !important; padding-right: 16px !important;
}
/* Sidebar profile card \u2014 resting shadow so it lifts off bg */
.sidebar-brand-card {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px;
    padding: 12px; margin-bottom: 16px; display: flex; align-items: center; gap: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 2px 8px rgba(0,0,0,0.04);
}
.sidebar-section-title {
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.05em; color: #64748B; margin-top: 16px; margin-bottom: 8px;
}

/* === Buttons === */
.stButton > button {
    border-radius: 8px !important; font-size: 13px !important;
    font-weight: 500 !important; padding: 8px 16px !important;
    transition: all 0.15s ease-in-out !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button[kind="primary"] {
    background: #4F46E5 !important; color: #FFFFFF !important;
    border: 1px solid #4F46E5 !important;
    box-shadow: 0 1px 2px rgba(79,70,229,0.2) !important;
}
.stButton > button[kind="primary"]:hover {
    background: #4338CA !important; border-color: #4338CA !important;
    box-shadow: 0 2px 4px rgba(79,70,229,0.25) !important;
}
.stButton > button[kind="secondary"] {
    background: #FFFFFF !important; border: 1px solid #E2E8F0 !important;
    color: #334155 !important; box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #F8FAFC !important; border-color: #CBD5E1 !important; color: #0F172A !important;
}

/* === File uploader \u2014 recessed (slot you drop INTO) === */
[data-testid="stFileUploader"] { border-radius: 8px !important; }
[data-testid="stFileUploader"] section {
    border: 1px dashed #CBD5E1 !important; background: #F8FAFC !important;
    border-radius: 8px !important; padding: 12px !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.06) !important;
    transition: all 0.15s ease-in-out !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #4F46E5 !important; background: #FAFAFF !important;
}

/* === Content cards \u2014 real hover lift === */
.distill-card {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
    padding: 16px 20px; margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 1px 3px rgba(0,0,0,0.04);
    transition: box-shadow 0.15s ease, border-color 0.15s ease, transform 0.15s ease;
}
.distill-card:hover {
    border-color: #CBD5E1;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08), 0 1px 3px rgba(0,0,0,0.04);
    transform: translateY(-1px);
}
.card-header-bar {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #F1F5F9;
}
/* Card title: explicitly 600 weight, 15px */
.card-title-text {
    font-size: 15px; font-weight: 600; letter-spacing: -0.01em; color: #0F172A;
    display: flex; align-items: center; gap: 6px; font-family: 'Inter', sans-serif;
}
.card-overline {
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.05em; color: #64748B;
}

/* === KPI Cards === */
.kpi-row {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px; margin-bottom: 16px;
}
.kpi-card {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 1px 3px rgba(0,0,0,0.04);
    transition: all 0.15s ease-in-out;
}
.kpi-card:hover {
    border-color: #CBD5E1; box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    transform: translateY(-2px);
}
.kpi-top-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
.kpi-icon-pill { width: 28px; height: 28px; border-radius: 6px; display: flex; align-items: center; justify-content: center; }
.icon-indigo  { background: #EEF2FF; }
.icon-emerald { background: #ECFDF5; }
.icon-violet  { background: #F5F3FF; }
.icon-amber   { background: #FFFBEB; }
.icon-rose    { background: #FFF1F2; }
.icon-cyan    { background: #ECFEFF; }

.badge-pill { font-size: 11px; font-weight: 600; letter-spacing: 0.03em; padding: 2px 8px; border-radius: 9999px; }
.badge-indigo  { background: #EEF2FF; color: #4338CA; border: 1px solid #E0E7FF; }
.badge-emerald { background: #ECFDF5; color: #047857; border: 1px solid #D1FAE5; }
.badge-amber   { background: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }
.badge-rose    { background: #FFF1F2; color: #BE123C; border: 1px solid #FECDD3; }
.badge-slate   { background: #F1F5F9; color: #475569; border: 1px solid #E2E8F0; }
.badge-cyan    { background: #ECFEFF; color: #0E7490; border: 1px solid #A5F3FC; }

.kpi-label-text { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; margin-bottom: 4px; }
.kpi-value-text { font-size: 24px; font-weight: 700; color: #0F172A; letter-spacing: -0.02em; line-height: 1.2; font-family: 'Inter', sans-serif; }
.kpi-meta-text  { font-size: 12px; font-weight: 400; color: #94A3B8; margin-top: 4px; }

/* === Empty state \u2014 recessed dropzone === */
.empty-drop-zone {
    border: 2px dashed #CBD5E1; background: #F8FAFC;
    box-shadow: inset 0 2px 6px rgba(0,0,0,0.04);
    border-radius: 12px; padding: 40px 24px; text-align: center;
    max-width: 640px; margin: 16px auto 0 auto;
    transition: all 0.15s ease-in-out;
}
.empty-drop-zone:hover { border-color: #4F46E5; background: #FAFAFF; }
.empty-upload-icon {
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 16px auto; width: 56px; height: 56px;
    background: #EEF2FF; border-radius: 12px; border: 1px solid #E0E7FF;
}
.empty-title {
    font-size: 18px; font-weight: 600; color: #0F172A; margin-bottom: 6px;
    letter-spacing: -0.01em; font-family: 'Inter', sans-serif;
}
.empty-desc { font-size: 13px; color: #64748B; max-width: 420px; margin: 0 auto; line-height: 1.5; }
.connecting-label-row {
    display: flex; align-items: center; justify-content: center;
    gap: 12px; margin: 16px auto; max-width: 640px;
}
.connecting-line { flex: 1; height: 1px; background: #E2E8F0; }
.connecting-text { font-size: 11px; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; color: #94A3B8; }

/* === Sample cards \u2014 physically liftable === */
.sample-card {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 2px 6px rgba(0,0,0,0.04);
    display: flex; flex-direction: column; justify-content: space-between;
    transition: box-shadow 0.15s ease, transform 0.15s ease, border-color 0.15s ease;
    margin-bottom: 12px;
}
.sample-card:hover {
    border-color: #CBD5E1; box-shadow: 0 8px 20px rgba(0,0,0,0.08); transform: translateY(-2px);
}
.sample-card-indigo { border-top: 3px solid #4F46E5; }
.sample-card-cyan   { border-top: 3px solid #0891B2; }
.sample-card-icon { width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center; margin-bottom: 10px; }
.sample-icon-indigo { background: #EEF2FF; border: 1px solid #E0E7FF; }
.sample-icon-cyan   { background: #ECFEFF; border: 1px solid #A5F3FC; }
.sample-card-title { font-size: 15px; font-weight: 600; color: #0F172A; margin-bottom: 4px; font-family: 'Inter', sans-serif; letter-spacing: -0.01em; }
.sample-card-desc  { font-size: 13px; color: #64748B; line-height: 1.4; margin-bottom: 12px; }

/* === Chat === */
.chat-container { display: flex; flex-direction: column; gap: 16px; margin-top: 12px; }
.chat-turn { display: flex; flex-direction: column; gap: 8px; }
.chat-user-bubble {
    align-self: flex-end; background: #4F46E5; color: #FFFFFF;
    padding: 10px 16px; border-radius: 14px 14px 2px 14px;
    max-width: 75%; font-size: 13px; font-weight: 500; line-height: 1.5;
    box-shadow: 0 1px 2px rgba(79,70,229,0.2);
}
.chat-assistant-bubble {
    align-self: flex-start; background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 14px 14px 14px 2px; padding: 16px; width: 100%;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 1px 3px rgba(0,0,0,0.04);
}
.chat-meta-pill {
    display: inline-flex; align-items: center; gap: 6px; padding: 3px 8px;
    border-radius: 9999px; background: #EEF2FF; border: 1px solid #E0E7FF;
    color: #4338CA; font-size: 11px; font-weight: 600; margin-bottom: 10px;
}
.chat-answer-text { font-size: 13px; color: #1E293B; line-height: 1.6; margin-bottom: 12px; }
.code-box {
    background: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 12px;
    color: #E2E8F0; font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px; line-height: 1.6; overflow-x: auto;
}

/* === Tabs === */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #F1F5F9 !important; border: 1px solid #E2E8F0;
    border-radius: 8px !important; padding: 3px !important; margin-bottom: 16px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px !important; padding: 6px 14px !important;
    font-size: 13px !important; font-weight: 500 !important;
    color: #64748B !important; border: none !important;
    transition: all 0.15s ease !important; font-family: 'Inter', sans-serif !important;
}
.stTabs [aria-selected="true"] {
    background: #FFFFFF !important; color: #0F172A !important;
    font-weight: 600 !important; box-shadow: 0 1px 2px rgba(0,0,0,0.06) !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #F8FAFC; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 9999px; }
::-webkit-scrollbar-thumb:hover { background: #94A3B8; }
</style>
"""
st.markdown(DISTILL_CSS, unsafe_allow_html=True)


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

# Pill 1: system state
if has_dataset:
    p1_cls, p1_dot, p1_txt = "status-pill-active", "dot-emerald", "AST Sandbox Active"
else:
    p1_cls, p1_dot, p1_txt = "status-pill-neutral", "dot-slate", "System Idle"

# Pill 2: dataset
if has_dataset:
    p2_cls, p2_dot = "status-pill-active", "dot-indigo"
    p2_txt = f"{active_file} &bull; {row_count_str}"
else:
    p2_cls, p2_dot, p2_txt = "status-pill-neutral", "dot-slate", "No Dataset Loaded"

# Pill 3: engine (amber = most worth noticing)
if openai_ready:
    p3_cls, p3_dot, p3_txt = "status-pill-active", "dot-emerald", "OpenAI GPT-4o Online"
else:
    p3_cls, p3_dot, p3_txt = "status-pill-amber", "dot-amber", "Offline Sandbox Fallback"

header_html = f"""
<div class="distill-header">
    <div class="header-left">
        <div>
            <h1 class="brand-title">Distill <span style="color:#4F46E5;">Data Lab</span></h1>
            <p class="brand-subtitle">Autonomous Tabular Analytics &amp; AST Sandboxed Intelligence</p>
        </div>
    </div>
    <div class="header-right">
        <div class="status-pill {p1_cls}"><span class="status-dot {p1_dot}"></span><span>{p1_txt}</span></div>
        <div class="status-pill {p2_cls}">
            <span class="status-dot {p2_dot}"></span>
            <span style="max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{p2_txt}</span>
        </div>
        <div class="status-pill {p3_cls}"><span class="status-dot {p3_dot}"></span><span>{p3_txt}</span></div>
    </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)


with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-brand-card">
        <div>
            <div style="font-size:13px;font-weight:600;color:#0F172A;line-height:1.2;font-family:'Inter',sans-serif;">Distill Data</div>
            <div style="font-size:11px;color:#64748B;font-weight:400;">Data Lab &bull; Enterprise Edition</div>
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
        with st.spinner("Analyzing dataset..."):
            _load_data_source(uploaded, uploaded.name)
        st.rerun()

    st.markdown('<div class="sidebar-section-title">Benchmark Samples</div>', unsafe_allow_html=True)
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        if st.button("Retail Sales", use_container_width=True, key="btn_sample_sales", type="secondary"):
            with st.spinner("Loading sales..."):
                _load_data_source("sample_data/sales_clean.csv", "sales_clean.csv")
            st.rerun()
    with b_col2:
        if st.button("Churn Messy", use_container_width=True, key="btn_sample_churn", type="secondary"):
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
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:8px;padding:12px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                <span style="font-size:13px;font-weight:600;color:#0F172A;max-width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:'Inter',sans-serif;">{st.session_state.filename}</span>
                <span class="badge-pill {quality_badge}">{quality_text}</span>
            </div>
            <div style="font-size:12px;color:#475569;">
                <strong>{eda_side.total_rows:,}</strong> records &times; <strong>{eda_side.total_columns}</strong> columns
            </div>
            <div style="font-size:11px;color:#94A3B8;margin-top:4px;">
                Missing: {health_side.missing_percentage:.1f}% &bull; {len(eda_side.numeric_profiles)} numeric profiles
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section-title">OpenAI Engine Config</div>', unsafe_allow_html=True)
        if openai_ready:
            st.markdown(f"""
            <div style="font-size:12px;color:#059669;background:#ECFDF5;border:1px solid #D1FAE5;padding:8px 12px;border-radius:8px;font-weight:500;display:flex;align-items:center;gap:6px;">
                {ICONS['check']} API Key Connected (GPT-4o)
            </div>
            """, unsafe_allow_html=True)
        else:
            api_key = st.text_input(
                "OpenAI API Key", type="password",
                placeholder="sk-proj-...", key="input_api_key",
                help="Stored only in local session memory.",
            )
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                st.rerun()

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        if st.button("&#8635; Reset Session", use_container_width=True, type="secondary", key="btn_reset_all"):
            for k in ["dataset", "filename", "health_report", "eda_report",
                      "inferred_types", "cleaned_df", "clean_log", "query_history"]:
                st.session_state[k] = None if k not in ["clean_log", "query_history"] else []
            st.rerun()


if not has_dataset:
    st.markdown(f"""
    <div class="empty-drop-zone">
        <div class="empty-upload-icon">{ICONS['upload_cloud']}</div>
        <h2 class="empty-title">Upload a dataset to begin</h2>
        <p class="empty-desc">
            Drag and drop a CSV or Excel file into the sidebar, or select one of the curated benchmark datasets below to explore instant statistical profiling and conversational analytics.
        </p>
    </div>
    <div class="connecting-label-row">
        <div class="connecting-line"></div>
        <span class="connecting-text">or start with a sample</span>
        <div class="connecting-line"></div>
    </div>
    """, unsafe_allow_html=True)

    c_empty1, c_empty2 = st.columns(2)
    with c_empty1:
        st.markdown(f"""
        <div class="sample-card sample-card-indigo">
            <div>
                <div class="sample-card-icon sample-icon-indigo">{ICONS['trending_up']}</div>
                <div class="sample-card-title">Retail Sales Analytics</div>
                <div class="sample-card-desc">Clean e-commerce dataset with 1,000 transactions, product categories, and revenue.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        if st.button("Load Retail Sales Sample", use_container_width=True, type="primary", key="btn_empty_sales"):
            _load_data_source("sample_data/sales_clean.csv", "sales_clean.csv")
            st.rerun()
    with c_empty2:
        st.markdown(f"""
        <div class="sample-card sample-card-cyan">
            <div>
                <div class="sample-card-icon sample-icon-cyan">{ICONS['users']}</div>
                <div class="sample-card-title">Customer Churn (Messy)</div>
                <div class="sample-card-desc">Real-world subscription records with null tokens, whitespace issues, and duplicate rows.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        if st.button("Load Churn Sample", use_container_width=True, type="secondary", key="btn_empty_churn"):
            _load_data_source("sample_data/customer_churn_messy.csv", "customer_churn_messy.csv")
            st.rerun()
    st.stop()


df = st.session_state.dataset
health = st.session_state.health_report
eda = st.session_state.eda_report
types = st.session_state.inferred_types

tab_overview, tab_chat, tab_health_page, tab_automl, tab_stats, tab_explorer = st.tabs([
    "Overview Dashboard",
    "Conversational Workspace",
    "Data Health & Diagnostics",
    "Predictive Modeling Studio",
    "Statistical Lab & Anomalies",
    "Raw Data Explorer",
])

with tab_overview:
    missing_pct = health.missing_percentage if health else 0.0
    dup_rows = health.duplicate_rows if health else 0
    num_profs = len(eda.numeric_profiles) if eda else 0
    cat_profs = len(eda.categorical_profiles) if eda else 0
    quality_score = max(0, int(100 - (missing_pct * 1.5) - (dup_rows * 2)))

    missing_icon_cls = "icon-amber" if missing_pct >= 5 else "icon-emerald"
    missing_badge_cls = "badge-amber" if missing_pct >= 5 else "badge-emerald"
    missing_badge_txt = "Missing Found" if missing_pct >= 5 else "Optimal"
    dup_icon_cls = "icon-rose" if dup_rows > 0 else "icon-emerald"
    dup_badge_cls = "badge-rose" if dup_rows > 0 else "badge-emerald"
    dup_badge_txt = f"{dup_rows} Dups" if dup_rows > 0 else "Zero Dups"
    dup_meta = "Deduplication ready" if dup_rows > 0 else "No redundancy"

    kpi_html = f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-top-row">
                <div class="kpi-icon-pill icon-indigo">{ICONS['folder']}</div>
                <span class="badge-pill badge-indigo">100% Parsed</span>
            </div>
            <div class="kpi-label-text">Total Records</div>
            <div class="kpi-value-text">{len(df):,}</div>
            <div class="kpi-meta-text">Rows loaded in memory</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-top-row">
                <div class="kpi-icon-pill icon-violet">{ICONS['columns']}</div>
                <span class="badge-pill badge-slate">{num_profs} Num &bull; {cat_profs} Cat</span>
            </div>
            <div class="kpi-label-text">Total Columns</div>
            <div class="kpi-value-text">{len(df.columns)}</div>
            <div class="kpi-meta-text">Inferred dimensions</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-top-row">
                <div class="kpi-icon-pill {missing_icon_cls}">{ICONS['alert_triangle']}</div>
                <span class="badge-pill {missing_badge_cls}">{missing_badge_txt}</span>
            </div>
            <div class="kpi-label-text">Missing Cells</div>
            <div class="kpi-value-text">{health.missing_cells if health else 0:,}</div>
            <div class="kpi-meta-text">{missing_pct:.1f}% overall missingness</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-top-row">
                <div class="kpi-icon-pill {dup_icon_cls}">{ICONS['copy']}</div>
                <span class="badge-pill {dup_badge_cls}">{dup_badge_txt}</span>
            </div>
            <div class="kpi-label-text">Duplicate Rows</div>
            <div class="kpi-value-text">{dup_rows}</div>
            <div class="kpi-meta-text">{dup_meta}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-top-row">
                <div class="kpi-icon-pill icon-emerald">{ICONS['star']}</div>
                <span class="badge-pill badge-emerald">Composite</span>
            </div>
            <div class="kpi-label-text">Quality Score</div>
            <div class="kpi-value-text">{quality_score}/100</div>
            <div class="kpi-meta-text">Health index benchmark</div>
        </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)

    if eda and eda.summary_narrative:
        st.markdown(f"""
        <div class="distill-card" style="border-left: 3px solid #4F46E5;">
            <div class="card-header-bar">
                <span class="card-title-text">{ICONS['sparkles']} AI Executive Synthesis</span>
                <span class="badge-pill badge-indigo">Automated Profiling</span>
            </div>
            <p style="font-size:13px;color:#334155;line-height:1.6;margin:0;">{eda.summary_narrative}</p>
        </div>
        """, unsafe_allow_html=True)

    if eda and eda.numeric_profiles:
        st.markdown('<div class="card-overline" style="margin: 20px 0 10px;">Numeric Dimension Benchmarks</div>', unsafe_allow_html=True)
        chunk_size = 3
        for i in range(0, len(eda.numeric_profiles), chunk_size):
            chunk = eda.numeric_profiles[i : i + chunk_size]
            row_cols = st.columns(len(chunk))
            for idx, prof in enumerate(chunk):
                with row_cols[idx]:
                    fig = go.Figure(go.Indicator(
                        mode="number+delta+gauge",
                        value=prof.mean,
                        number={"font": {"size": 20, "family": "Inter", "color": "#0F172A"}},
                        delta={"reference": prof.median, "relative": False, "position": "bottom", "valueformat": ".2f"},
                        gauge={
                            "axis": {"range": [prof.min, prof.max], "tickfont": {"size": 9, "color": "#64748B"}},
                            "bar": {"color": "#4F46E5"}, "bgcolor": "#F8FAFC", "borderwidth": 0,
                            "steps": [
                                {"range": [prof.min, prof.q1], "color": "#EEF2FF"},
                                {"range": [prof.q1, prof.q3], "color": "#E0E7FF"},
                            ],
                        },
                        title={"text": f"<b>{prof.column}</b><br><span style='font-size:11px;color:#64748B;'>Mean vs Median</span>",
                               "font": {"size": 13, "family": "Inter"}},
                    ))
                    fig.update_layout(height=195, margin=dict(l=12, r=12, t=40, b=12),
                                      paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
                    st.plotly_chart(fig, use_container_width=True, key=f"gauge_{prof.column}")
                    if prof.outlier_count > 0:
                        st.caption(f"**{prof.outlier_count}** outlier(s) detected via IQR.")

    if eda and eda.categorical_profiles:
        st.markdown('<div class="card-overline" style="margin: 20px 0 10px;">Categorical Distributions &amp; Cardinality</div>', unsafe_allow_html=True)
        cat_cols = st.columns(min(len(eda.categorical_profiles), 3))
        for idx, cat_p in enumerate(eda.categorical_profiles[:3]):
            with cat_cols[idx]:
                if cat_p.top_categories:
                    cat_df = pd.DataFrame(cat_p.top_categories)
                    fig = px.bar(cat_df.head(7), x="count", y="value", orientation="h",
                                 title=f"{cat_p.column} ({cat_p.cardinality} unique)",
                                 color_discrete_sequence=["#4F46E5"], template="none")
                    fig.update_layout(
                        height=240, margin=dict(l=8, r=8, t=32, b=8),
                        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                        yaxis=dict(title="", autorange="reversed", tickfont=dict(color="#334155", size=10)),
                        xaxis=dict(title="Count", showgrid=True, gridcolor="#F1F5F9", tickfont=dict(color="#64748B")),
                        font=dict(family="Inter", size=11),
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"cat_{cat_p.column}")

    if eda and eda.datetime_profiles:
        st.markdown('<div class="card-overline" style="margin: 20px 0 10px;">Temporal Dimensions</div>', unsafe_allow_html=True)
        for dt_p in eda.datetime_profiles:
            st.markdown(f"""
            <div class="distill-card" style="display:flex;justify-content:space-between;align-items:center;padding:12px 16px;">
                <div>
                    <strong style="color:#0F172A;font-size:13px;font-family:'Inter',sans-serif;">{dt_p.column}</strong>
                    <span style="color:#64748B;font-size:12px;margin-left:8px;">Frequency: <strong>{dt_p.detected_frequency}</strong></span>
                </div>
                <div style="font-size:12px;color:#475569;">
                    <span class="badge-pill badge-indigo">{dt_p.min_date} &rarr; {dt_p.max_date}</span>
                    <span style="color:#94A3B8;margin-left:6px;">({dt_p.date_range_days} days)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


with tab_chat:
    st.markdown(f"""
    <div class="distill-card">
        <div class="card-header-bar">
            <span class="card-title-text">{ICONS['message_square']} Conversational Analysis Assistant</span>
            <span class="badge-pill badge-indigo">Zero Data Leakage &bull; Schema Prompts</span>
        </div>
        <p style="font-size:13px;color:#64748B;margin:0;line-height:1.5;">
            Ask questions in plain English. The query engine generates sandboxed Python Pandas &amp; Plotly code and executes it within an isolated subprocess in under 5 seconds.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<span class='card-overline'>Suggested Prompts:</span>", unsafe_allow_html=True)
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
            if st.button(ex, key=f"pill_{i}", use_container_width=True, type="secondary"):
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
        execute_click = st.button("Run Analysis", type="primary", use_container_width=True, key="btn_exec_query")
    with col_clear:
        if st.button("Clear", type="secondary", use_container_width=True, key="btn_clear_thread"):
            st.session_state.query_history = []
            st.rerun()

    if "pending_question" in st.session_state:
        del st.session_state["pending_question"]

    if execute_click and user_input.strip():
        with st.spinner("Generating analytical plan & executing sandboxed code..."):
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
            mode_label = "Offline Mock Mode" if item.get("is_mock") else "GPT-4o Plan"
            st.markdown(f"""
            <div class="chat-turn">
                <div class="chat-user-bubble">{item['question']}</div>
                <div class="chat-assistant-bubble">
                    <div class="chat-meta-pill">
                        {ICONS['zap_indigo']} In-Memory AST Sandbox &bull; {item['time_ms']}ms &bull; {mode_label}
                    </div>
                    <div class="chat-answer-text">{item['answer']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if item["result_type"] == "chart" and item["result_json"]:
                try:
                    fig = pio.from_json(item["result_json"])
                    fig.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                                      margin=dict(l=10, r=10, t=32, b=10), height=400,
                                      font=dict(family="Inter", size=11))
                    st.plotly_chart(fig, use_container_width=True, key=f"chat_fig_{idx}")
                except Exception as err:
                    st.warning(f"Chart render issue: {err}")
            elif item["result_type"] == "table" and item["result_json"]:
                try:
                    tbl = pd.read_json(item["result_json"], orient="records")
                    st.dataframe(tbl, use_container_width=True, height=260, key=f"chat_tbl_{idx}")
                except Exception as err:
                    st.warning(f"Table render issue: {err}")
            elif item["result_type"] == "scalar" and item["result_json"]:
                st.metric("Computed Metric", item["result_json"])
            elif item["result_type"] == "error":
                st.error(f"Sandbox Runtime Error: {item.get('error', 'Execution interrupted')}")

            if item.get("code"):
                with st.expander("Inspect Executed Python Code (AST Verified)", expanded=False):
                    st.markdown(f'<div class="code-box">{item["code"]}</div>', unsafe_allow_html=True)
                    st.caption(f"Verified via AST allowlist &bull; Executed in {item['time_ms']}ms")

            st.markdown("<hr style='border:0;border-top:1px solid #E2E8F0;margin:16px 0;'>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;padding:32px 16px;text-align:center;color:#94A3B8;margin-top:16px;">
            <strong style="color:#475569;font-size:13px;">No queries in this session yet</strong>
            <p style="font-size:12px;color:#94A3B8;margin-top:4px;">Select a prompt above or ask any question to inspect results.</p>
        </div>
        """, unsafe_allow_html=True)


with tab_health_page:
    st.markdown(f"""
    <div class="distill-card">
        <div class="card-header-bar">
            <span class="card-title-text">{ICONS['shield']} Data Quality &amp; Health Diagnostics</span>
            <span class="badge-pill badge-emerald">Non-Destructive Operations</span>
        </div>
        <p style="font-size:13px;color:#64748B;margin:0;line-height:1.5;">
            Transparent audit verifying null token distributions, whitespace anomalies, duplicate rows, and schema fidelity.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if health.anomalies:
        for anomaly in health.anomalies:
            st.warning(f"**Anomaly Identified:** {anomaly}")

    if health.cleaning_recommendations:
        st.markdown("<span class='card-overline'>Recommended Remediations:</span>", unsafe_allow_html=True)
        for rec in health.cleaning_recommendations:
            st.markdown(f"- {rec}")

    st.markdown('<div class="card-overline" style="margin:20px 0 8px;">Column Health Diagnostics</div>', unsafe_allow_html=True)
    c_rows = []
    for col_n, col_h in health.columns_health.items():
        warns = " | ".join(col_h.warnings) if col_h.warnings else "Healthy"
        c_rows.append({
            "Column": col_n,
            "Detected Type": col_h.inferred_type,
            "Missing Cells": f"{col_h.missing_count} ({col_h.missing_pct:.1f}%)",
            "Unique Values": col_h.unique_count,
            "Sample Values": ", ".join(str(v) for v in col_h.sample_values[:3]),
            "Diagnostic Status": warns,
        })
    st.dataframe(pd.DataFrame(c_rows), use_container_width=True, height=340)

    st.markdown('<div class="card-overline" style="margin:20px 0 8px;">1-Click Non-Destructive Cleaning Pipeline</div>', unsafe_allow_html=True)
    cl_col1, cl_col2, cl_col3, cl_col4 = st.columns(4)
    with cl_col1:
        opt_dedup = st.checkbox("Remove duplicate rows", value=True, key="c_dedup")
    with cl_col2:
        opt_ws = st.checkbox("Strip whitespace", value=True, key="c_ws")
    with cl_col3:
        opt_null = st.checkbox("Normalize null tokens (NA/null)", value=True, key="c_null")
    with cl_col4:
        opt_num = st.checkbox("Coerce numeric text fields", value=True, key="c_num")

    if st.button("Clean & Remediate Dataset", type="primary", key="btn_clean_exec"):
        with st.spinner("Executing non-destructive cleaning pipeline..."):
            cleaned, log_actions = clean_dataset(
                df, remove_duplicates=opt_dedup, strip_whitespace=opt_ws,
                normalize_nulls=opt_null, coerce_numeric=opt_num,
            )
        st.session_state.cleaned_df = cleaned
        st.session_state.clean_log = log_actions
        if log_actions:
            for action in log_actions:
                st.success(f"{action}")
        else:
            st.info("No modifications needed \u2014 dataset satisfies all quality thresholds.")

    if st.session_state.cleaned_df is not None:
        if st.button("Replace Active Session Dataset with Cleaned Copy", type="secondary", key="btn_replace_cleaned"):
            _load_data_source(st.session_state.cleaned_df, f"Cleaned_{st.session_state.filename}")
            st.success("Active dataset updated to cleaned version.")
            st.rerun()



with tab_automl:
    st.markdown("""
    <div class="card-header-bar">
        <div>
            <div class="card-overline">SUPERVISED INTELLIGENCE</div>
            <div class="card-title-text">AutoML & Predictive Modeling Studio</div>
        </div>
        <span class="badge badge-indigo">Multi-Model Tournament</span>
    </div>
    """, unsafe_allow_html=True)

    c_sel1, c_sel2, c_sel3 = st.columns([2, 1, 1])
    with c_sel1:
        target_candidate = st.selectbox(
            "Prediction Target Variable",
            options=list(df.columns),
            index=len(df.columns) - 1,
            key="automl_target_col"
        )
    with c_sel2:
        try:
            inferred_tt = infer_task_type(df, target_candidate)
            default_idx = 0 if inferred_tt == TaskType.CLASSIFICATION else 1
        except Exception:
            default_idx = 0
        chosen_task_str = st.selectbox(
            "Task Type",
            options=["Classification", "Regression"],
            index=default_idx,
            key="automl_task_type"
        )
        task_enum = TaskType.CLASSIFICATION if chosen_task_str == "Classification" else TaskType.REGRESSION
    with c_sel3:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        run_tourney = st.button("Launch Tournament", type="primary", use_container_width=True, key="btn_run_automl")

    if run_tourney:
        with st.spinner(f"Training 5 candidate models against '{target_candidate}'..."):
            try:
                res = run_automl_tournament(df, target_candidate, task_type=task_enum)
                st.session_state.automl_result = res
                st.success(f"Tournament complete! Winner: {res.best_model_name}")
            except Exception as e:
                st.error(f"AutoML tournament failed: {e}")

    automl_res = st.session_state.automl_result
    if automl_res is not None and automl_res.target_column in df.columns:
        st.markdown(f"""
        <div class="distill-card" style="margin-top:16px; margin-bottom:16px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-size:11px; font-weight:600; letter-spacing:0.05em; color:#4F46E5; text-transform:uppercase;">TOURNAMENT WINNER</div>
                    <div style="font-size:18px; font-weight:700; color:#0F172A; margin-top:2px;">{automl_res.best_model_name}</div>
                </div>
                <span class="badge badge-emerald">Best Benchmark Score</span>
            </div>
            <p style="font-size:13px; color:#64748B; margin:8px 0 0 0;">{automl_res.summary}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='font-size:14px; font-weight:600; color:#0F172A; margin-bottom:10px;'>Leaderboard & Candidate Comparison</div>", unsafe_allow_html=True)
        leaderboard_rows = []
        for idx, cand in enumerate(automl_res.candidate_models):
            row = {"Rank": f"#{idx+1}", "Model Architecture": cand.model_name, "Fit Time (s)": cand.fit_time_seconds}
            row.update(cand.metrics)
            leaderboard_rows.append(row)
        st.dataframe(pd.DataFrame(leaderboard_rows), use_container_width=True, hide_index=True)

        # Feature Importance & Live Prediction Playground
        col_chart, col_play = st.columns([1, 1])
        with col_chart:
            st.markdown("<div style='font-size:14px; font-weight:600; color:#0F172A; margin:16px 0 10px 0;'>Feature Influence & Drivers</div>", unsafe_allow_html=True)
            best_eval = next((m for m in automl_res.candidate_models if m.model_name == automl_res.best_model_name), None)
            if best_eval and best_eval.feature_importances:
                imp_df = pd.DataFrame(best_eval.feature_importances)
                fig_imp = px.bar(
                    imp_df.sort_values(by="importance", ascending=True),
                    x="importance",
                    y="feature",
                    orientation="h",
                    color="importance",
                    color_continuous_scale=["#E0E7FF", "#4F46E5"],
                    title=f"Top Drivers for {automl_res.target_column}",
                )
                fig_imp.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_family="Inter",
                    height=360,
                    margin=dict(l=20, r=20, t=40, b=20),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Feature importances not directly extractable for this model architecture.")

        with col_play:
            st.markdown("<div style='font-size:14px; font-weight:600; color:#0F172A; margin:16px 0 10px 0;'>Interactive Prediction Sandbox</div>", unsafe_allow_html=True)
            with st.expander("Configure Sample Input Values", expanded=True):
                user_inputs = {}
                feature_subset = automl_res.feature_columns[:8]  # top 8 for clean UI
                for feat in feature_subset:
                    s_col = df[feat].dropna()
                    if pd.api.types.is_numeric_dtype(s_col):
                        min_v = float(s_col.min())
                        max_v = float(s_col.max())
                        mean_v = float(s_col.mean())
                        user_inputs[feat] = st.number_input(f"{feat}", value=round(mean_v, 2), key=f"pred_in_{feat}")
                    else:
                        top_vals = list(s_col.astype(str).unique()[:10])
                        user_inputs[feat] = st.selectbox(f"{feat}", options=top_vals, key=f"pred_in_{feat}")

                if st.button("Run Model Inference", type="primary", use_container_width=True, key="btn_exec_prediction"):
                    single_res = predict_single(
                        pipeline=automl_res.best_pipeline,
                        input_values=user_inputs,
                        feature_columns=automl_res.feature_columns,
                        task_type=automl_res.task_type,
                        classes=automl_res.classes_,
                    )
                    st.markdown(f"""
                    <div style="background:#F8FAFC; border:1px solid #CBD5E1; border-radius:10px; padding:12px; margin-top:10px; text-align:center;">
                        <div style="font-size:11px; font-weight:600; color:#64748B; text-transform:uppercase;">PREDICTED {automl_res.target_column.upper()}</div>
                        <div style="font-size:22px; font-weight:700; color:#4F46E5; margin-top:2px;">{single_res.prediction}</div>
                        {"<div style='font-size:12px; color:#10B981; font-weight:600; margin-top:4px;'>Confidence: " + str(round(single_res.confidence * 100, 1)) + "%</div>" if single_res.confidence is not None else ""}
                    </div>
                    """, unsafe_allow_html=True)



with tab_stats:
    st.markdown("""
    <div class="card-header-bar">
        <div>
            <div class="card-overline">INFERENTIAL RIGOR</div>
            <div class="card-title-text">Advanced Statistical Testing & Multidimensional Anomalies</div>
        </div>
        <span class="badge badge-cyan">Hypothesis Engine</span>
    </div>
    """, unsafe_allow_html=True)

    stat_mode = st.radio(
        "Investigation Mode",
        options=["Two-Sample Hypothesis Test", "Normality & Distribution Check", "Significance Correlation Matrix", "Multivariate Anomaly Discovery"],
        horizontal=True,
        key="radio_stat_mode"
    )

    numeric_columns = list(df.select_dtypes(include=[np.number]).columns)
    categorical_columns = list(df.select_dtypes(exclude=[np.number]).columns)

    if stat_mode == "Two-Sample Hypothesis Test":
        c_test1, c_test2, c_test3 = st.columns(3)
        with c_test1:
            numeric_target = st.selectbox("Metric to Compare (Numeric)", options=numeric_columns, key="hyp_num_target")
        with c_test2:
            grouping_col = st.selectbox("Grouping Variable (Categorical)", options=categorical_columns if categorical_columns else list(df.columns), key="hyp_group_col")
        with c_test3:
            unique_groups = list(df[grouping_col].dropna().astype(str).unique())[:10] if grouping_col in df.columns else []
            if len(unique_groups) >= 2:
                sel_g1 = st.selectbox("Group A", options=unique_groups, index=0, key="hyp_g1")
                sel_g2 = st.selectbox("Group B", options=unique_groups, index=min(1, len(unique_groups)-1), key="hyp_g2")
            else:
                sel_g1, sel_g2 = None, None
                st.warning("Selected grouping variable must have at least 2 distinct values.")

        if sel_g1 and sel_g2 and sel_g1 != sel_g2:
            s_a = df[df[grouping_col].astype(str) == sel_g1][numeric_target]
            s_b = df[df[grouping_col].astype(str) == sel_g2][numeric_target]

            try:
                rec_type, rec_reason = recommend_hypothesis_test(s_a, s_b)
                st.info(f"Recommended Statistical Test: **{rec_reason}**")
                res = run_two_sample_test(s_a, s_b, name_a=f"{grouping_col}={sel_g1}", name_b=f"{grouping_col}={sel_g2}", test_type=rec_type)

                sig_badge = '<span class="badge badge-emerald">Significant Difference (p < 0.05)</span>' if res.is_significant else '<span class="badge badge-slate">No Significant Difference (p >= 0.05)</span>'

                st.markdown(f"""
                <div class="distill-card" style="margin-top:12px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div style="font-size:16px; font-weight:600; color:#0F172A;">{res.test_name}</div>
                        {sig_badge}
                    </div>
                    <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; margin-top:12px;">
                        <div style="background:#F8FAFC; padding:8px 12px; border-radius:6px;">
                            <div style="font-size:11px; color:#64748B;">TEST STATISTIC</div>
                            <div style="font-size:16px; font-weight:700; color:#0F172A;">{res.statistic}</div>
                        </div>
                        <div style="background:#F8FAFC; padding:8px 12px; border-radius:6px;">
                            <div style="font-size:11px; color:#64748B;">P-VALUE</div>
                            <div style="font-size:16px; font-weight:700; color:#4F46E5;">{res.p_value}</div>
                        </div>
                        <div style="background:#F8FAFC; padding:8px 12px; border-radius:6px;">
                            <div style="font-size:11px; color:#64748B;">{res.group_names[0]} Mean</div>
                            <div style="font-size:16px; font-weight:700; color:#0F172A;">{res.group_means[0]}</div>
                        </div>
                        <div style="background:#F8FAFC; padding:8px 12px; border-radius:6px;">
                            <div style="font-size:11px; color:#64748B;">{res.group_names[1]} Mean</div>
                            <div style="font-size:16px; font-weight:700; color:#0F172A;">{res.group_means[1]}</div>
                        </div>
                    </div>
                    <p style="font-size:13px; color:#334155; margin:12px 0 0 0; line-height:1.5;">{res.interpretation}</p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Test failed: {e}")

    elif stat_mode == "Normality & Distribution Check":
        norm_col = st.selectbox("Select Continuous Metric", options=numeric_columns, key="norm_target_col")
        if norm_col:
            try:
                norm_res = check_normality(df[norm_col], norm_col)
                status_pill = '<span class="badge badge-emerald">Normal (Gaussian)</span>' if norm_res.is_normal else '<span class="badge badge-amber">Non-Normal Distribution</span>'
                st.markdown(f"""
                <div class="distill-card" style="margin-top:12px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div style="font-size:16px; font-weight:600; color:#0F172A;">Normality Assessment: {norm_col}</div>
                        {status_pill}
                    </div>
                    <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; margin-top:12px;">
                        <div style="background:#F8FAFC; padding:8px 12px; border-radius:6px;">
                            <div style="font-size:11px; color:#64748B;">P-VALUE</div>
                            <div style="font-size:16px; font-weight:700; color:#4F46E5;">{norm_res.p_value}</div>
                        </div>
                        <div style="background:#F8FAFC; padding:8px 12px; border-radius:6px;">
                            <div style="font-size:11px; color:#64748B;">STATISTIC</div>
                            <div style="font-size:16px; font-weight:700; color:#0F172A;">{norm_res.statistic}</div>
                        </div>
                        <div style="background:#F8FAFC; padding:8px 12px; border-radius:6px;">
                            <div style="font-size:11px; color:#64748B;">SKEWNESS</div>
                            <div style="font-size:16px; font-weight:700; color:#0F172A;">{norm_res.skewness}</div>
                        </div>
                        <div style="background:#F8FAFC; padding:8px 12px; border-radius:6px;">
                            <div style="font-size:11px; color:#64748B;">KURTOSIS</div>
                            <div style="font-size:16px; font-weight:700; color:#0F172A;">{norm_res.kurtosis}</div>
                        </div>
                    </div>
                    <p style="font-size:13px; color:#334155; margin:12px 0 0 0;">{norm_res.interpretation}</p>
                </div>
                """, unsafe_allow_html=True)

                fig_hist = px.histogram(df, x=norm_col, marginal="box", nbins=30, color_discrete_sequence=["#4F46E5"])
                fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_family="Inter", height=320)
                st.plotly_chart(fig_hist, use_container_width=True)
            except Exception as e:
                st.error(f"Normality analysis failed: {e}")

    elif stat_mode == "Significance Correlation Matrix":
        if len(numeric_columns) >= 2:
            corr_m, pval_m, sig_pairs = compute_correlation_significance(df)
            st.markdown("<div style='font-size:14px; font-weight:600; color:#0F172A; margin:12px 0 8px 0;'>Correlation Matrix Heatmap</div>", unsafe_allow_html=True)
            fig_hm = px.imshow(
                corr_m,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                aspect="auto"
            )
            fig_hm.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_family="Inter", height=380)
            st.plotly_chart(fig_hm, use_container_width=True)

            st.markdown("<div style='font-size:14px; font-weight:600; color:#0F172A; margin:12px 0 8px 0;'>Statistically Significant Pairs Ranked</div>", unsafe_allow_html=True)
            table_rows = [
                {"Metric 1": p.var1, "Metric 2": p.var2, "Correlation r": p.coefficient, "p-value": p.p_value, "Strength": p.strength, "Sig Level": p.significance_symbol}
                for p in sig_pairs if p.is_significant
            ]
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
        else:
            st.warning("Need at least 2 numeric columns for correlation analysis.")

    elif stat_mode == "Multivariate Anomaly Discovery":
        if len(numeric_columns) >= 2:
            c_anom1, c_anom2 = st.columns([3, 1])
            with c_anom1:
                contam = st.slider("Expected Contamination Rate (Anomaly %)", min_value=0.01, max_value=0.15, value=0.05, step=0.01, key="slider_contam")
            with c_anom2:
                st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                run_anom = st.button("Detect Anomalies", type="primary", use_container_width=True, key="btn_run_anomalies")

            if run_anom or ("anomaly_res" in st.session_state and st.session_state.anomaly_res is not None):
                if run_anom:
                    with st.spinner("Executing Isolation Forest across multidimensional space..."):
                        try:
                            anom_res = detect_multivariate_anomalies(df, contamination=contam)
                            st.session_state.anomaly_res = anom_res
                        except Exception as e:
                            st.error(f"Anomaly detection failed: {e}")
                            st.session_state.anomaly_res = None

                anom_res = st.session_state.get("anomaly_res")
                if anom_res:
                    st.markdown(f"""
                    <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:12px; margin-top:12px;">
                        <div class="kpi-card">
                            <div class="kpi-label">TOTAL RECORDS AUDITED</div>
                            <div class="kpi-value">{anom_res.total_records}</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-label">OUTLIERS FLAGGED</div>
                            <div class="kpi-value" style="color:#EF4444;">{anom_res.outlier_count}</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-label">OUTLIER RATIO</div>
                            <div class="kpi-value" style="color:#F59E0B;">{anom_res.outlier_percentage}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    col_sc1, col_sc2 = st.columns(2)
                    feat_x = col_sc1.selectbox("Projection Axis X", options=anom_res.feature_columns, index=0, key="anom_x")
                    feat_y = col_sc2.selectbox("Projection Axis Y", options=anom_res.feature_columns, index=min(1, len(anom_res.feature_columns)-1), key="anom_y")

                    plot_df = df.copy()
                    plot_df["Status"] = ["Anomaly" if i in anom_res.outlier_indices else "Inlier" for i in df.index]
                    plot_df["Anomaly Score"] = anom_res.anomaly_scores

                    fig_anom = px.scatter(
                        plot_df,
                        x=feat_x,
                        y=feat_y,
                        color="Status",
                        color_discrete_map={"Inlier": "#94A3B8", "Anomaly": "#EF4444"},
                        hover_data=["Anomaly Score"],
                        title="Multivariate Isolation Forest Projection",
                    )
                    fig_anom.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_family="Inter", height=380)
                    st.plotly_chart(fig_anom, use_container_width=True)

                    st.markdown("<div style='font-size:14px; font-weight:600; color:#0F172A; margin:12px 0 8px 0;'>Top Flagged Anomalies</div>", unsafe_allow_html=True)
                    st.dataframe(anom_res.top_anomalous_records, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric features for multivariate anomaly analysis.")

with tab_explorer:
    st.markdown(f"""
    <div class="distill-card">
        <div class="card-header-bar">
            <span class="card-title-text">{ICONS['table']} Interactive Raw Data Grid</span>
            <span class="badge-pill badge-slate">Viewing {len(df):,} records</span>
        </div>
        <p style="font-size:13px;color:#64748B;margin:0;line-height:1.5;">
            Filter, sort, and inspect tabular records. Export filtered views directly to CSV.
        </p>
    </div>
    """, unsafe_allow_html=True)

    f_col1, f_col2 = st.columns([3, 1])
    with f_col1:
        col_selector = st.multiselect("Columns to display", options=list(df.columns), default=list(df.columns), key="grid_cols")
    with f_col2:
        row_limit = st.selectbox("Rows limit", [50, 100, 500, 1000, len(df)], index=1, key="grid_limit")

    filtered_df = df[col_selector].head(row_limit) if col_selector else df.head(row_limit)
    st.dataframe(filtered_df, use_container_width=True, height=420)

    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Export View to CSV",
        data=csv_bytes,
        file_name=f"export_{st.session_state.filename}",
        mime="text/csv",
        type="secondary",
        key="btn_download_csv",
    )
