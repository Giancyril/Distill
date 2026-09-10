"""
AI Data Analysis - Main Streamlit Application
Turn raw data into actionable insights in seconds.
"""
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="AI Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling injection for professional aesthetic
st.markdown("""
<style>
    /* Clean main typography & card spacing */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1300px;
    }
    
    /* Header card styling */
    .hero-container {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .hero-title {
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        opacity: 0.9;
    }
    
    /* KPI Metric Cards */
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 0.75rem;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">📊 AI Data Analysis</div>
    <div class="hero-subtitle">Upload your CSV or Excel dataset to automatically profile trends, surface anomalies, and query in plain English.</div>
</div>
""", unsafe_allow_html=True)

# Session state initialization
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar upload & configuration
with st.sidebar:
    st.header("Data Source")
    uploaded_file = st.file_uploader(
        "Upload dataset (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        help="Upload tabular data up to 50MB."
    )
    
    # Preset sample dataset quick loader
    st.markdown("---")
    st.subheader("Try Sample Datasets")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📈 Sales 2024", use_container_width=True):
            st.session_state.dataset = pd.read_csv("sample_data/sales_clean.csv")
            st.session_state.filename = "sales_clean.csv"
            st.rerun()
    with col2:
        if st.button("👥 Churn Data", use_container_width=True):
            st.session_state.dataset = pd.read_csv("sample_data/customer_churn_messy.csv")
            st.session_state.filename = "customer_churn_messy.csv"
            st.rerun()

    if st.session_state.dataset is not None:
        if st.button("Reset Dataset", type="secondary", use_container_width=True):
            st.session_state.dataset = None
            st.session_state.filename = None
            st.session_state.chat_history = []
            st.rerun()

if uploaded_file is not None and (st.session_state.filename != uploaded_file.name):
    try:
        if uploaded_file.name.endswith(".csv"):
            st.session_state.dataset = pd.read_csv(uploaded_file)
        else:
            st.session_state.dataset = pd.read_excel(uploaded_file)
        st.session_state.filename = uploaded_file.name
        st.success(f"Successfully loaded `{uploaded_file.name}` ({len(st.session_state.dataset)} rows)")
    except Exception as e:
        st.error(f"Error loading file: {e}")

# Main content
if st.session_state.dataset is None:
    st.info("👈 Please upload a CSV or Excel file in the sidebar, or select one of the sample datasets to get started.")
else:
    df = st.session_state.dataset
    st.subheader(f"Active Dataset: {st.session_state.filename}")
    
    # Top KPI summary cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Rows</div>
            <div class="kpi-value">{len(df):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Columns</div>
            <div class="kpi-value">{len(df.columns)}</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        missing_count = int(df.isna().sum().sum())
        missing_pct = (missing_count / (df.size)) * 100 if df.size > 0 else 0
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Missing Cells</div>
            <div class="kpi-value">{missing_count:,} <span style="font-size: 0.9rem; color: #64748B;">({missing_pct:.1f}%)</span></div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        numeric_cols = len(df.select_dtypes(include=['number']).columns)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Numeric Features</div>
            <div class="kpi-value">{numeric_cols}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Data preview table
    with st.expander("🔍 Preview Raw Records (First 10 rows)", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
