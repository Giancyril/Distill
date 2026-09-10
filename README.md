# AI Data Analysis

A production-grade, AI-augmented Data Analysis platform designed to transform raw tabular data into actionable insights in seconds. Upload any CSV or Excel file, ask questions in plain English, and instantly receive interactive Plotly charts, data tables, narrative summaries, and automated exploratory analysis — all powered by GPT-4o with a sandboxed, security-hardened execution engine.

## Features

### Core Functionality
- **Drag-and-drop Upload**: Supports CSV and Excel (`.xlsx`, `.xls`) up to 50MB with automatic encoding sniffing and delimiter detection
- **Automated EDA**: Instantly profiles numeric distributions (mean, median, IQR, outliers, skewness), categorical frequencies, and datetime ranges on upload
- **Natural Language Queries**: Ask questions in plain English — GPT-4o translates them into Pandas/Plotly code and executes it
- **Sandboxed Code Execution**: LLM-generated code runs in a restricted subprocess with AST validation, 5-second timeout, and zero filesystem/network access
- **Session Query History**: All queries and results are preserved within the session with expandable code viewers
- **Sample Datasets**: Built-in Sales 2024 and Customer Churn datasets for instant demos

### AI & Analysis Engine
- **GPT-4o / GPT-4o-mini**: Schema-aware prompt synthesis — only column names, types, and 3 sample values are sent (never raw data rows) for privacy and cost efficiency
- **Intent-aware Charting**: LLM classifies analytical intent (trend, comparison, distribution, correlation, composition) and routes to the optimal Plotly chart type
- **Mock Fallback**: Descriptive statistics table returned when no API key is configured, enabling offline use
- **Graceful Degradation**: All errors (LLM timeouts, malformed code, sandbox violations) surface with clear user-facing messages

### Data Health & Cleaning
- **Health Reports**: Per-column missingness counts and percentages, duplicate row detection, whitespace anomalies, high-missingness alerts
- **Null Token Normalization**: Recognizes `n/a`, `NA`, `none`, `null`, `N.A.`, `--`, `missing` and normalizes them to `NaN`
- **Transparent Cleaning**: Deduplication, whitespace trimming, null normalization, and string-to-numeric coercion — each action logged for auditability
- **Type Inference**: Classifies columns as `numeric`, `categorical`, `datetime`, `boolean`, `id_text`, or `text`

### Security Architecture
- **AST Allowlisting**: Static analysis rejects all `import os`, `import sys`, `import subprocess`, `socket`, `shutil`, `pickle`, `ctypes`, and 15+ other dangerous modules before execution
- **Dunder Attribute Blocking**: `__subclasses__`, `__globals__`, `__builtins__`, `__import__` and similar dunder-based exploit patterns are rejected
- **Subprocess Isolation**: Code runs in a completely separate Python process with DataFrame passed via stdin pickle; stdout JSON is the only output channel
- **Hard Timeout**: 5-second wall-clock timeout kills any hanging or infinite-loop code

### Premium UI
- **Inter Typography**: Clean, professional sans-serif font throughout
- **Indigo/Violet Design System**: Curated HSL color palette with `#6366F1` primary, `#8B5CF6` secondary, `#10B981` success, `#F59E0B` warning
- **Animated KPI Cards**: Gradient top-border cards with live metrics from the loaded dataset
- **Multi-tab Layout**: Dashboard (auto-charts + KPIs), Query Workspace (NL query + history), Data Health (reports + cleaning controls)
- **Collapsible Code Inspector**: Every query result includes a "View Generated Code" expander for full auditability

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web App (app/)                 │
│  Dashboard Tab: KPI Cards, EDA Gauges, Categorical Charts   │
│  Query Workspace: NL Query Bar, Result Cards, Code Viewer   │
│  Data Health: Column Reports, Anomaly Alerts, Cleaning UI   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             Core Analysis Engine (core/) [Framework-agnostic]│
│  ├── ingestion.py:     CSV/Excel loading, encoding sniffing  │
│  ├── cleaning.py:      Type detection, health auditing       │
│  ├── eda.py:           Statistical profiling, EDA reports    │
│  ├── query_engine.py:  Schema extraction, GPT-4o prompting   │
│  ├── sandbox.py:       AST validation + subprocess isolation  │
│  └── charts.py:        Plotly chart generator, design system │
└──────────────────────────────┬──────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
┌────────────────────────┐   ┌───────────────────────────────┐
│    OpenAI GPT-4o-mini  │   │    Sandboxed Subprocess       │
│  JSON-mode response    │   │  exec() in restricted env     │
│  Schema-only prompts   │   │  5s timeout, stdin/stdout IPC │
└────────────────────────┘   └───────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit 1.35+ with custom CSS injection |
| Data Processing | Pandas 2.2+, NumPy |
| Visualization | Plotly Express + Graph Objects |
| AI / LLM | OpenAI Python SDK (GPT-4o / GPT-4o-mini) |
| Security | Python `ast` module + `subprocess` isolation |
| Testing | pytest 8.0+, pytest-asyncio |
| File Formats | CSV, Excel (openpyxl) |

## Setup

### Prerequisites
- Python 3.11+
- An OpenAI API key (optional — mock fallback available without it)

### Installation

```bash
git clone <repo-url>
cd "AI Data Analysis"
pip install -r requirements.txt
```

### Configuration

```bash
# Create a .env file
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

Or enter the API key directly in the sidebar of the app.

### Run

```bash
streamlit run app/main.py
```

The app will open at `http://localhost:8501`.

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run sandbox security tests only
python -m pytest tests/test_sandbox.py -k "test_blocked" -v

# Run with coverage
python -m pytest tests/ --cov=core --cov-report=term-missing
```

## Sample Datasets

| Dataset | Rows | Columns | Description |
|---------|------|---------|-------------|
| `sales_clean.csv` | 250 | 8 | 2024 product sales with revenue, region, and category |
| `customer_churn_messy.csv` | 8 | 9 | Intentionally dirty dataset for cleaning pipeline demo |

## Example Queries

Once a dataset is loaded, try these natural language questions:

- *"What was our total revenue by product category?"*
- *"Show me the monthly sales trend for 2024"*
- *"Which region had the highest average order value?"*
- *"Show me the distribution of order amounts"*
- *"What percentage of customers churned by plan type?"*

## Security Guarantees

The sandbox provides a defense-in-depth approach:

1. **Layer 1 — AST Static Analysis**: Code is parsed into an Abstract Syntax Tree before execution. Any dangerous import, attribute access, or builtin call is rejected immediately with no subprocess spawned.

2. **Layer 2 — Subprocess Isolation**: Validated code runs in a separate Python process. The main process passes only a pickled DataFrame via stdin; the subprocess can only write JSON to stdout.

3. **Layer 3 — Hard Timeout**: `subprocess.run(timeout=5)` kills any process exceeding 5 seconds, preventing infinite loops and resource exhaustion.

4. **Layer 4 — Restricted Namespace**: Code is executed via `exec()` in a minimal namespace containing only `df`, `pd`, `go`, `px`, and `np` — no filesystem, no network, no builtins.

## License

MIT
