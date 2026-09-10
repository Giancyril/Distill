"""
core/query_engine.py — Natural Language → Pandas/Plotly Code via OpenAI.
Builds a minimal schema prompt, calls GPT, validates response structure,
and returns executable code ready for core.sandbox.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """
    Holds the structured output of a natural language query.
    The caller is responsible for executing generated_code via sandbox.
    """
    user_question: str
    generated_code: str            # Pandas/Plotly Python code string
    chart_intent: str              # e.g. "bar_chart", "line_chart", "table"
    natural_language_answer: str   # Pre-execution plain-text plan / answer
    schema_used: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    is_mock: bool = False          # True when running in offline/mock mode


@dataclass
class ColumnSchema:
    """Minimal representation of a single column for the LLM prompt."""
    name: str
    inferred_type: str
    non_null_count: int
    sample_values: List[Any]


# ---------------------------------------------------------------------------
# Schema Extraction
# ---------------------------------------------------------------------------

def extract_schema(
    df: pd.DataFrame,
    inferred_types: Optional[Dict[str, str]] = None,
    max_sample_values: int = 3,
) -> List[ColumnSchema]:
    """
    Extract a minimal, privacy-safe schema summary from a DataFrame.
    Never includes full data — only column names, types, and a few samples.
    """
    schemas = []
    for col in df.columns:
        series = df[col]
        col_type = (inferred_types or {}).get(col, str(series.dtype))
        non_null_count = int(series.notna().sum())
        samples = series.dropna().head(max_sample_values).tolist()
        # Truncate long string samples
        samples = [
            str(v)[:60] if isinstance(v, str) else v
            for v in samples
        ]
        schemas.append(ColumnSchema(
            name=str(col),
            inferred_type=col_type,
            non_null_count=non_null_count,
            sample_values=samples,
        ))
    return schemas


def _format_schema_for_prompt(
    schemas: List[ColumnSchema],
    total_rows: int,
) -> str:
    """Format schema list as a compact markdown table for injection into the prompt."""
    lines = [
        f"Dataset: {total_rows:,} rows × {len(schemas)} columns\n",
        "| Column | Type | Non-Null | Samples |",
        "|--------|------|----------|---------|",
    ]
    for s in schemas:
        samples_str = ", ".join(str(v) for v in s.sample_values[:3])
        lines.append(
            f"| {s.name} | {s.inferred_type} | {s.non_null_count} | {samples_str} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert data analyst. Given a dataset schema and a user question, generate a Python code snippet that answers the question using pandas and plotly.

RULES:
1. Use df as the DataFrame variable (already available).
2. For charts, assign the plotly figure to _result.
3. For tables, assign a filtered/aggregated DataFrame to _result.
4. For scalar answers (totals, counts), assign the value to _result.
5. Never import os, sys, subprocess, socket, shutil, pickle, or any file/network library.
6. Never use open(), eval(), exec(), or __import__().
7. Keep code concise — prefer 3–10 lines.
8. Use plotly.express (px) for visualization, not matplotlib.
9. Apply a clean indigo color scheme: primary '#6366F1', secondary '#8B5CF6'.

Respond ONLY with valid JSON in this exact format:
{
  "code": "<python code string>",
  "chart_intent": "<bar_chart|line_chart|scatter|pie_chart|histogram|box_plot|table|scalar>",
  "answer": "<brief plain-English explanation of what the code computes>"
}"""


def _build_user_prompt(schema_text: str, question: str) -> str:
    return f"""## Dataset Schema
{schema_text}

## User Question
{question}

Generate the analysis code and explanation."""


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

def _call_openai(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call OpenAI and return the raw content string."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=800,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


def _parse_llm_response(raw: str) -> Dict[str, str]:
    """Parse and validate the LLM JSON response."""
    # Strip markdown code fences if present
    raw = re.sub(r"`(?:json)?\s*", "", raw).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\nRaw: {raw[:300]}")

    required = {"code", "chart_intent", "answer"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"LLM response missing required keys: {missing}")

    return data


# ---------------------------------------------------------------------------
# Mock Fallback (offline / no API key)
# ---------------------------------------------------------------------------

def _mock_query(question: str, schemas: List[ColumnSchema]) -> QueryResult:
    """
    Generate a safe mock response when OpenAI is unavailable.
    Returns a descriptive statistics table as the default result.
    """
    code = "import pandas as pd\n_result = df.describe(include='all').T.reset_index()"
    return QueryResult(
        user_question=question,
        generated_code=code,
        chart_intent="table",
        natural_language_answer=(
            "OpenAI API key not configured. Showing descriptive statistics for all columns. "
            "Set OPENAI_API_KEY in your .env file to enable natural language queries."
        ),
        is_mock=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_query(
    question: str,
    df: pd.DataFrame,
    inferred_types: Optional[Dict[str, str]] = None,
    model: str = "gpt-4o-mini",
) -> QueryResult:
    """
    Translate a natural language question into executable Pandas/Plotly code.

    Parameters
    ----------
    question : str
        The user's plain-English question about the dataset.
    df : pd.DataFrame
        The active dataset.
    inferred_types : dict, optional
        Column type map from core.cleaning.detect_column_types.
    model : str
        OpenAI model to use (default: gpt-4o-mini for cost efficiency).

    Returns
    -------
    QueryResult
        Contains generated_code ready for sandbox execution.
    """
    if not _OPENAI_AVAILABLE or not os.getenv("OPENAI_API_KEY"):
        schemas = extract_schema(df, inferred_types)
        return _mock_query(question, schemas)

    schemas = extract_schema(df, inferred_types)
    schema_text = _format_schema_for_prompt(schemas, total_rows=len(df))
    user_prompt = _build_user_prompt(schema_text, question)

    try:
        raw_response = _call_openai(SYSTEM_PROMPT, user_prompt, model=model)
        parsed = _parse_llm_response(raw_response)
    except Exception as e:
        return QueryResult(
            user_question=question,
            generated_code="",
            chart_intent="none",
            natural_language_answer="",
            error=str(e),
        )

    return QueryResult(
        user_question=question,
        generated_code=parsed["code"],
        chart_intent=parsed["chart_intent"],
        natural_language_answer=parsed["answer"],
        schema_used={s.name: s.inferred_type for s in schemas},
    )
