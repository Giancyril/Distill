"""
core/sandbox.py — Secure code execution via AST allowlisting + subprocess isolation.
LLM-generated Pandas/Plotly code is validated and executed in a restricted subprocess.
"""
from __future__ import annotations

import ast
import json
import pickle
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# AST Security Configuration
# ---------------------------------------------------------------------------

BLOCKED_IMPORTS: set[str] = {
    "os", "sys", "subprocess", "shutil", "socket", "pickle", "builtins",
    "importlib", "ctypes", "multiprocessing", "threading", "asyncio",
    "http", "urllib", "requests", "pathlib", "io", "glob", "tempfile",
    "signal", "resource", "gc", "weakref", "inspect", "traceback",
}

BLOCKED_ATTRIBUTES: set[str] = {
    "__import__", "__subclasses__", "__globals__", "__builtins__",
    "__class__", "__bases__", "__mro__", "__dict__", "__code__",
    "__closure__", "__reduce__", "__reduce_ex__", "mro",
    "func_globals", "gi_frame", "cr_frame", "ag_frame",
}

BLOCKED_BUILTINS: set[str] = {
    "open", "eval", "exec", "compile", "__import__", "breakpoint",
    "input",
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class SandboxResult:
    """Result of a sandboxed code execution."""
    success: bool
    result_json: Optional[str] = None
    result_type: str = "none"
    error_message: Optional[str] = None
    blocked_reason: Optional[str] = None
    executed_code: str = ""
    execution_time_ms: int = 0


# ---------------------------------------------------------------------------
# AST Validator
# ---------------------------------------------------------------------------

class ASTSecurityVisitor(ast.NodeVisitor):
    """Walk the AST and raise SecurityError on any disallowed pattern."""

    def __init__(self):
        self.violations: List[str] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            top_pkg = alias.name.split(".")[0]
            if top_pkg in BLOCKED_IMPORTS:
                self.violations.append(f"Blocked import: '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = (node.module or "").split(".")[0]
        if module in BLOCKED_IMPORTS:
            self.violations.append(f"Blocked import from: '{node.module}'")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr in BLOCKED_ATTRIBUTES:
            self.violations.append(f"Blocked attribute access: '.{node.attr}'")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_BUILTINS:
            self.violations.append(f"Blocked builtin call: '{node.func.id}()'")
        if isinstance(node.func, ast.Name) and node.func.id == "__import__":
            self.violations.append("Blocked __import__() call")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global):
        self.generic_visit(node)


def validate_ast(code: str) -> Optional[str]:
    """
    Parse and statically analyze code for security violations.
    Returns None if code is safe, or a string describing the violation.
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        return f"SyntaxError: {e}"

    visitor = ASTSecurityVisitor()
    visitor.visit(tree)

    if visitor.violations:
        return "; ".join(visitor.violations)
    return None


# ---------------------------------------------------------------------------
# Subprocess Runner
# ---------------------------------------------------------------------------

def _build_runner_script(user_code: str) -> str:
    """
    Build a self-contained Python script that:
    1. Reads a pickled DataFrame from stdin
    2. Executes user_code in a controlled exec() environment
    3. Prints results as JSON to stdout
    """
    # Escape the user code for embedding in the script
    user_code_repr = repr(user_code)

    script = f"""
import sys, json, pickle, pandas as pd

# Load DataFrame from stdin
df_bytes = sys.stdin.buffer.read()
df = pickle.loads(df_bytes)

# Prepare execution namespace
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

_ns = dict(df=df, pd=pd, go=go, px=px, np=np)
_user_code = {user_code_repr}

try:
    exec(_user_code, _ns)
except Exception as e:
    print(json.dumps({{"type": "error", "data": str(e)}}))
    sys.exit(0)

_result = _ns.get("_result")
_fig = None
_table = None

try:
    import plotly.basedatatypes as _base
    if isinstance(_result, _base.BaseFigure):
        _fig = _result
    elif isinstance(_result, pd.DataFrame):
        _table = _result
    elif _result is None:
        for _v in _ns.values():
            if isinstance(_v, _base.BaseFigure):
                _fig = _v
                break
            elif isinstance(_v, pd.DataFrame) and not _v.equals(df):
                _table = _v
                break

    if _fig is not None:
        print(json.dumps({{"type": "chart", "data": _fig.to_json()}}))
    elif _table is not None:
        print(json.dumps({{"type": "table", "data": _table.head(500).to_json(orient="records", date_format="iso")}}))
    elif _result is not None:
        print(json.dumps({{"type": "scalar", "data": str(_result)}}))
    else:
        print(json.dumps({{"type": "none", "data": None}}))
except Exception as e:
    print(json.dumps({{"type": "error", "data": str(e)}}))
"""
    return script


def execute_in_sandbox(
    code: str,
    df: pd.DataFrame,
    timeout_seconds: int = 5,
) -> SandboxResult:
    """
    Validate and execute code in a sandboxed subprocess.

    Parameters
    ----------
    code : str
        Python code string to execute (generated by LLM).
    df : pd.DataFrame
        The active dataset; will be available as `df` inside the code.
    timeout_seconds : int
        Hard wall-clock timeout for the subprocess (default 5s).

    Returns
    -------
    SandboxResult
    """
    import time

    # 1. AST Validation
    violation = validate_ast(code)
    if violation:
        return SandboxResult(
            success=False,
            blocked_reason=violation,
            executed_code=code,
        )

    # 2. Build runner script
    runner_src = _build_runner_script(code)

    # 3. Serialize DataFrame
    df_pickle = pickle.dumps(df, protocol=4)

    # 4. Execute in subprocess
    t_start = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner_src],
            input=df_pickle,
            capture_output=True,
            timeout=timeout_seconds,
        )
        elapsed_ms = int((time.monotonic() - t_start) * 1000)

        if proc.returncode != 0:
            stderr_msg = proc.stderr.decode("utf-8", errors="replace").strip()
            return SandboxResult(
                success=False,
                error_message=stderr_msg or "Subprocess exited with non-zero code.",
                executed_code=code,
                execution_time_ms=elapsed_ms,
            )

        stdout = proc.stdout.decode("utf-8", errors="replace").strip()
        if not stdout:
            return SandboxResult(
                success=False,
                error_message="Subprocess produced no output.",
                executed_code=code,
                execution_time_ms=elapsed_ms,
            )

        output = json.loads(stdout)
        if output["type"] == "error":
            return SandboxResult(
                success=False,
                error_message=output["data"],
                executed_code=code,
                execution_time_ms=elapsed_ms,
            )

        return SandboxResult(
            success=True,
            result_json=output.get("data"),
            result_type=output.get("type", "none"),
            executed_code=code,
            execution_time_ms=elapsed_ms,
        )

    except subprocess.TimeoutExpired:
        return SandboxResult(
            success=False,
            error_message=f"Execution timed out after {timeout_seconds}s.",
            executed_code=code,
            execution_time_ms=timeout_seconds * 1000,
        )
    except Exception as e:
        return SandboxResult(
            success=False,
            error_message=f"Internal sandbox error: {str(e)}",
            executed_code=code,
        )
