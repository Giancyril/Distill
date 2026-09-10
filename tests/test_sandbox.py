"""
tests/test_sandbox.py — Security tests for core/sandbox.py
Verifies that dangerous code patterns are blocked by AST validation.
"""
import pandas as pd
import pytest

from core.sandbox import SandboxResult, execute_in_sandbox, validate_ast


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "product": ["Alpha", "Beta", "Gamma"],
        "revenue": [1000, 2500, 750],
        "units": [10, 25, 8],
    })


# ---------------------------------------------------------------------------
# AST Validation Tests
# ---------------------------------------------------------------------------

class TestASTValidation:
    def test_clean_code_passes(self):
        code = "_result = df['revenue'].sum()"
        assert validate_ast(code) is None

    def test_import_os_blocked(self):
        code = "import os\nos.system('rm -rf /')"
        violation = validate_ast(code)
        assert violation is not None
        assert "os" in violation

    def test_import_sys_blocked(self):
        violation = validate_ast("import sys")
        assert violation is not None

    def test_import_subprocess_blocked(self):
        violation = validate_ast("import subprocess")
        assert violation is not None

    def test_import_socket_blocked(self):
        violation = validate_ast("import socket")
        assert violation is not None

    def test_from_os_import_blocked(self):
        violation = validate_ast("from os import system")
        assert violation is not None

    def test_dunder_subclasses_blocked(self):
        violation = validate_ast("x = ().__class__.__bases__[0].__subclasses__()")
        assert violation is not None

    def test_dunder_import_blocked(self):
        violation = validate_ast("__import__('os').system('id')")
        assert violation is not None

    def test_open_builtin_blocked(self):
        violation = validate_ast("f = open('/etc/passwd', 'r')")
        assert violation is not None

    def test_eval_blocked(self):
        violation = validate_ast("eval('import os')")
        assert violation is not None

    def test_exec_blocked(self):
        violation = validate_ast("exec('import subprocess')")
        assert violation is not None

    def test_plotly_import_allowed(self):
        code = "import plotly.express as px\n_result = px.bar(df, x='product', y='revenue')"
        assert validate_ast(code) is None

    def test_numpy_import_allowed(self):
        code = "import numpy as np\n_result = np.mean(df['revenue'])"
        assert validate_ast(code) is None

    def test_syntax_error_caught(self):
        violation = validate_ast("def broken(: pass")
        assert violation is not None
        assert "SyntaxError" in violation


# ---------------------------------------------------------------------------
# Sandbox Execution Tests
# ---------------------------------------------------------------------------

class TestSandboxExecution:
    def test_safe_scalar(self, sample_df):
        result = execute_in_sandbox("_result = df['revenue'].sum()", sample_df)
        assert result.success is True
        assert result.result_type == "scalar"
        assert "4250" in str(result.result_json)

    def test_table_result(self, sample_df):
        code = "_result = df.groupby('product')['revenue'].sum().reset_index()"
        result = execute_in_sandbox(code, sample_df)
        assert result.success is True
        assert result.result_type == "table"

    def test_blocked_os_import(self, sample_df):
        code = "import os\nos.system('echo hacked')"
        result = execute_in_sandbox(code, sample_df)
        assert result.success is False
        assert result.blocked_reason is not None

    def test_blocked_open(self, sample_df):
        code = "f = open('/etc/passwd'); _result = f.read()"
        result = execute_in_sandbox(code, sample_df)
        assert result.success is False

    def test_blocked_eval(self, sample_df):
        result = execute_in_sandbox("_result = eval('1+1')", sample_df)
        assert result.success is False

    def test_timeout_enforced(self, sample_df):
        code = "while True: pass"
        result = execute_in_sandbox(code, sample_df, timeout_seconds=2)
        assert result.success is False
        assert "timed out" in (result.error_message or "").lower()

    def test_result_contains_code(self, sample_df):
        code = "_result = len(df)"
        result = execute_in_sandbox(code, sample_df)
        assert result.executed_code == code

    def test_execution_time_recorded(self, sample_df):
        result = execute_in_sandbox("_result = df.shape", sample_df)
        assert result.execution_time_ms >= 0
