"""
Unit tests for data ingestion pipeline.
"""
import io
import pytest
import pandas as pd
from core.ingestion import load_dataset, IngestionError


def test_load_clean_csv():
    df = load_dataset("sample_data/sales_clean.csv")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 250
    assert "order_id" in df.columns
    assert "revenue" in df.columns


def test_load_messy_csv():
    df = load_dataset("sample_data/customer_churn_messy.csv")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 8
    assert "customer_id" in df.columns


def test_load_bytesio_buffer():
    csv_bytes = b"name,score,passed\nAlice,95,True\nBob,82,True\nCharlie,54,False\n"
    buffer = io.BytesIO(csv_bytes)
    setattr(buffer, "name", "grades.csv")
    df = load_dataset(buffer)
    assert len(df) == 3
    assert list(df.columns) == ["name", "score", "passed"]


def test_load_nonexistent_file():
    with pytest.raises(IngestionError, match="File not found"):
        load_dataset("nonexistent_path_xyz.csv")


def test_load_empty_csv():
    buffer = io.BytesIO(b"")
    setattr(buffer, "name", "empty.csv")
    with pytest.raises(IngestionError, match="Could not determine delimiter|contains no tabular records|empty"):
        load_dataset(buffer)

