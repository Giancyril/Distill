"""
Scaffolding verification tests.
"""
import os
import pandas as pd


def test_project_structure():
    assert os.path.isdir("app")
    assert os.path.isdir("core")
    assert os.path.isdir("tests")
    assert os.path.isdir("sample_data")
    assert os.path.isfile("sample_data/sales_clean.csv")
    assert os.path.isfile("sample_data/customer_churn_messy.csv")


def test_sample_datasets_load():
    df_sales = pd.read_csv("sample_data/sales_clean.csv")
    assert len(df_sales) == 250
    assert "revenue" in df_sales.columns
    assert "profit" in df_sales.columns

    df_churn = pd.read_csv("sample_data/customer_churn_messy.csv")
    assert len(df_churn) == 8
    assert "customer_id" in df_churn.columns
