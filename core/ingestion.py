"""
Data Ingestion Module.
Robust loading for CSV and Excel files with encoding detection and validation.
"""
from __future__ import annotations

import io
import os
from typing import Union, BinaryIO, Optional
import pandas as pd


MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB default safeguard


class IngestionError(Exception):
    """Raised when data ingestion fails validation or parsing."""
    pass


def load_dataset(
    source: Union[str, BinaryIO, io.BytesIO],
    filename: Optional[str] = None,
    max_size_bytes: int = MAX_FILE_SIZE_BYTES
) -> pd.DataFrame:
    """
    Loads tabular data from a file path or file-like buffer into a pandas DataFrame.
    Supports CSV and Excel (.xlsx, .xls) with encoding sniffing and safe size limits.
    """
    # 1. Size and format sniffing
    name = filename or (source if isinstance(source, str) else getattr(source, "name", "dataset.csv"))
    name_lower = str(name).lower()

    if isinstance(source, str):
        if not os.path.exists(source):
            raise IngestionError(f"File not found: {source}")
        file_size = os.path.getsize(source)
        if file_size > max_size_bytes:
            raise IngestionError(
                f"File size ({file_size / (1024 * 1024):.1f} MB) exceeds maximum allowed {max_size_bytes / (1024 * 1024):.0f} MB."
            )

    # 2. Parse Excel
    if name_lower.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(source, engine="openpyxl" if name_lower.endswith(".xlsx") else None)
            if df.empty:
                raise IngestionError("Uploaded Excel sheet contains no records.")
            return df
        except IngestionError:
            raise
        except Exception as e:
            raise IngestionError(f"Failed to parse Excel workbook: {e}") from e

    # 3. Parse CSV with encoding fallbacks
    encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    last_error = None

    for enc in encodings_to_try:
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            
            # Use sep=None with engine='python' for automatic delimiter detection (comma, tab, semicolon)
            df = pd.read_csv(
                source,
                encoding=enc,
                sep=None,
                engine="python",
                on_bad_lines="skip"
            )
            
            if df.empty:
                raise IngestionError("Uploaded CSV file contains no tabular records.")
            
            # Strip whitespace from column names automatically
            df.columns = [str(col).strip() for col in df.columns]
            return df
        except IngestionError:
            raise
        except Exception as e:
            last_error = e
            continue

    raise IngestionError(f"Failed to decode CSV file with supported encodings. Last error: {last_error}")
