"""
CSV parsing utilities with type inference.

This module provides centralized CSV parsing logic to eliminate code duplication
across data sources. It handles:
- Cleaning keys and values (strip whitespace)
- Automatic type conversion (int, float, string, None)
- Row-level error handling
"""

import csv
from typing import Dict, List, Any, TextIO


def parse_csv_row(row: Dict[str, str]) -> Dict[str, Any]:
    """
    Parse a single CSV row with type inference.

    Args:
        row: Dictionary mapping column names to string values

    Returns:
        Dictionary with cleaned keys and type-converted values

    Example:
        >>> parse_csv_row({'  age  ': '42', 'name': 'John  ', 'score': '98.5'})
        {'age': 42, 'name': 'John', 'score': 98.5}
    """
    record = {}

    for key, value in row.items():
        # Clean the key
        clean_key = key.strip()

        # Clean the value
        clean_value = value.strip() if value else None

        # Try to convert numeric values
        if clean_value and clean_value.replace('.', '').replace('-', '').isdigit():
            try:
                if '.' not in clean_value:
                    record[clean_key] = int(clean_value)
                else:
                    record[clean_key] = float(clean_value)
            except ValueError:
                # If conversion fails, keep as string
                record[clean_key] = clean_value
        else:
            record[clean_key] = clean_value

    return record


def parse_csv_file(file_handle: TextIO, encoding: str = 'utf-8') -> List[Dict[str, Any]]:
    """
    Parse entire CSV file from file handle.

    Args:
        file_handle: File object to read CSV from
        encoding: Character encoding (default: utf-8)

    Returns:
        List of dictionaries, one per CSV row

    Example:
        >>> with open('data.csv', 'r', encoding='utf-8') as f:
        ...     records = parse_csv_file(f)
    """
    records = []
    reader = csv.DictReader(file_handle)

    for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
        try:
            record = parse_csv_row(row)
            records.append(record)
        except Exception as e:
            print(f"Warning: Error processing row {row_num}: {str(e)}")
            continue

    return records


def parse_csv_string(csv_content: str) -> List[Dict[str, Any]]:
    """
    Parse CSV from string content (useful for S3 data).

    Args:
        csv_content: CSV data as string

    Returns:
        List of dictionaries, one per CSV row

    Example:
        >>> csv_data = "name,age\\nJohn,42\\nJane,35"
        >>> records = parse_csv_string(csv_data)
    """
    records = []
    csv_lines = csv_content.splitlines()
    reader = csv.DictReader(csv_lines)

    for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
        try:
            record = parse_csv_row(row)
            records.append(record)
        except Exception as e:
            print(f"Warning: Error processing row {row_num}: {str(e)}")
            continue

    return records
