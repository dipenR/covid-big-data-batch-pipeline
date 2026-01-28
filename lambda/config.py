"""
Configuration management for Lambda functions.

This module provides centralized configuration management, handling environment
variables with proper error handling and validation.

"""

import os
from typing import Tuple, Optional


def get_env_variable(key: str, default: Optional[str] = None, required: bool = False) -> str:
    """
    Safely get environment variable with optional default and required validation.

    Args:
        key: Environment variable name
        default: Default value if variable not set (default: None)
        required: If True, raises exception when variable not found (default: False)

    Returns:
        Environment variable value or default

    Raises:
        EnvironmentError: If required=True and variable not found

    Example:
        >>> bucket = get_env_variable('S3_BUCKET', required=True)
        >>> optional_val = get_env_variable('OPTIONAL_VAR', default='default_value')
    """

    value = os.environ.get(key, default)

    if required and value is None:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Please set this variable before running the Lambda function."
        )

    return value


def get_s3_bucket() -> str:
    """
    Get raw data S3 bucket name from environment.

    Returns:
        S3 bucket name for raw data storage

    Raises:
        EnvironmentError: If RAW_DATA_BUCKET not set

    Example:
        >>> bucket = get_s3_bucket()
        'covid-project-raw-data'
    """
    return get_env_variable('RAW_DATA_BUCKET', required=True)


def get_kaggle_credentials() -> Tuple[str, str]:
    """
    Get Kaggle username and API key from environment.

    Returns:
        Tuple of (username, api_key)

    Raises:
        EnvironmentError: If KAGGLE_USERNAME or KAGGLE_KEY not set

    Example:
        >>> username, api_key = get_kaggle_credentials()
    """
    # FIX: This properly fixes the bug from the original ingestion.py line 13
    # where os.environ.get['KAGGLE_KEY'] was incorrectly used
    username = get_env_variable('KAGGLE_USERNAME', required=True)
    api_key = get_env_variable('KAGGLE_KEY', required=True)

    return username, api_key
