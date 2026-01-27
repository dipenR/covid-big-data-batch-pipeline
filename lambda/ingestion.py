"""
Lambda handler for COVID data ingestion.

Simplified POC version with inline data source handlers.
Refactored from 289-line monolithic file to use reusable utilities.
"""

import json
import boto3
import os
import tempfile
from kaggle.api.kaggle_api_extended import KaggleApi

from common.csv_parser import parse_csv_file, parse_csv_string
from common.aws_utils import (
    parse_s3_uri, get_s3_csv_content, put_s3_object, generate_s3_key,
    build_success_response, build_error_response, build_data_envelope
)

s3_client = boto3.client("s3")

def get_env_variable(key: str, required: bool = False):
    value = os.environ.get(key)
    if required and value is None:
        raise EnvironmentError(f"Required environment variable '{key}' is not set")
    return value


def fetch_from_kaggle(dataset_identifier: str, specific_file: str = None):
    """Fetch dataset from Kaggle."""
    if not dataset_identifier:
        raise ValueError("kaggle_dataset parameter is required")

    # Initialize and authenticate Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Use temp directory for downloads
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Downloading Kaggle dataset: {dataset_identifier}")

        # Download and unzip dataset
        api.dataset_download_files(dataset_identifier, path=temp_dir, unzip=True)

        # Find CSV files
        csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
        if not csv_files:
            raise ValueError("No CSV files found in Kaggle dataset")

        # Use specific file or first CSV
        target_file = specific_file if specific_file in csv_files else csv_files[0]
        file_path = os.path.join(temp_dir, target_file)

        # Parse CSV
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            records = parse_csv_file(csvfile)

        # Return data envelope
        return build_data_envelope(
            source="kaggle",
            records=records,
            metadata={"dataset": dataset_identifier, "file": target_file}
        )


def fetch_from_local(file_path: str):
    """Read CSV file from local file system."""
    # Resolve relative paths from project root
    if not os.path.isabs(file_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(project_root, file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found at {file_path}")

    # Parse CSV
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        records = parse_csv_file(csvfile)

    # Return data envelope
    return build_data_envelope(
        source="test_data",
        records=records,
        metadata={"dataset": os.path.basename(file_path).replace('.csv', '')}
    )


def fetch_from_s3(s3_uri: str):
    """Read CSV file from S3."""
    bucket, key = parse_s3_uri(s3_uri)
    print(f"Fetching from S3: bucket={bucket}, key={key}")

    # Download and parse CSV
    csv_content = get_s3_csv_content(bucket, key, s3_client)
    records = parse_csv_string(csv_content)

    # Return data envelope
    return build_data_envelope(
        source="test_data_s3",
        records=records,
        metadata={"dataset": os.path.basename(key).replace('.csv', '')}
    )


def lambda_handler(event, context=None):
    """
    Orchestrates data ingestion from various sources.

    Args:
        event: Lambda event dict containing:
            - source: Data source type ("production", "kaggle", or "test")
            - kaggle_dataset: (if source="kaggle") Dataset identifier
            - kaggle_file: (optional) Specific CSV file name
            - test_data_location: (if source="test") File path or S3 URI
        context: Lambda context object (optional)

    Returns:
        Lambda response dict with statusCode and body

    Environment Variables:
        RAW_DATA_BUCKET: S3 bucket for storing raw data
        KAGGLE_USERNAME: Kaggle API username
        KAGGLE_KEY: Kaggle API key
    """
    try:
        source_type = event.get("source", "test")

        # Route to appropriate data source handler
        if source_type == "production":
            raise NotImplementedError(
                "Production data source not yet implemented. "
                "Future: Twitter API, live feeds, etc."
            )

        elif source_type == "kaggle":
            data = fetch_from_kaggle(
                dataset_identifier=event.get("kaggle_dataset"),
                specific_file=event.get("kaggle_file")
            )

        else:
            # Test data source (local file or S3)
            file_path = event.get("test_data_location", "data/covidvaccine.csv")

            if file_path.startswith("s3://"):
                data = fetch_from_s3(s3_uri=file_path)
            else:
                data = fetch_from_local(file_path=file_path)

        # Store fetched data to S3
        bucket = get_env_variable('RAW_DATA_BUCKET', required=True)
        key = generate_s3_key()
        put_s3_object(bucket, key, json.dumps(data), s3_client)

        # Build and return success response
        s3_location = f"s3://{bucket}/{key}"
        return build_success_response(data, s3_location)

    except Exception as e:
        print(f"Error: {str(e)}")
        return build_error_response(e)


if __name__ == "__main__":
    """Local testing block."""
    # Set test environment variables
    os.environ["RAW_DATA_BUCKET"] = "covid-project-raw-data"

    # Test event
    event = {
        "source": "test",
        "test_data_location": "data/covidvaccine.csv"
    }

    # Execute lambda handler
    result = lambda_handler(event, {})
    print(json.dumps(result, indent=2))
