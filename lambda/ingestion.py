"""
Lambda handler for COVID data ingestion.

Uploads raw CSV files to S3 without parsing (parsing handled by Spark).
Optimized for performance and Lambda resource limits.
"""

import json
import boto3
import os
import traceback
from kaggle.api.kaggle_api_extended import KaggleApi

from aws_utils import parse_s3_uri

s3_client = boto3.client("s3")

def get_env_variable(key: str, required: bool = False):
    value = os.environ.get(key)
    if required and value is None:
        raise EnvironmentError(f"Required environment variable '{key}' is not set")
    return value


def fetch_from_kaggle(dataset_identifier: str, specific_file: str = None):
    """Fetch dataset from Kaggle and return file path.

    Returns dict with file_path for uploading raw CSV to S3.
    """
    if not dataset_identifier:
        raise ValueError("kaggle_dataset parameter is required")

    # Initialize and authenticate Kaggle API
    api = KaggleApi()
    api.authenticate()

    temp_dir = "/tmp/kaggle_data"
    os.makedirs(temp_dir, exist_ok=True)

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

    print(f"Downloaded CSV: {target_file} ({os.path.getsize(file_path) / 1024 / 1024:.2f} MB)")

    # Return file path instead of parsing - let Spark handle parsing
    return {
        "file_path": file_path,
        "file_name": target_file,
        "source": "kaggle",
        "metadata": {
            "dataset": dataset_identifier,
            "file_size_mb": os.path.getsize(file_path) / 1024 / 1024
        }
    }


def fetch_from_local(file_path: str):
    """Read CSV file from local file system and return file path."""
    # Resolve relative paths from project root
    if not os.path.isabs(file_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(project_root, file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found at {file_path}")

    print(f"Using local CSV: {file_path} ({os.path.getsize(file_path) / 1024 / 1024:.2f} MB)")

    # Return file path instead of parsing
    return {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "source": "test_local",
        "metadata": {
            "dataset": os.path.basename(file_path).replace('.csv', ''),
            "file_size_mb": os.path.getsize(file_path) / 1024 / 1024
        }
    }


def fetch_from_s3(s3_uri: str):
    """Download CSV file from S3 to local temp and return file path."""
    bucket, key = parse_s3_uri(s3_uri)
    print(f"Downloading from S3: bucket={bucket}, key={key}")

    # Download to /tmp
    local_path = f"/tmp/{os.path.basename(key)}"
    s3_client.download_file(bucket, key, local_path)

    print(f"Downloaded CSV: {os.path.basename(key)} ({os.path.getsize(local_path) / 1024 / 1024:.2f} MB)")

    # Return file path instead of parsing
    return {
        "file_path": local_path,
        "file_name": os.path.basename(key),
        "source": "test_s3",
        "metadata": {
            "dataset": os.path.basename(key).replace('.csv', ''),
            "file_size_mb": os.path.getsize(local_path) / 1024 / 1024,
            "s3_uri": s3_uri
        }
    }


def lambda_handler(event, context=None):
    """
    Orchestrates data ingestion from various sources and uploads raw CSV to S3.

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
            file_info = fetch_from_kaggle(
                dataset_identifier=event.get("kaggle_dataset"),
                specific_file=event.get("kaggle_file")
            )

        else:
            # Test data source (local file or S3)
            file_path = event.get("test_data_location", "data/covidvaccine.csv")

            if file_path.startswith("s3://"):
                file_info = fetch_from_s3(s3_uri=file_path)
            else:
                file_info = fetch_from_local(file_path=file_path)

        # Upload raw CSV file to S3 (not JSON!)
        bucket = get_env_variable('RAW_DATA_BUCKET', required=True)
        s3_key = f"raw/{file_info['file_name']}"

        print(f"Uploading to S3: s3://{bucket}/{s3_key}")
        s3_client.upload_file(file_info['file_path'], bucket, s3_key)

        # Build success response
        s3_location = f"s3://{bucket}/{s3_key}"
        response_body = {
            "message": "Data ingestion successful",
            "s3_location": s3_location,
            "source": file_info['source'],
            "file_name": file_info['file_name'],
            "file_size_mb": file_info['metadata']['file_size_mb'],
            "metadata": file_info['metadata']
        }

        print(f"Upload complete: {s3_location}")
        return {
            "statusCode": 200,
            "body": json.dumps(response_body)
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e),
                "error_type": type(e).__name__
            })
        }


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
