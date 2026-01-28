"""
AWS utility functions for S3 operations and Lambda response formatting.

Consolidates S3 operations and response building for simpler imports.
"""

import json
import boto3
from datetime import datetime
from typing import Tuple, Dict, List, Any, Optional


def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """Parse S3 URI into bucket and key components."""
    if not s3_uri.startswith("s3://"):
        raise ValueError("S3 URI must start with 's3://'")

    uri_parts = s3_uri[5:].split('/', 1)
    bucket = uri_parts[0]
    key = uri_parts[1] if len(uri_parts) > 1 else ""
    return bucket, key


def get_s3_csv_content(bucket: str, key: str, s3_client=None) -> str:
    """Download CSV file from S3 and decode to string."""
    if s3_client is None:
        s3_client = boto3.client('s3')

    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return response['Body'].read().decode('utf-8')
    except Exception as e:
        raise Exception(f"Failed to read from S3 (bucket={bucket}, key={key}): {str(e)}")


def put_s3_object(bucket: str, key: str, data: str, s3_client=None) -> dict:
    """Upload data to S3."""
    if s3_client is None:
        s3_client = boto3.client('s3')

    try:
        return s3_client.put_object(Bucket=bucket, Key=key, Body=data)
    except Exception as e:
        raise Exception(f"Failed to upload to S3 (bucket={bucket}, key={key}): {str(e)}")


def generate_s3_key(prefix: str = "covid_data") -> str:
    """Generate timestamped S3 key path."""
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"{prefix}/{timestamp}/data.json"


def build_success_response(data: Dict[str, Any], s3_location: str) -> Dict[str, Any]:
    """Build Lambda success response (HTTP 200)."""
    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Data successfully ingested",
            "s3_key": s3_location,
            "record_count": len(data.get("records", []))
        })
    }


def build_error_response(error: Exception, status_code: int = 500) -> Dict[str, Any]:
    """Build Lambda error response."""
    return {
        "statusCode": status_code,
        "body": json.dumps({"error": str(error)})
    }


def build_data_envelope(
    source: str,
    records: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Wrap records with metadata envelope."""
    envelope = {
        "source": source,
        "timestamp": datetime.utcnow().isoformat(),
        "record_count": len(records),
        "records": records
    }

    if metadata:
        envelope.update(metadata)

    return envelope
