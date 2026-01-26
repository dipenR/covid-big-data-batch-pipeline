import json
import boto3
import os
from datetime import datetime

s3_client = boto3.client("s3")
BUCKET_NAME = os.environ["RAW_DATA_BUCKET"]


def lambda_handler(event, context):
    try:
        if event.get("source") == "production":
            data = fetch_production_data()
        else:
            data = fetch_test_data()

        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        key = f"covid_data/{timestamp}/data.json"

        s3_client.put_object(Bucket=BUCKET_NAME, Key=key, Body=json.dumps(data))

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "Data successfully ingested",
                    "s3_key": f"s3://{BUCKET_NAME}/{key}",
                }
            ),
        }
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}


def fetch_production_data():
    pass  # Implementation for fetching production data


def fetch_test_data():
    # TODO : replace with kaggle API fetch OR download and fetch file
    pass
