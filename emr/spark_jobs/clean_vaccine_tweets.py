#!/usr/bin/env python3
"""
Clean the COVID Vaccine Tweets dataset (Kaggle: covidvaccine-tweets) with Spark.

Reads a raw CSV from S3 (or HDFS/local), performs common cleanup:
- Parse timestamps (format: dd-MM-yyyy HH:mm)
- Cast numeric columns to long
- Robustly cast booleans (true/false/0/1/yes/no)
- Clean tweet text (remove URLs, @mentions, normalize whitespace, lowercase)
- Parse hashtags string like "['Tag1','Tag2']" into array<string>
- Drop invalid rows (null timestamp / empty text), remove duplicates
- Write partitioned Parquet to S3

Usage (EMR):
spark-submit --deploy-mode cluster clean_vaccine_tweets.py \
  --input s3://YOUR_BUCKET/raw/covidvaccine.csv \
  --output s3://YOUR_BUCKET/clean/covidvaccine_parquet/

Or via EMR Step with args appended after script path.
"""

import argparse
from pyspark.sql import SparkSession, functions as F


def to_bool(col):
    """Robust boolean parser for strings like True/False, true/false, 1/0, yes/no."""
    c = F.lower(col.cast("string"))
    return (
        F.when(c.isin("true", "t", "1", "yes", "y"), F.lit(True))
         .when(c.isin("false", "f", "0", "no", "n"), F.lit(False))
         .otherwise(F.lit(None).cast("boolean"))
    )


def main():
    parser = argparse.ArgumentParser(description="Spark cleanup job for COVID Vaccine Tweets dataset")
    parser.add_argument("--input", required=False, help="Input CSV path (e.g., s3://bucket/raw/covidvaccine.csv)")
    parser.add_argument("--output", required=False, help="Output folder for Parquet (e.g., s3://bucket/clean/...) ")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("clean-covidvaccine-tweets").getOrCreate()

    # Allow passing paths either via CLI args OR --conf spark.etl.input/output
    input_path = args.input or spark.sparkContext.getConf().get("spark.etl.input", None)
    output_path = args.output or spark.sparkContext.getConf().get("spark.etl.output", None)

    if not input_path or not output_path:
        raise ValueError(
            "Missing input/output. Provide --input and --output OR set --conf spark.etl.input / spark.etl.output"
        )

    # Read CSV (tweets/text can include commas/quotes/newlines)
    df = (
        spark.read.option("header", True)
        .option("multiLine", True)
        .option("escape", '"')
        .option("quote", '"')
        .option("mode", "PERMISSIVE")
        .csv(input_path)
    )

    # Parse timestamps (dataset format: dd-MM-yyyy HH:mm)
    df = (
        df.withColumn("tweet_ts", F.to_timestamp("date", "dd-MM-yyyy HH:mm"))
          .withColumn("user_created_ts", F.to_timestamp("user_created", "dd-MM-yyyy HH:mm"))
    )

    # Cast numerics
    for colname in ["user_followers", "user_friends", "user_favourites"]:
        if colname in df.columns:
            df = df.withColumn(colname, F.col(colname).cast("long"))

    # Cast booleans robustly
    if "user_verified" in df.columns:
        df = df.withColumn("user_verified", to_bool(F.col("user_verified")))
    if "is_retweet" in df.columns:
        df = df.withColumn("is_retweet", to_bool(F.col("is_retweet")))

    # Clean tweet text
    if "text" in df.columns:
        df = df.withColumn("text_raw", F.col("text"))
        df = df.withColumn(
            "text_clean",
            F.lower(
                F.trim(
                    F.regexp_replace(
                        F.regexp_replace(
                            F.regexp_replace(F.col("text"), r"https?://\S+|www\.\S+", " "),
                            r"@\w+",
                            " ",
                        ),
                        r"\s+",
                        " ",
                    )
                )
            ),
        )

    # Parse hashtags string like "['A','B']" into array<string>
    if "hashtags" in df.columns:
        df = df.withColumn(
            "hashtags_str",
            F.regexp_replace(F.coalesce(F.col("hashtags"), F.lit("")), r"[\[\]']", ""),
        )
        df = df.withColumn("hashtags_arr", F.split(F.col("hashtags_str"), r"\s*,\s*"))
        df = df.withColumn(
            "hashtags_arr",
            F.expr("filter(transform(hashtags_arr, x -> lower(trim(x))), x -> x != '')"),
        )

    # Basic location cleanup (optional but useful)
    if "user_location" in df.columns:
        df = df.withColumn(
            "user_location_clean",
            F.lower(F.trim(F.regexp_replace(F.col("user_location"), r"\s+", " "))),
        )

    # Sanity filters: require timestamp + non-empty text_clean (if present)
    if "tweet_ts" in df.columns:
        df = df.filter(F.col("tweet_ts").isNotNull())
    if "text_clean" in df.columns:
        df = df.filter(F.col("text_clean").isNotNull() & (F.length("text_clean") > 0))

    # Filter negative numeric counts (keep nulls)
    for colname in ["user_followers", "user_friends", "user_favourites"]:
        if colname in df.columns:
            df = df.filter(F.col(colname).isNull() | (F.col(colname) >= 0))

    # Deduplicate (best-effort key)
    dedup_cols = [c for c in ["user_name", "tweet_ts", "text_clean"] if c in df.columns]
    if len(dedup_cols) >= 2:
        df = df.dropDuplicates(dedup_cols)

    # Partition by date for efficient downstream queries
    if "tweet_ts" in df.columns:
        df = df.withColumn("tweet_date", F.to_date(F.col("tweet_ts")))

    # Choose a sensible output column set (keep originals + cleaned fields)
    desired = [
        "user_name",
        "user_location",
        "user_location_clean",
        "user_description",
        "user_created_ts",
        "user_followers",
        "user_friends",
        "user_favourites",
        "user_verified",
        "tweet_ts",
        "tweet_date",
        "text_raw",
        "text_clean",
        "hashtags",
        "hashtags_arr",
        "source",
        "is_retweet",
    ]
    cols = [c for c in desired if c in df.columns]
    out = df.select(*cols) if cols else df

    (
        out.write.mode("overwrite")
        .partitionBy("tweet_date" if "tweet_date" in out.columns else [])
        .parquet(output_path)
    )

    spark.stop()


if __name__ == "__main__":
    main()
