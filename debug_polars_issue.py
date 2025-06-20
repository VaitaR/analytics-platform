#!/usr/bin/env python3
"""
Diagnostic script for Polars struct.fields() compatibility issues
Tests different approaches to JSON processing in Polars
"""

import logging

import pandas as pd
import polars as pl

from logging_config import quick_debug_setup


def create_test_data():
    """Create test data similar to what the app processes"""
    data = {
        "user_id": ["user_1", "user_2", "user_3"],
        "event_name": ["sign_up", "login", "purchase"],
        "timestamp": [
            "2024-01-01 10:00:00",
            "2024-01-01 11:00:00",
            "2024-01-01 12:00:00",
        ],
        "event_properties": [
            '{"platform": "mobile", "utm_source": "google"}',
            '{"platform": "desktop", "utm_source": "direct"}',
            '{"platform": "mobile", "utm_source": "facebook", "amount": 99.99}',
        ],
        "user_properties": [
            '{"country": "US", "subscription": "free"}',
            '{"country": "UK", "subscription": "premium"}',
            '{"country": "DE", "subscription": "basic", "age": 25}',
        ],
    }
    return pd.DataFrame(data)


def test_polars_struct_fields():
    """Test different approaches to struct.fields() in Polars"""
    logger = logging.getLogger(__name__)

    # Create test data
    df = create_test_data()
    logger.info(f"📊 Created test data with {len(df)} rows")

    # Convert to Polars
    pl_df = pl.from_pandas(df)
    logger.info("🔄 Converted to Polars DataFrame")

    # Test JSON processing for event_properties
    column = "event_properties"
    logger.info(f"🧪 Testing JSON processing for column: {column}")

    try:
        # Filter out nulls
        valid_props = pl_df.filter(pl.col(column).is_not_null())
        logger.debug(f"   Filtered to {valid_props.height} non-null rows")

        # Test JSON decode
        logger.debug("   Attempting JSON decode...")
        decoded = valid_props.select(pl.col(column).str.json_decode().alias("decoded_props"))
        logger.info(f"✅ JSON decode successful, got {decoded.height} rows")

        # Test modern struct.fields() approach
        logger.debug("   Testing modern struct.fields() API...")
        try:
            all_keys = (
                decoded.select(pl.col("decoded_props").struct.fields())
                .to_series()
                .explode()
                .unique()
            )
            logger.info(
                f"✅ Modern struct.fields() successful, found {len(all_keys)} keys: {all_keys.to_list()}"
            )
            return "modern_api", all_keys.to_list()

        except Exception as e:
            logger.warning(f"❌ Modern struct.fields() failed: {str(e)}")
            logger.debug(f"   Error type: {type(e).__name__}")

            # Test fallback approach
            logger.debug("   Testing fallback approach...")
            try:
                sample_struct = decoded.filter(pl.col("decoded_props").is_not_null()).limit(1)
                if not sample_struct.is_empty():
                    first_row = sample_struct.row(0, named=True)
                    if first_row["decoded_props"] is not None:
                        # Get all keys from sample rows
                        all_keys_set = set()
                        for i in range(min(decoded.height, 10)):  # Sample first 10 rows
                            try:
                                row = decoded.row(i, named=True)
                                if row["decoded_props"] is not None:
                                    all_keys_set.update(row["decoded_props"].keys())
                            except Exception as row_e:
                                logger.debug(f"   Row {i} error: {str(row_e)}")
                                continue

                        all_keys = list(all_keys_set)
                        logger.info(
                            f"✅ Fallback approach successful, found {len(all_keys)} keys: {all_keys}"
                        )
                        return "fallback_api", all_keys
                    logger.error("   No valid decoded props found")
                    return "error", []
                logger.error("   No non-null structs found")
                return "error", []

            except Exception as e2:
                logger.error(f"❌ Fallback approach also failed: {str(e2)}")
                return "error", []

    except Exception as e:
        logger.error(f"❌ JSON decode failed: {str(e)}")
        return "error", []


def test_json_schema_inference():
    """Test different JSON schema inference approaches"""
    logger = logging.getLogger(__name__)

    # Create test data with varying JSON schemas
    varying_data = [
        '{"platform": "mobile", "utm_source": "google"}',
        '{"platform": "desktop", "utm_source": "direct", "campaign": "summer"}',
        '{"platform": "mobile", "utm_source": "facebook", "amount": 99.99, "currency": "USD"}',
    ]

    df = pl.DataFrame({"json_col": varying_data})
    logger.info("🧪 Testing schema inference with varying JSON structures")

    # Test different schema inference approaches
    approaches = [
        ("default", {}),
        ("large_window", {"infer_schema_length": 50000}),
        ("no_inference", {"infer_schema_length": None}),
        ("minimal", {"infer_schema_length": 1}),
    ]

    for approach_name, kwargs in approaches:
        logger.debug(f"   Testing {approach_name} approach...")
        try:
            decoded = df.select(pl.col("json_col").str.json_decode(**kwargs).alias("decoded"))
            logger.info(f"✅ {approach_name} schema inference successful")

            # Try to get fields
            try:
                fields = (
                    decoded.select(pl.col("decoded").struct.fields())
                    .to_series()
                    .explode()
                    .unique()
                )
                logger.info(f"   Fields extracted: {fields.to_list()}")
            except Exception as fe:
                logger.debug(f"   Fields extraction failed: {str(fe)}")

        except Exception as e:
            logger.debug(f"   {approach_name} failed: {str(e)}")


def main():
    """Main diagnostic function"""
    # Setup enhanced logging
    logger = quick_debug_setup()
    logger.info("🔍 Starting Polars compatibility diagnostics")

    # Test basic struct.fields() functionality
    method, keys = test_polars_struct_fields()
    logger.info(f"📋 Result: {method} method worked, found keys: {keys}")

    # Test schema inference
    test_json_schema_inference()

    # Test with the actual app data source manager
    logger.info("🧪 Testing with actual DataSourceManager...")
    try:
        from app import DataSourceManager

        data_manager = DataSourceManager()
        sample_data = data_manager.get_sample_data()
        logger.info(f"✅ Got sample data with {len(sample_data)} rows")

        # Test segmentation properties extraction
        properties = data_manager.get_segmentation_properties(sample_data)
        logger.info(f"✅ Extracted properties: {properties}")

    except Exception as e:
        logger.error(f"❌ DataSourceManager test failed: {str(e)}")
        logger.error(f"   Error type: {type(e).__name__}")

    logger.info("🏁 Diagnostics completed")


if __name__ == "__main__":
    main()
