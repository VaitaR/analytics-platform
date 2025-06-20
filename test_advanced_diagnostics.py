#!/usr/bin/env python3
"""
Test script for advanced diagnostics system
Demonstrates how to use smart logging for better error identification
"""

import pandas as pd
import polars as pl

from advanced_diagnostics import SmartDiagnosticLogger, smart_diagnostic

# Create a diagnostic logger
diagnostic_logger = SmartDiagnosticLogger("test_diagnostics", level=10)  # DEBUG level


@smart_diagnostic(diagnostic_logger)
def problematic_polars_function(df, column_name):
    """Function that will fail with Polars struct.fields() issue"""

    # Convert to Polars if needed
    if isinstance(df, pd.DataFrame):
        pl_df = pl.from_pandas(df)
    else:
        pl_df = df

    # This will cause the struct.fields() error
    try:
        decoded = pl_df.select(
            pl.col(column_name).str.json_decode().alias("decoded_props")
        )

        # This line will fail in newer Polars versions
        all_keys = (
            decoded.select(pl.col("decoded_props").struct.fields())
            .to_series()
            .explode()
            .unique()
        )

        return all_keys.to_list()

    except Exception:
        # Log additional context before re-raising
        diagnostic_logger.logger.error(
            f"🔍 Additional context: column={column_name}, df_type={type(df)}"
        )
        raise


@smart_diagnostic(diagnostic_logger)
def working_fallback_function(df, column_name):
    """Function that uses fallback approach"""

    if isinstance(df, pd.DataFrame):
        pl_df = pl.from_pandas(df)
    else:
        pl_df = df

    decoded = pl_df.select(pl.col(column_name).str.json_decode().alias("decoded_props"))

    # Use fallback approach
    if not decoded.is_empty():
        sample_struct = decoded.filter(pl.col("decoded_props").is_not_null()).limit(1)
        if not sample_struct.is_empty():
            first_row = sample_struct.row(0, named=True)
            if first_row["decoded_props"] is not None:
                all_keys_set = set()
                for i in range(min(decoded.height, 10)):
                    try:
                        row = decoded.row(i, named=True)
                        if row["decoded_props"] is not None:
                            all_keys_set.update(row["decoded_props"].keys())
                    except:
                        continue
                return list(all_keys_set)

    return []


def create_test_data():
    """Create test data for diagnostics"""
    data = {
        "user_id": ["user_1", "user_2", "user_3"],
        "event_name": ["sign_up", "login", "purchase"],
        "properties": [
            '{"platform": "mobile", "utm_source": "google"}',
            '{"platform": "desktop", "utm_source": "direct"}',
            '{"platform": "mobile", "utm_source": "facebook", "amount": 99.99}',
        ],
    }
    return pd.DataFrame(data)


def main():
    """Test the advanced diagnostics system"""

    print("🧪 Testing Advanced Diagnostics System")
    print("=" * 50)

    # Create test data
    test_df = create_test_data()
    diagnostic_logger.logger.info(f"📊 Created test data: {test_df.shape}")

    # Test 1: Function that will fail
    print("\n🔥 Test 1: Function that will fail with detailed diagnostics")
    try:
        result = problematic_polars_function(test_df, "properties")
        print(f"✅ Unexpected success: {result}")
    except Exception:
        print("❌ Expected failure captured with full context")

    # Test 2: Function that works
    print("\n✅ Test 2: Function that works with fallback")
    try:
        result = working_fallback_function(test_df, "properties")
        print(f"✅ Success: Found {len(result)} keys: {result}")
    except Exception as e:
        print(f"❌ Unexpected failure: {e}")

    # Test 3: Generate diagnostic report
    print("\n📋 Test 3: Generate comprehensive diagnostic report")
    report = diagnostic_logger.generate_diagnostic_report("test_diagnostic_report.json")

    print("📊 Diagnostics Summary:")
    print(f"   Total calls: {report['summary']['total_calls']}")
    print(f"   Total failures: {report['summary']['total_failures']}")
    print(f"   Success rate: {report['summary']['success_rate']:.1f}%")
    print(f"   Data snapshots: {report['summary']['data_snapshots']}")

    # Show failure analysis
    if report["failure_points"]:
        print("\n🔍 Failure Analysis:")
        for i, failure in enumerate(report["failure_points"]):
            print(f"   Failure {i + 1}:")
            print(f"     Function: {failure['function']}")
            print(
                f"     Error: {failure['exception_type']}: {failure['exception_message']}"
            )
            print(
                f"     Suggestions: {len(failure['suggested_fixes'])} recommendations"
            )
            for suggestion in failure["suggested_fixes"]:
                print(f"       💡 {suggestion}")

    # Show recommendations
    if report["recommendations"]:
        print("\n🎯 System Recommendations:")
        for rec in report["recommendations"]:
            print(f"   📝 {rec}")

    print("\n📄 Full report saved to: test_diagnostic_report.json")


if __name__ == "__main__":
    main()
