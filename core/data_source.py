"""
Data Source Management Module for Funnel Analytics Platform
===========================================================

This module contains the DataSourceManager class responsible for:
- Loading data from various sources (files, ClickHouse, sample data)
- Data validation and preprocessing
- JSON property extraction and analysis
- Performance monitoring for data operations

Classes:
    DataSourceManager: Main class for data source management

Usage:
    from core.data_source import DataSourceManager
    manager = DataSourceManager()
    data = manager.get_sample_data()
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

try:
    import clickhouse_connect
except ImportError:
    clickhouse_connect = None


def _data_source_performance_monitor(func_name: str):
    """Decorator for monitoring DataSourceManager function performance"""

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            try:
                result = func(self, *args, **kwargs)
                execution_time = time.time() - start_time

                if not hasattr(self, "_performance_metrics"):
                    self._performance_metrics = {}

                if func_name not in self._performance_metrics:
                    self._performance_metrics[func_name] = []

                self._performance_metrics[func_name].append(execution_time)

                self.logger.info(f"{func_name} executed in {execution_time:.4f} seconds")
                return result

            except Exception as e:
                execution_time = time.time() - start_time
                self.logger.error(
                    f"{func_name} failed after {execution_time:.4f} seconds: {str(e)}"
                )
                raise

        return wrapper

    return decorator


class DataSourceManager:
    """Manages different data sources for funnel analysis"""

    def __init__(self):
        self.clickhouse_client = None
        self.logger = logging.getLogger(__name__)
        self._performance_metrics = {}  # Performance monitoring for data operations

    def _safe_json_decode(self, column_expr: pl.Expr, infer_all: bool = True) -> pl.Expr:
        """
        Safely decode JSON strings to struct type with flexible schema handling.

        Args:
            column_expr: Polars column expression containing JSON strings
            infer_all: Whether to infer schema from all rows (True) or sample (False)

        Returns:
            Polars expression for decoded JSON struct
        """
        # Always try with explicit schema inference control first
        try:
            # Try with larger schema inference window to handle varying fields
            return column_expr.str.json_decode(infer_schema_length=50000)
        except Exception as e:
            error_msg = str(e).lower()
            if (
                "extra field" in error_msg
                or "consider increasing infer_schema_length" in error_msg
            ):
                self.logger.debug(
                    f"JSON schema mismatch detected (extra fields), trying relaxed approach: {str(e)}"
                )
                try:
                    # Try with null inference (no schema validation)
                    return column_expr.str.json_decode(infer_schema_length=None)
                except Exception as e2:
                    self.logger.debug(f"Extended schema inference failed: {str(e2)}")
                    try:
                        # Final attempt: Use very basic schema inference
                        return column_expr.str.json_decode(infer_schema_length=1)
                    except Exception as e3:
                        self.logger.debug(f"All JSON decode attempts failed: {str(e3)}")
                        # Return original column as string - will be handled by pandas fallback
                        return column_expr
            else:
                self.logger.debug(f"JSON decode failed: {str(e)}")
                try:
                    # Second attempt: Use null schema inference
                    return column_expr.str.json_decode(infer_schema_length=None)
                except Exception as e2:
                    self.logger.debug(f"Fallback JSON decode failed: {str(e2)}")
                    # Final fallback: Return original column as string
                    return column_expr

    def _safe_json_field_access(self, column_expr: pl.Expr, field_name: str) -> pl.Expr:
        """
        Safely access a field from a JSON struct with error handling.

        Args:
            column_expr: Polars column expression containing JSON strings
            field_name: Name of the field to extract

        Returns:
            Polars expression for the field value or null if field doesn't exist
        """
        try:
            # Attempt to decode JSON and access field
            return self._safe_json_decode(column_expr).struct.field(field_name)
        except Exception as e:
            self.logger.debug(f"JSON field access failed for field '{field_name}': {str(e)}")
            # Return null expression as fallback
            return pl.lit(None)

    def validate_event_data(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Validate that DataFrame has required columns for funnel analysis"""
        required_columns = ["user_id", "event_name", "timestamp"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"

        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            except:
                return False, "Cannot convert timestamp column to datetime"

        return True, "Data validation successful"

    def load_from_file(self, uploaded_file) -> pd.DataFrame:
        """Load event data from uploaded file"""
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".parquet"):
                df = pd.read_parquet(uploaded_file)
            else:
                raise ValueError("Unsupported file format. Please use CSV or Parquet files.")

            # Validate data
            is_valid, message = self.validate_event_data(df)
            if not is_valid:
                raise ValueError(message)

            return df
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return pd.DataFrame()

    def connect_clickhouse(
        self, host: str, port: int, username: str, password: str, database: str
    ) -> bool:
        """Connect to ClickHouse database"""
        try:
            self.clickhouse_client = clickhouse_connect.get_client(
                host=host,
                port=port,
                username=username,
                password=password,
                database=database,
            )
            # Test connection
            result = self.clickhouse_client.query("SELECT 1")
            return True
        except Exception as e:
            st.error(f"ClickHouse connection failed: {str(e)}")
            return False

    def load_from_clickhouse(self, query: str) -> pd.DataFrame:
        """Load data from ClickHouse using custom query"""
        if not self.clickhouse_client:
            st.error("ClickHouse client not connected")
            return pd.DataFrame()

        try:
            result = self.clickhouse_client.query_df(query)

            # Validate data
            is_valid, message = self.validate_event_data(result)
            if not is_valid:
                st.error(f"Query result validation failed: {message}")
                return pd.DataFrame()

            return result
        except Exception as e:
            st.error(f"ClickHouse query failed: {str(e)}")
            return pd.DataFrame()

    @_data_source_performance_monitor("get_sample_data")
    def get_sample_data(self) -> pd.DataFrame:
        """Generate sample event data for demonstration"""
        np.random.seed(42)

        # Generate users with user properties
        n_users = 10000
        user_ids = [f"user_{i:05d}" for i in range(n_users)]

        # Generate user properties for segmentation
        user_properties = {}
        for user_id in user_ids:
            user_properties[user_id] = {
                "country": np.random.choice(
                    ["US", "UK", "DE", "FR", "CA"], p=[0.4, 0.2, 0.15, 0.15, 0.1]
                ),
                "subscription_plan": np.random.choice(
                    ["free", "basic", "premium"], p=[0.6, 0.3, 0.1]
                ),
                "age_group": np.random.choice(
                    ["18-25", "26-35", "36-45", "46+"], p=[0.3, 0.4, 0.2, 0.1]
                ),
                "registration_date": (
                    datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365))
                ).strftime("%Y-%m-%d"),
            }

        events_data = []
        event_sequence = [
            "User Sign-Up",
            "Verify Email",
            "First Login",
            "Profile Setup",
            "Tutorial Completed",
            "First Purchase",
        ]

        # Generate realistic funnel progression
        current_users = set(user_ids)
        base_time = datetime(2024, 1, 1)

        for step_idx, event_name in enumerate(event_sequence):
            # Calculate dropout rate (realistic funnel)
            dropout_rates = [0.0, 0.25, 0.20, 0.25, 0.20, 0.22]
            remaining_users = list(current_users)

            if step_idx > 0:
                # Remove some users (dropout)
                n_remaining = int(len(remaining_users) * (1 - dropout_rates[step_idx]))
                remaining_users = np.random.choice(remaining_users, n_remaining, replace=False)
                current_users = set(remaining_users)

            # Generate events for remaining users
            for user_id in remaining_users:
                # Add some time variance between steps with cohort effect
                user_props = user_properties[user_id]
                reg_date = datetime.strptime(user_props["registration_date"], "%Y-%m-%d")

                # Time variance based on registration cohort
                cohort_factor = (reg_date - base_time).days / 365 + 1
                hours_offset = np.random.exponential(24 * step_idx * cohort_factor + 1)
                timestamp = reg_date + timedelta(hours=hours_offset)

                # Add event properties for segmentation
                properties = {
                    "platform": np.random.choice(
                        ["mobile", "desktop", "tablet"], p=[0.6, 0.3, 0.1]
                    ),
                    "utm_source": np.random.choice(
                        ["organic", "google_ads", "facebook", "email", "direct"],
                        p=[0.3, 0.25, 0.2, 0.15, 0.1],
                    ),
                    "utm_campaign": np.random.choice(
                        ["summer_sale", "new_user", "retargeting", "brand"],
                        p=[0.3, 0.3, 0.25, 0.15],
                    ),
                    "app_version": np.random.choice(
                        ["2.1.0", "2.2.0", "2.3.0"], p=[0.2, 0.3, 0.5]
                    ),
                    "device_type": np.random.choice(["ios", "android", "web"], p=[0.4, 0.4, 0.2]),
                }

                events_data.append(
                    {
                        "user_id": user_id,
                        "event_name": event_name,
                        "timestamp": timestamp,
                        "event_properties": json.dumps(properties),
                        "user_properties": json.dumps(user_props),
                    }
                )

        # Add some non-funnel events for path analysis
        additional_events = [
            "Page View",
            "Product View",
            "Search",
            "Add to Wishlist",
            "Help Page Visit",
            "Contact Support",
            "Settings View",
            "Logout",
        ]

        # Generate additional events between funnel steps
        for user_id in user_ids[:5000]:  # Only for subset to make data manageable
            n_additional = np.random.poisson(3)  # Average 3 additional events per user
            for _ in range(n_additional):
                event_name = np.random.choice(additional_events)
                timestamp = base_time + timedelta(
                    hours=np.random.uniform(0, 24 * 30)  # Within 30 days
                )

                properties = {
                    "platform": np.random.choice(
                        ["mobile", "desktop", "tablet"], p=[0.6, 0.3, 0.1]
                    ),
                    "utm_source": np.random.choice(
                        ["organic", "google_ads", "facebook", "email", "direct"],
                        p=[0.3, 0.25, 0.2, 0.15, 0.1],
                    ),
                    "page_category": np.random.choice(
                        ["product", "help", "account", "home"], p=[0.4, 0.2, 0.2, 0.2]
                    ),
                }

                events_data.append(
                    {
                        "user_id": user_id,
                        "event_name": event_name,
                        "timestamp": timestamp,
                        "event_properties": json.dumps(properties),
                        "user_properties": json.dumps(user_properties[user_id]),
                    }
                )

        df = pd.DataFrame(events_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    @_data_source_performance_monitor("get_segmentation_properties")
    def get_segmentation_properties(self, df: pd.DataFrame) -> dict[str, list[str]]:
        """Extract available properties for segmentation"""
        properties = {"event_properties": set(), "user_properties": set()}

        # Use Polars for efficient JSON processing if data is large enough to benefit
        if len(df) > 1000:
            try:
                # Convert to Polars for efficient JSON processing
                pl_df = pl.from_pandas(df)

                # Extract event properties
                if "event_properties" in pl_df.columns:
                    # Filter out nulls first
                    valid_props = pl_df.filter(pl.col("event_properties").is_not_null())
                    if not valid_props.is_empty():
                        # Decode JSON strings to struct type with flexible schema
                        decoded = valid_props.select(
                            self._safe_json_decode(pl.col("event_properties")).alias(
                                "decoded_props"
                            )
                        )
                        # Get all unique keys from all JSON objects
                        if not decoded.is_empty():
                            try:
                                # Try modern Polars API first
                                self.logger.debug(
                                    "Trying modern Polars struct.fields() API for event_properties"
                                )
                                all_keys = (
                                    decoded.select(pl.col("decoded_props").struct.fields())
                                    .to_series()
                                    .explode()
                                    .unique()
                                )
                                self.logger.debug(
                                    f"Successfully extracted {len(all_keys)} event property keys using modern API"
                                )
                            except Exception as e:
                                self.logger.debug(
                                    f"Modern Polars API failed for event_properties: {str(e)}, trying fallback"
                                )
                                try:
                                    # Fallback: Get schema from first non-null struct
                                    sample_struct = decoded.filter(
                                        pl.col("decoded_props").is_not_null()
                                    ).limit(1)
                                    if not sample_struct.is_empty():
                                        first_row = sample_struct.row(0, named=True)
                                        if first_row["decoded_props"] is not None:
                                            all_keys = pl.Series(
                                                list(first_row["decoded_props"].keys())
                                            )
                                            self.logger.debug(
                                                f"Successfully extracted {len(all_keys)} event property keys using fallback API"
                                            )
                                        else:
                                            all_keys = pl.Series([])
                                    else:
                                        all_keys = pl.Series([])
                                except Exception as e2:
                                    self.logger.warning(
                                        f"Both Polars methods failed for event_properties: {str(e2)}"
                                    )
                                    all_keys = pl.Series([])

                            # Add to properties set
                            if len(all_keys) > 0:
                                properties["event_properties"].update(all_keys.to_list())

                # Extract user properties
                if "user_properties" in pl_df.columns:
                    # Filter out nulls first
                    valid_props = pl_df.filter(pl.col("user_properties").is_not_null())
                    if not valid_props.is_empty():
                        # Decode JSON strings to struct type with flexible schema
                        decoded = valid_props.select(
                            self._safe_json_decode(pl.col("user_properties")).alias(
                                "decoded_props"
                            )
                        )
                        # Get all unique keys from all JSON objects
                        if not decoded.is_empty():
                            try:
                                # Try modern Polars API first
                                self.logger.debug(
                                    "Trying modern Polars struct.fields() API for user_properties"
                                )
                                all_keys = (
                                    decoded.select(pl.col("decoded_props").struct.fields())
                                    .to_series()
                                    .explode()
                                    .unique()
                                )
                                self.logger.debug(
                                    f"Successfully extracted {len(all_keys)} user property keys using modern API"
                                )
                            except Exception as e:
                                self.logger.debug(
                                    f"Modern Polars API failed for user_properties: {str(e)}, trying fallback"
                                )
                                try:
                                    # Fallback: Get schema from first non-null struct
                                    sample_struct = decoded.filter(
                                        pl.col("decoded_props").is_not_null()
                                    ).limit(1)
                                    if not sample_struct.is_empty():
                                        first_row = sample_struct.row(0, named=True)
                                        if first_row["decoded_props"] is not None:
                                            all_keys = pl.Series(
                                                list(first_row["decoded_props"].keys())
                                            )
                                            self.logger.debug(
                                                f"Successfully extracted {len(all_keys)} user property keys using fallback API"
                                            )
                                        else:
                                            all_keys = pl.Series([])
                                    else:
                                        all_keys = pl.Series([])
                                except Exception as e2:
                                    self.logger.warning(
                                        f"Both Polars methods failed for user_properties: {str(e2)}"
                                    )
                                    all_keys = pl.Series([])

                            # Add to properties set
                            if len(all_keys) > 0:
                                properties["user_properties"].update(all_keys.to_list())

            except Exception as e:
                error_msg = str(e).lower()
                if (
                    "extra field" in error_msg
                    or "consider increasing infer_schema_length" in error_msg
                ):
                    self.logger.debug(
                        f"Polars JSON schema inference detected varying structures (using pandas fallback): {str(e)}"
                    )
                else:
                    self.logger.warning(
                        f"Polars JSON processing failed: {str(e)}, falling back to Pandas"
                    )
                # Fall back to pandas implementation
                return self._get_segmentation_properties_pandas(df)
        else:
            # For small datasets, use pandas implementation
            return self._get_segmentation_properties_pandas(df)

        return {k: sorted(list(v)) for k, v in properties.items() if v}

    def _get_segmentation_properties_pandas(self, df: pd.DataFrame) -> dict[str, list[str]]:
        """Legacy pandas implementation for extracting segmentation properties"""
        properties = {"event_properties": set(), "user_properties": set()}

        # Extract event properties
        if "event_properties" in df.columns:
            for prop_str in df["event_properties"].dropna():
                try:
                    props = json.loads(prop_str)
                    if props and isinstance(props, dict):
                        properties["event_properties"].update(props.keys())
                except (
                    json.JSONDecodeError,
                    TypeError,
                ):  # Handle both JSON errors and type errors
                    self.logger.debug(f"Failed to decode event_properties: {prop_str[:50]}")
                    continue

        # Extract user properties
        if "user_properties" in df.columns:
            for prop_str in df["user_properties"].dropna():
                try:
                    props = json.loads(prop_str)
                    if props and isinstance(props, dict):
                        properties["user_properties"].update(props.keys())
                except (
                    json.JSONDecodeError,
                    TypeError,
                ):  # Handle both JSON errors and type errors
                    self.logger.debug(f"Failed to decode user_properties: {prop_str[:50]}")
                    continue

        return {k: sorted(list(v)) for k, v in properties.items() if v}

    def get_property_values(self, df: pd.DataFrame, prop_name: str, prop_type: str) -> list[str]:
        """Get unique values for a specific property"""
        column = f"{prop_type}"

        if column not in df.columns:
            return []

        # Use Polars for efficient JSON processing if data is large enough to benefit
        if len(df) > 1000:
            try:
                # Convert to Polars for efficient JSON processing
                pl_df = pl.from_pandas(df)

                # Filter out nulls
                valid_props = pl_df.filter(pl.col(column).is_not_null())
                if valid_props.is_empty():
                    return []

                # Decode JSON strings to struct type with flexible schema
                decoded = valid_props.select(
                    self._safe_json_decode(pl.col(column)).alias("decoded_props")
                )

                # Extract the specific property value and get unique values
                if not decoded.is_empty():
                    # Use struct.field to extract the property value
                    values = (
                        decoded.select(
                            pl.col("decoded_props").struct.field(prop_name).alias("prop_value")
                        )
                        .filter(pl.col("prop_value").is_not_null())
                        .select(pl.col("prop_value").cast(pl.Utf8))
                        .unique()
                        .to_series()
                        .sort()
                        .to_list()
                    )
                    return values
            except Exception as e:
                error_msg = str(e).lower()
                if (
                    "extra field" in error_msg
                    or "consider increasing infer_schema_length" in error_msg
                ):
                    self.logger.debug(
                        f"Polars JSON schema inference detected varying structures in get_property_values (using pandas fallback): {str(e)}"
                    )
                else:
                    self.logger.warning(
                        f"Polars JSON processing failed in get_property_values: {str(e)}, falling back to Pandas"
                    )
                # Fall back to pandas implementation
                return self._get_property_values_pandas(df, prop_name, prop_type)

        # For small datasets, use pandas implementation
        return self._get_property_values_pandas(df, prop_name, prop_type)

    def _get_property_values_pandas(
        self, df: pd.DataFrame, prop_name: str, prop_type: str
    ) -> list[str]:
        """Legacy pandas implementation for getting property values"""
        values = set()
        column = f"{prop_type}"

        if column in df.columns:
            for prop_str in df[column].dropna():
                try:
                    props = json.loads(prop_str)
                    if props is not None and prop_name in props:
                        values.add(str(props[prop_name]))
                except (
                    json.JSONDecodeError,
                    TypeError,
                ):  # Handle both JSON errors and None types
                    self.logger.debug(
                        f"Failed to decode JSON in get_property_values: {prop_str[:50]}"
                    )
                    continue

        return sorted(list(values))

    @_data_source_performance_monitor("get_event_metadata")
    def get_event_metadata(self, df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Extract event metadata for enhanced display with statistics"""
        # Calculate basic statistics for all events
        event_stats = {}
        total_events = len(df)
        total_users = df["user_id"].nunique() if "user_id" in df.columns else 0

        for event_name in df["event_name"].unique():
            event_data = df[df["event_name"] == event_name]
            event_count = len(event_data)
            unique_users = (
                event_data["user_id"].nunique() if "user_id" in event_data.columns else 0
            )

            # Calculate percentages
            event_percentage = (event_count / total_events * 100) if total_events > 0 else 0
            user_penetration = (unique_users / total_users * 100) if total_users > 0 else 0

            event_stats[event_name] = {
                "total_occurrences": event_count,
                "unique_users": unique_users,
                "event_percentage": event_percentage,
                "user_penetration": user_penetration,
                "avg_events_per_user": ((event_count / unique_users) if unique_users > 0 else 0),
            }

        # Try to load demo events metadata
        try:
            # Auto-generate demo data if it doesn't exist
            if not os.path.exists("test_data/demo_events.csv"):
                try:
                    from tests.test_data_generator import ensure_test_data

                    ensure_test_data()
                except ImportError:
                    self.logger.warning(
                        "Test data generator not available, creating minimal demo data"
                    )
                    os.makedirs("test_data", exist_ok=True)
                    # Create minimal demo data
                    minimal_demo = pd.DataFrame(
                        [
                            {
                                "name": "Page View",
                                "category": "Navigation",
                                "description": "User views a page",
                                "frequency": "high",
                            },
                            {
                                "name": "User Sign-Up",
                                "category": "Conversion",
                                "description": "User creates an account",
                                "frequency": "medium",
                            },
                            {
                                "name": "First Purchase",
                                "category": "Revenue",
                                "description": "User makes first purchase",
                                "frequency": "low",
                            },
                        ]
                    )
                    minimal_demo.to_csv("test_data/demo_events.csv", index=False)

            demo_df = pd.read_csv("test_data/demo_events.csv")
            metadata = {}

            # First, add all demo events with their base metadata
            for _, row in demo_df.iterrows():
                base_metadata = {
                    "category": row["category"],
                    "description": row["description"],
                    "frequency": row["frequency"],
                }
                # Add statistics if event exists in current data
                if row["name"] in event_stats:
                    base_metadata.update(event_stats[row["name"]])
                else:
                    # Add zero statistics for demo events not in current data
                    base_metadata.update(
                        {
                            "total_occurrences": 0,
                            "unique_users": 0,
                            "event_percentage": 0.0,
                            "user_penetration": 0.0,
                            "avg_events_per_user": 0.0,
                        }
                    )
                metadata[row["name"]] = base_metadata

            # Then, add any events from current data that aren't in demo file
            for event_name, event_stats_data in event_stats.items():
                if event_name not in metadata:
                    # Categorize unknown events
                    event_lower = event_name.lower()
                    if any(word in event_lower for word in ["sign", "login", "register", "auth"]):
                        category = "Authentication"
                    elif any(
                        word in event_lower for word in ["onboard", "tutorial", "setup", "profile"]
                    ):
                        category = "Onboarding"
                    elif any(
                        word in event_lower
                        for word in ["purchase", "buy", "payment", "checkout", "cart"]
                    ):
                        category = "E-commerce"
                    elif any(
                        word in event_lower for word in ["view", "click", "search", "browse"]
                    ):
                        category = "Engagement"
                    elif any(word in event_lower for word in ["share", "invite", "social"]):
                        category = "Social"
                    elif any(word in event_lower for word in ["mobile", "app", "notification"]):
                        category = "Mobile"
                    else:
                        category = "Other"

                    # Estimate frequency based on statistics
                    event_percentage = event_stats_data.get("event_percentage", 0)
                    if event_percentage > 10:
                        frequency = "high"
                    elif event_percentage > 5:
                        frequency = "medium"
                    else:
                        frequency = "low"

                    base_metadata = {
                        "category": category,
                        "description": f"Event: {event_name}",
                        "frequency": frequency,
                    }
                    base_metadata.update(event_stats_data)
                    metadata[event_name] = base_metadata

            return metadata
        except (
            FileNotFoundError,
            pd.errors.EmptyDataError,
        ) as e:  # More specific exceptions
            # If demo file doesn't exist or is empty, create basic categorization
            self.logger.warning(
                f"Demo events file not found or empty ({e}), generating basic metadata."
            )
            events = df["event_name"].unique()
            metadata = {}

            # Basic categorization based on event names
            for event in events:
                event_lower = event.lower()
                if any(word in event_lower for word in ["sign", "login", "register", "auth"]):
                    category = "Authentication"
                elif any(
                    word in event_lower for word in ["onboard", "tutorial", "setup", "profile"]
                ):
                    category = "Onboarding"
                elif any(
                    word in event_lower
                    for word in ["purchase", "buy", "payment", "checkout", "cart"]
                ):
                    category = "E-commerce"
                elif any(word in event_lower for word in ["view", "click", "search", "browse"]):
                    category = "Engagement"
                elif any(word in event_lower for word in ["share", "invite", "social"]):
                    category = "Social"
                elif any(word in event_lower for word in ["mobile", "app", "notification"]):
                    category = "Mobile"
                else:
                    category = "Other"

                # Estimate frequency based on count in data
                event_count = len(df[df["event_name"] == event])
                total_events = len(df)
                frequency_ratio = event_count / total_events

                if frequency_ratio > 0.1:
                    frequency = "high"
                elif frequency_ratio > 0.01:
                    frequency = "medium"
                else:
                    frequency = "low"

                # Combine base metadata with statistics
                base_metadata = {
                    "category": category,
                    "description": f"Event: {event}",
                    "frequency": frequency,
                }
                base_metadata.update(event_stats.get(event, {}))
                metadata[event] = base_metadata

            return metadata
