"""
Enhanced logging configuration for funnel analytics platform
Provides detailed logging for debugging Polars/Pandas compatibility issues
"""

import logging
import sys
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for better readability"""

    # Color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        # Add color to level name
        level_color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        record.levelname = f"{level_color}{record.levelname}{self.COLORS['RESET']}"

        # Format the message
        return super().format(record)


def setup_enhanced_logging(
    level: str = "INFO",
    enable_file_logging: bool = True,
    enable_polars_debug: bool = True,
    log_file_path: str = "funnel_analytics.log",
):
    """
    Setup enhanced logging configuration for debugging

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        enable_file_logging: Whether to log to file
        enable_polars_debug: Whether to enable detailed Polars debugging
        log_file_path: Path to log file
    """

    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    console_format = ColoredFormatter(
        "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (if enabled)
    if enable_file_logging:
        log_path = Path(log_file_path)
        log_path.parent.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file

        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-20s:%(lineno)-4d | %(funcName)-20s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        # Log session start
        logger.info(f"🚀 Logging session started - Level: {level}")
        logger.info(f"📝 Log file: {log_path.absolute()}")

    # Enhanced Polars debugging
    if enable_polars_debug:
        # Set specific loggers for Polars-related issues
        polars_logger = logging.getLogger("polars")
        polars_logger.setLevel(logging.DEBUG)

        # DataSource manager logger
        datasource_logger = logging.getLogger("__main__")
        if numeric_level <= logging.DEBUG:
            datasource_logger.setLevel(logging.DEBUG)

        logger.info("🔍 Enhanced Polars debugging enabled")

    # Log system information
    logger.info("🖥️  System Information:")
    logger.info(f"   Python: {sys.version.split()[0]}")

    try:
        import polars as pl

        logger.info(f"   Polars: {pl.__version__}")
    except ImportError:
        logger.warning("   Polars: Not installed")

    try:
        import pandas as pd

        logger.info(f"   Pandas: {pd.__version__}")
    except ImportError:
        logger.warning("   Pandas: Not installed")

    try:
        import streamlit as st

        logger.info(f"   Streamlit: {st.__version__}")
    except ImportError:
        logger.warning("   Streamlit: Not installed")

    return logger


def log_dataframe_info(df, name: str = "DataFrame", logger=None):
    """
    Log detailed information about a DataFrame for debugging

    Args:
        df: DataFrame (Polars or Pandas)
        name: Name to identify the DataFrame
        logger: Logger instance (if None, uses root logger)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.debug(f"📊 {name} Information:")
    logger.debug(f"   Type: {type(df).__name__}")

    try:
        logger.debug(f"   Shape: {df.shape}")
        logger.debug(f"   Columns: {list(df.columns)}")

        # Memory usage (if available)
        if hasattr(df, "memory_usage"):
            try:
                memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
                logger.debug(f"   Memory: {memory_mb:.2f} MB")
            except:
                pass
        elif hasattr(df, "estimated_size"):
            try:
                memory_mb = df.estimated_size() / 1024 / 1024
                logger.debug(f"   Memory: {memory_mb:.2f} MB")
            except:
                pass

        # Sample data
        if len(df) > 0:
            logger.debug(
                f"   Sample row: {df.head(1).to_dict('records')[0] if hasattr(df, 'to_dict') else 'N/A'}"
            )

    except Exception as e:
        logger.warning(f"   Error getting {name} info: {str(e)}")


def log_polars_operation(operation_name: str, func):
    """
    Decorator to log Polars operations with detailed error information

    Args:
        operation_name: Name of the operation for logging
        func: Function to wrap
    """

    def wrapper(*args, **kwargs):
        logger = logging.getLogger(__name__)
        logger.debug(f"🔄 Starting Polars operation: {operation_name}")

        try:
            result = func(*args, **kwargs)
            logger.debug(f"✅ Polars operation completed: {operation_name}")
            return result
        except Exception as e:
            logger.error(f"❌ Polars operation failed: {operation_name}")
            logger.error(f"   Error: {str(e)}")
            logger.error(f"   Error type: {type(e).__name__}")

            # Log additional context for common Polars errors
            error_msg = str(e).lower()
            if "struct" in error_msg and "fields" in error_msg:
                logger.error(
                    "   🔍 This appears to be a Polars struct.fields() API compatibility issue"
                )
                logger.error("   💡 Suggestion: Check Polars version compatibility")
            elif "schema" in error_msg:
                logger.error("   🔍 This appears to be a schema inference issue")
                logger.error(
                    "   💡 Suggestion: Try with explicit schema or pandas fallback"
                )
            elif "json" in error_msg:
                logger.error("   🔍 This appears to be a JSON processing issue")
                logger.error(
                    "   💡 Suggestion: Check JSON format and schema consistency"
                )

            raise

    return wrapper


# Quick setup function for common debugging scenarios
def quick_debug_setup():
    """Quick setup for debugging Polars compatibility issues"""
    return setup_enhanced_logging(
        level="DEBUG",
        enable_file_logging=True,
        enable_polars_debug=True,
        log_file_path="debug_polars.log",
    )


if __name__ == "__main__":
    # Test the logging setup
    logger = quick_debug_setup()
    logger.info("🧪 Testing enhanced logging configuration")
    logger.debug("Debug message test")
    logger.warning("Warning message test")
    logger.error("Error message test")
    print("✅ Logging configuration test completed")
