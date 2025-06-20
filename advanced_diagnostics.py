"""
Advanced diagnostics system for funnel analytics platform
Provides detailed context capture and failure point analysis
"""

import functools
import hashlib
import inspect
import json
import logging
import sys
import traceback
from datetime import datetime
from typing import Any, Callable


class DiagnosticContext:
    """Captures detailed context for function calls and failures"""

    def __init__(self):
        self.call_stack: list[dict[str, Any]] = []
        self.data_snapshots: dict[str, Any] = {}
        self.performance_metrics: dict[str, float] = {}
        self.failure_points: list[dict[str, Any]] = []

    def capture_call(
        self, func_name: str, args: tuple, kwargs: dict, locals_dict: dict = None
    ):
        """Capture function call with full context"""
        call_info = {
            "timestamp": datetime.now().isoformat(),
            "function": func_name,
            "args_summary": self._summarize_args(args),
            "kwargs_summary": self._summarize_kwargs(kwargs),
            "locals_summary": (
                self._summarize_locals(locals_dict) if locals_dict else {}
            ),
            "call_id": self._generate_call_id(func_name, args, kwargs),
        }
        self.call_stack.append(call_info)
        return call_info["call_id"]

    def capture_data_state(self, data_name: str, data: Any, context: str = ""):
        """Capture state of data objects for analysis"""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "type": type(data).__name__,
            "summary": self._analyze_data_object(data),
        }

        data_key = f"{data_name}_{context}_{len(self.data_snapshots)}"
        self.data_snapshots[data_key] = snapshot
        return data_key

    def capture_failure(
        self,
        exception: Exception,
        func_name: str,
        call_id: str,
        locals_dict: dict = None,
        data_context: dict = None,
    ):
        """Capture detailed failure information"""
        failure_info = {
            "timestamp": datetime.now().isoformat(),
            "function": func_name,
            "call_id": call_id,
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "traceback": traceback.format_exc(),
            "locals_at_failure": (
                self._summarize_locals(locals_dict) if locals_dict else {}
            ),
            "data_context": data_context or {},
            "suggested_fixes": self._analyze_failure_pattern(exception, func_name),
        }
        self.failure_points.append(failure_info)
        return failure_info

    def _summarize_args(self, args: tuple) -> list[dict[str, Any]]:
        """Create safe summary of function arguments"""
        summaries = []
        for i, arg in enumerate(args):
            summary = {
                "position": i,
                "type": type(arg).__name__,
                "summary": self._analyze_data_object(arg),
            }
            summaries.append(summary)
        return summaries

    def _summarize_kwargs(self, kwargs: dict) -> dict[str, dict[str, Any]]:
        """Create safe summary of keyword arguments"""
        summaries = {}
        for key, value in kwargs.items():
            summaries[key] = {
                "type": type(value).__name__,
                "summary": self._analyze_data_object(value),
            }
        return summaries

    def _summarize_locals(self, locals_dict: dict) -> dict[str, Any]:
        """Create safe summary of local variables"""
        if not locals_dict:
            return {}

        summary = {}
        for key, value in locals_dict.items():
            if not key.startswith("_"):  # Skip private variables
                try:
                    summary[key] = {
                        "type": type(value).__name__,
                        "summary": self._analyze_data_object(value),
                    }
                except Exception:
                    summary[key] = {"type": "unknown", "error": "failed_to_analyze"}
        return summary

    def _analyze_data_object(self, obj: Any) -> dict[str, Any]:
        """Analyze data object and return safe summary"""
        try:
            obj_type = type(obj).__name__

            # Handle DataFrames (Pandas/Polars)
            if hasattr(obj, "shape") and hasattr(obj, "columns"):
                return {
                    "shape": getattr(obj, "shape", "unknown"),
                    "columns": list(getattr(obj, "columns", [])),
                    "dtypes": str(getattr(obj, "dtypes", "unknown")),
                    "memory_usage": self._get_memory_usage(obj),
                    "has_nulls": self._check_nulls(obj),
                    "sample_values": self._get_sample_values(obj),
                }

            # Handle Series
            if hasattr(obj, "dtype") and hasattr(obj, "shape"):
                return {
                    "shape": getattr(obj, "shape", "unknown"),
                    "dtype": str(getattr(obj, "dtype", "unknown")),
                    "sample_values": self._get_sample_values(obj),
                }

            # Handle collections
            if isinstance(obj, (list, tuple, set)):
                return {
                    "length": len(obj),
                    "sample_items": (
                        list(obj)[:3] if len(obj) <= 3 else list(obj)[:2] + ["..."]
                    ),
                    "item_types": list(set(type(item).__name__ for item in obj)),
                }

            # Handle dictionaries
            if isinstance(obj, dict):
                return {
                    "keys_count": len(obj),
                    "keys_sample": list(obj.keys())[:5],
                    "value_types": list(set(type(v).__name__ for v in obj.values())),
                }

            # Handle basic types
            if isinstance(obj, (str, int, float, bool)):
                return {
                    "value": str(obj)[:100] + "..." if len(str(obj)) > 100 else str(obj)
                }

            # Handle other objects
            return {
                "attributes": [attr for attr in dir(obj) if not attr.startswith("_")][
                    :10
                ],
                "str_repr": str(obj)[:100] + "..." if len(str(obj)) > 100 else str(obj),
            }

        except Exception as e:
            return {"analysis_error": str(e), "type": obj_type}

    def _get_memory_usage(self, obj) -> str:
        """Get memory usage of object if available"""
        try:
            if hasattr(obj, "memory_usage"):
                return f"{obj.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
            if hasattr(obj, "estimated_size"):
                return f"{obj.estimated_size() / 1024 / 1024:.2f} MB"
            return "unknown"
        except:
            return "unknown"

    def _check_nulls(self, obj) -> dict[str, Any]:
        """Check for null values in data object"""
        try:
            if hasattr(obj, "isnull"):
                null_counts = obj.isnull().sum()
                return {
                    "total_nulls": (
                        int(null_counts.sum())
                        if hasattr(null_counts, "sum")
                        else int(null_counts)
                    )
                }
            if hasattr(obj, "null_count"):
                return {"total_nulls": obj.null_count()}
            return {"status": "unknown"}
        except:
            return {"status": "error"}

    def _get_sample_values(self, obj) -> list[Any]:
        """Get sample values from data object"""
        try:
            if hasattr(obj, "head"):
                sample = obj.head(3)
                if hasattr(sample, "to_dict"):
                    return sample.to_dict("records")
                return sample.to_list() if hasattr(sample, "to_list") else list(sample)
            if hasattr(obj, "__getitem__") and hasattr(obj, "__len__"):
                return [obj[i] for i in range(min(3, len(obj)))]
            return []
        except:
            return []

    def _generate_call_id(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate unique ID for function call"""
        content = f"{func_name}_{len(args)}_{len(kwargs)}_{datetime.now().timestamp()}"
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def _analyze_failure_pattern(
        self, exception: Exception, func_name: str
    ) -> list[str]:
        """Analyze failure pattern and suggest fixes"""
        suggestions = []
        error_msg = str(exception).lower()

        # Polars-specific errors
        if "struct" in error_msg and "fields" in error_msg:
            suggestions.extend(
                [
                    "Polars struct.fields() API compatibility issue detected",
                    "Try using fallback method: sample rows and extract keys manually",
                    "Consider updating Polars version or using pandas fallback",
                ]
            )

        # DataFrame attribute errors
        elif "columns" in error_msg or "with_columns" in error_msg:
            suggestions.extend(
                [
                    "DataFrame type mismatch - object doesn't have expected attributes",
                    "Check if object is actually a DataFrame (Pandas/Polars)",
                    "Add type checking: hasattr(obj, 'columns') before accessing",
                ]
            )

        # JSON processing errors
        elif "json" in error_msg:
            suggestions.extend(
                [
                    "JSON processing error - check data format",
                    "Consider adding json validation before processing",
                    "Try different JSON parsing approach or schema inference",
                ]
            )

        # Schema inference errors
        elif "schema" in error_msg or "infer" in error_msg:
            suggestions.extend(
                [
                    "Schema inference failed - data structure inconsistent",
                    "Try increasing infer_schema_length parameter",
                    "Consider manual schema specification",
                ]
            )

        return suggestions


class SmartDiagnosticLogger:
    """Enhanced logger with automatic context capture and intelligent analysis"""

    def __init__(self, name: str = __name__, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.context = DiagnosticContext()
        self.setup_logger(level)

    def setup_logger(self, level: int):
        """Setup logger with enhanced formatting"""
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(funcName)s | %(message)s",
                datefmt="%H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(level)

    def log_function_call(
        self, func_name: str, args: tuple, kwargs: dict, locals_dict: dict = None
    ) -> str:
        """Log function call with full context"""
        call_id = self.context.capture_call(func_name, args, kwargs, locals_dict)

        self.logger.debug(f"🔄 CALL_START [{call_id}] {func_name}")
        self.logger.debug(f"   Args: {len(args)} items")
        self.logger.debug(f"   Kwargs: {list(kwargs.keys())}")

        # Log critical data objects
        for i, arg in enumerate(args):
            if self._is_critical_data_object(arg):
                data_key = self.context.capture_data_state(
                    f"arg_{i}", arg, f"call_{call_id}"
                )
                self.logger.debug(
                    f"   📊 Arg[{i}]: {self.context.data_snapshots[data_key]['summary']}"
                )

        return call_id

    def log_function_success(self, call_id: str, func_name: str, result: Any = None):
        """Log successful function completion"""
        self.logger.debug(f"✅ CALL_SUCCESS [{call_id}] {func_name}")
        if result is not None and self._is_critical_data_object(result):
            data_key = self.context.capture_data_state(
                "result", result, f"success_{call_id}"
            )
            self.logger.debug(
                f"   📈 Result: {self.context.data_snapshots[data_key]['summary']}"
            )

    def log_function_failure(
        self,
        call_id: str,
        func_name: str,
        exception: Exception,
        locals_dict: dict = None,
        data_context: dict = None,
    ):
        """Log function failure with detailed analysis"""
        failure_info = self.context.capture_failure(
            exception, func_name, call_id, locals_dict, data_context
        )

        self.logger.error(f"❌ CALL_FAILURE [{call_id}] {func_name}")
        self.logger.error(
            f"   Exception: {failure_info['exception_type']}: {failure_info['exception_message']}"
        )

        # Log suggestions
        for suggestion in failure_info["suggested_fixes"]:
            self.logger.error(f"   💡 {suggestion}")

        # Log data context if available
        if failure_info["data_context"]:
            self.logger.error(f"   📊 Data context: {failure_info['data_context']}")

        return failure_info

    def _is_critical_data_object(self, obj: Any) -> bool:
        """Check if object is a critical data structure worth logging"""
        return (
            hasattr(obj, "shape")  # DataFrames/Series
            or isinstance(obj, (list, dict))
            and len(obj) > 0  # Non-empty collections
            or hasattr(obj, "columns")  # DataFrame-like objects
        )

    def generate_diagnostic_report(self, output_file: str = None) -> dict[str, Any]:
        """Generate comprehensive diagnostic report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_calls": len(self.context.call_stack),
                "total_failures": len(self.context.failure_points),
                "data_snapshots": len(self.context.data_snapshots),
                "success_rate": (
                    (len(self.context.call_stack) - len(self.context.failure_points))
                    / len(self.context.call_stack)
                    * 100
                    if self.context.call_stack
                    else 0
                ),
            },
            "call_stack": self.context.call_stack,
            "failure_points": self.context.failure_points,
            "data_snapshots": self.context.data_snapshots,
            "recommendations": self._generate_recommendations(),
        }

        if output_file:
            with open(output_file, "w") as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"📋 Diagnostic report saved to: {output_file}")

        return report

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on failure patterns"""
        recommendations = []

        # Analyze failure patterns
        failure_types = {}
        for failure in self.context.failure_points:
            exc_type = failure["exception_type"]
            failure_types[exc_type] = failure_types.get(exc_type, 0) + 1

        if failure_types:
            recommendations.append(
                f"Most common failure: {max(failure_types, key=failure_types.get)} ({failure_types[max(failure_types, key=failure_types.get)]} occurrences)"
            )

        # Check for data-related issues
        polars_issues = sum(
            1
            for f in self.context.failure_points
            if "struct" in f["exception_message"].lower()
        )
        if polars_issues > 0:
            recommendations.append(
                f"Detected {polars_issues} Polars compatibility issues - consider updating fallback logic"
            )

        return recommendations


def smart_diagnostic(logger: SmartDiagnosticLogger = None):
    """Decorator for automatic function diagnostics"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = SmartDiagnosticLogger(func.__module__)

            # Get function signature for better logging
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Start logging
            call_id = logger.log_function_call(
                func.__name__, args, kwargs, dict(bound_args.arguments)
            )

            try:
                result = func(*args, **kwargs)
                logger.log_function_success(call_id, func.__name__, result)
                return result
            except Exception as e:
                # Capture local variables at failure point
                frame = sys.exc_info()[2].tb_frame
                locals_dict = frame.f_locals.copy()

                logger.log_function_failure(
                    call_id,
                    func.__name__,
                    e,
                    locals_dict,
                    {"args": bound_args.arguments},
                )
                raise

        return wrapper

    return decorator


# Global diagnostic logger instance
global_diagnostic_logger = SmartDiagnosticLogger("funnel_analytics")


# Convenience functions
def log_call(func_name: str, *args, **kwargs) -> str:
    """Quick function to log a call"""
    return global_diagnostic_logger.log_function_call(func_name, args, kwargs)


def log_success(call_id: str, func_name: str, result: Any = None):
    """Quick function to log success"""
    global_diagnostic_logger.log_function_success(call_id, func_name, result)


def log_failure(call_id: str, func_name: str, exception: Exception, **context):
    """Quick function to log failure"""
    return global_diagnostic_logger.log_function_failure(
        call_id, func_name, exception, data_context=context
    )


def generate_report(output_file: str = "diagnostic_report.json") -> dict[str, Any]:
    """Generate diagnostic report"""
    return global_diagnostic_logger.generate_diagnostic_report(output_file)
