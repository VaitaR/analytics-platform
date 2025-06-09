#!/usr/bin/env python3
"""
Test Runner for Funnel Analytics Testing System

This script provides various ways to run the automated tests for the funnel analytics engine.
It supports different test categories, coverage reporting, and parallel execution.

Categories:
    - Basic tests: Basic scenarios, conversion window, counting methods
    - Advanced tests: Edge cases, segmentation, integration flow, no-reload improvements
    - Polars tests: Polars engine, path analysis, pandas comparison
    - Comprehensive tests: All configuration combinations, edge cases, performance
    - Fallback detection tests: Tests that detect silent fallbacks to slower implementations
    - Benchmarks: Performance tests and comparisons between implementations

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --basic-all       # Run all basic tests
    python run_tests.py --coverage        # Run tests with coverage report
    python run_tests.py --parallel        # Run tests in parallel
    python run_tests.py --marker edge_case # Run tests with specific marker
    python run_tests.py --benchmarks      # Run all performance benchmarks
"""

import sys
import os
import subprocess
import argparse
import re
import json
from pathlib import Path
from typing import List, Dict, Callable, Optional, Tuple, Set, Union, Any


class TestCategory:
    """Class to organize tests into logical categories."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.test_functions: Dict[str, Tuple[List[str], str, Optional[str]]] = {}
    
    def add_test(self, name: str, test_files: List[str], description: str, marker: Optional[str] = None):
        """
        Add a test to this category.
        
        Args:
            name: Unique name for this test
            test_files: List of test files or specific test paths
            description: Human-readable description
            marker: Optional pytest marker (e.g. 'performance', 'data_integrity')
        """
        self.test_functions[name] = (test_files, description, marker)
    
    def get_all_test_files(self) -> Set[str]:
        """
        Get all unique test files in this category.
        Returns only file paths, not specific test nodes or markers.
        """
        all_files = set()
        for test_files, _, _ in self.test_functions.values():
            # Filter out specific test nodes (containing ::) for validation purposes
            file_paths = [f for f in test_files if "::" not in f]
            all_files.update(file_paths)
        return all_files


def setup_environment() -> bool:
    """Setup the testing environment and paths."""
    # Add the project root to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Ensure tests directory exists
    tests_dir = project_root / "tests"
    if not tests_dir.exists():
        print(f"❌ Tests directory not found: {tests_dir}")
        return False
    
    return True


def run_command(cmd: List[str], description: str = "", capture_output: bool = True) -> Tuple[bool, str, str]:
    """
    Run a command and return the result.
    
    Args:
        cmd: Command to run as a list of strings
        description: Description of the command
        capture_output: Whether to capture and return stdout/stderr
    
    Returns:
        Tuple of (success, stdout, stderr)
    """
    if description:
        print(f"🔄 {description}")
    
    print(f"   Running: {' '.join(cmd)}")
    
    try:
        if capture_output:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                print(f"✅ {description or 'Command'} completed successfully")
                if result.stdout and len(result.stdout) < 1000:  # Only print short outputs
                    print(result.stdout)
            else:
                print(f"❌ {description or 'Command'} failed")
                if result.stderr and len(result.stderr) < 1000:
                    print("STDERR:", result.stderr)
                if result.stdout and len(result.stdout) < 1000:
                    print("STDOUT:", result.stdout)
            
            return result.returncode == 0, result.stdout, result.stderr
        else:
            # Run without capturing output (directly to terminal)
            result = subprocess.run(cmd, check=False)
            return result.returncode == 0, "", ""
    
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        print("   Make sure pytest is installed: pip install pytest")
        return False, "", f"Command not found: {cmd[0]}"
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False, "", str(e)


def run_pytest(test_files: List[str], description: str, 
               parallel: bool = False, 
               coverage: bool = False, 
               markers: Optional[List[str]] = None,
               specific_marker: Optional[str] = None,
               verbose: bool = True,
               capture_output: bool = True,
               extra_args: Optional[List[str]] = None) -> Tuple[bool, str, str]:
    """
    Run pytest with specified test files and options.
    
    Args:
        test_files: List of test files or specific test paths
        description: Human-readable description of the test run
        parallel: Whether to run tests in parallel
        coverage: Whether to collect coverage information
        markers: Additional markers to filter tests
        specific_marker: A specific marker to use for this test run
        verbose: Whether to use verbose output
        capture_output: Whether to capture output or stream directly to terminal
        extra_args: Additional pytest arguments
    
    Returns:
        Tuple of (success, stdout, stderr)
    """
    cmd = ["python", "-m", "pytest"]
    
    # Add test files
    cmd.extend(test_files)
    
    # Add verbose flag if requested
    if verbose:
        cmd.append("-v")
    
    # Add specific marker if provided
    if specific_marker:
        cmd.extend(["-m", specific_marker])
    
    # Add parallel option if requested
    if parallel:
        cmd.extend(["-n", "auto"])
    
    # Add coverage options if requested
    if coverage:
        cmd.extend(["--cov=app", "--cov-report=html", "--cov-report=term-missing"])
    
    # Add markers if specified
    if markers:
        # Only add -m if we haven't already added it with specific_marker
        if not specific_marker:
            for marker in markers:
                cmd.extend(["-m", marker])
    
    # Add any extra arguments
    if extra_args:
        cmd.extend(extra_args)
    
    # Update description with options
    full_description = description
    if parallel:
        full_description += " (parallel)"
    if coverage:
        full_description += " with coverage"
    if markers:
        full_description += f" with markers: {', '.join(markers)}"
    if specific_marker and not markers:
        full_description += f" with marker: {specific_marker}"
    
    return run_command(cmd, full_description, capture_output)


def check_test_dependencies() -> bool:
    """Check if all test dependencies are installed."""
    print("🔍 Checking test dependencies...")
    
    dependencies = ["pytest", "pandas", "numpy", "scipy", "polars"]
    missing = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"   ✅ {dep}")
        except ImportError:
            print(f"   ❌ {dep}")
            missing.append(dep)
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies available")
    return True


def validate_test_files() -> bool:
    """
    Validate that all test files are present and importable.
    Only validates actual files, not specific test nodes or markers.
    """
    print("🔍 Validating test files...")
    
    # Get all test files from all categories
    test_files = set()
    for category in TEST_CATEGORIES.values():
        test_files.update(category.get_all_test_files())
    
    # Add essential test files that might not be in categories
    test_files.add("tests/conftest.py")
    
    missing = []
    for test_file in sorted(test_files):
        if Path(test_file).exists():
            print(f"   ✅ {test_file}")
        else:
            print(f"   ❌ {test_file}")
            missing.append(test_file)
    
    if missing:
        print(f"\n❌ Missing test files: {', '.join(missing)}")
        return False
    
    print("✅ All test files present")
    return True


def generate_test_report() -> bool:
    """Generate a comprehensive test report."""
    print("📊 Generating comprehensive test report...")
    
    cmd = [
        "python", "-m", "pytest", "tests/",
        "--cov=app",
        "--cov-report=html:htmlcov",
        "--cov-report=term-missing",
        "--cov-report=xml:coverage.xml",
        "--junit-xml=test-results.xml",
        "-v"
    ]
    
    success, _, _ = run_command(cmd, "Generating test report")
    
    if success:
        print("\n📋 Test Report Generated:")
        print("   📄 HTML Coverage Report: htmlcov/index.html")
        print("   📄 XML Coverage Report: coverage.xml")
        print("   📄 JUnit XML Report: test-results.xml")
    
    return success


def parse_fallback_test_output(output: str) -> Dict[str, Any]:
    """
    Parse the output from fallback tests to extract structured information about failures.
    
    Args:
        output: stdout/stderr output from pytest
        
    Returns:
        Dictionary with structured information about fallbacks
    """
    results = {
        "config_combinations": [],
        "failures": [],
        "error_types": {
            "nested_object_types": 0,
            "original_order": 0,
            "cross_join_keys": 0,
            "other": 0
        },
        "component_stats": {
            "path_analysis": {"total": 0, "failures": 0},
            "time_to_convert": {"total": 0, "failures": 0},
            "cohort_analysis": {"total": 0, "failures": 0}
        },
        "success_rate": 0.0
    }
    
    # Extract all test configurations and their errors
    fallback_pattern = r"Fallback detected in (\w+) for: (\w+), (\w+), (\w+)\. Fallback messages: (.+)"
    fallback_matches = re.findall(fallback_pattern, output)
    
    # Process each fallback
    for component, funnel_order, reentry_mode, counting_method, error_messages in fallback_matches:
        config = f"{funnel_order}-{reentry_mode}-{counting_method}"
        
        # Add unique configs
        if config not in [c["config"] for c in results["config_combinations"]]:
            results["config_combinations"].append({
                "config": config,
                "funnel_order": funnel_order,
                "reentry_mode": reentry_mode,
                "counting_method": counting_method
            })
        
        # Track component stats
        if component in results["component_stats"]:
            results["component_stats"][component]["total"] += 1
            results["component_stats"][component]["failures"] += 1
        
        # Categorize error types
        error_type = "other"
        if "nested object types" in error_messages.lower():
            error_type = "nested_object_types"
            results["error_types"]["nested_object_types"] += 1
        elif "_original_order" in error_messages:
            error_type = "original_order"
            results["error_types"]["original_order"] += 1
        elif "cross join" in error_messages.lower():
            error_type = "cross_join_keys"
            results["error_types"]["cross_join_keys"] += 1
        else:
            results["error_types"]["other"] += 1
        
        results["failures"].append({
            "component": component,
            "config": config,
            "funnel_order": funnel_order,
            "reentry_mode": reentry_mode,
            "counting_method": counting_method,
            "error_type": error_type,
            "error_messages": error_messages
        })
    
    # Calculate success rates
    total_tests = 0
    for component in results["component_stats"]:
        # Each component is tested with each configuration
        component_tests = len(results["config_combinations"]) if results["config_combinations"] else 12  # Default to 12 combinations
        results["component_stats"][component]["total"] = component_tests
        total_tests += component_tests
    
    total_failures = len(results["failures"])
    results["success_rate"] = 0 if total_tests == 0 else (total_tests - total_failures) / total_tests * 100
    
    return results


def generate_fallback_report(results: Dict[str, Any]) -> str:
    """
    Generate a human-readable report from fallback test results.
    
    Args:
        results: Results dictionary from parse_fallback_test_output
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("# Fallback Detection Report")
    report.append("\n## Summary")
    
    # Calculate overall success rate
    success_rate = results["success_rate"]
    report.append(f"- Overall success rate: {success_rate:.1f}%")
    report.append(f"- Total fallbacks detected: {len(results['failures'])}")
    
    # Component summary
    report.append("\n## Component Performance")
    for component, stats in results["component_stats"].items():
        if stats["total"] > 0:
            failure_rate = (stats["failures"] / stats["total"]) * 100
            report.append(f"- {component}: {stats['failures']} fallbacks out of {stats['total']} tests ({failure_rate:.1f}% failure rate)")
    
    # Error type summary
    report.append("\n## Error Types")
    for error_type, count in results["error_types"].items():
        if count > 0:
            report.append(f"- {error_type}: {count} occurrences")
    
    # Configuration patterns
    report.append("\n## Configuration Patterns")
    report.append("Configurations with fallbacks:")
    
    # Group by config
    config_failures = {}
    for failure in results["failures"]:
        config = failure["config"]
        if config not in config_failures:
            config_failures[config] = []
        config_failures[config].append(failure)
    
    for config, failures in config_failures.items():
        report.append(f"\n### {config}")
        report.append(f"- Total fallbacks: {len(failures)}")
        
        # Group by error type within config
        error_types = {}
        for failure in failures:
            error_type = failure["error_type"]
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(failure)
        
        for error_type, type_failures in error_types.items():
            report.append(f"- {error_type}: {len(type_failures)} occurrences")
    
    # Recommendations based on patterns
    report.append("\n## Recommendations")
    
    if results["error_types"]["nested_object_types"] > 0:
        report.append("- Fix nested object types errors by using `strict=False` when converting pandas to polars")
    
    if results["error_types"]["original_order"] > 0:
        report.append("- Fix original_order errors by preserving row indices when working with polars DataFrames")
    
    if results["error_types"]["cross_join_keys"] > 0:
        report.append("- Fix cross join errors by using proper cross join syntax in polars (don't pass join keys)")
    
    return "\n".join(report)


def run_fallback_report() -> bool:
    """
    Run the fallback tests and generate a comprehensive report of the fallbacks.
    
    Returns:
        True if the report was generated successfully
    """
    print("📊 Running fallback tests and generating report...")
    
    # Run the fallback tests with additional args to capture all output
    success, stdout, stderr = run_pytest(
        ["tests/test_fallback_comprehensive.py"], 
        "Running fallback detection tests", 
        capture_output=True,
        extra_args=["-v"]
    )
    
    # Combine stdout and stderr
    output = stdout + stderr
    
    # Parse the output
    results = parse_fallback_test_output(output)
    
    # Generate the report
    report = generate_fallback_report(results)
    
    # Write to file
    report_path = Path("FALLBACK_REPORT.md")
    report_path.write_text(report)
    
    print(f"\n📋 Fallback Report Generated: {report_path}")
    print("Here's a summary of the findings:")
    print(f"- Overall success rate: {results['success_rate']:.1f}%")
    print(f"- Total fallbacks detected: {len(results['failures'])}")
    
    for component, stats in results["component_stats"].items():
        if stats["total"] > 0:
            failure_rate = (stats["failures"] / stats["total"]) * 100
            print(f"- {component}: {failure_rate:.1f}% failure rate")
    
    return True


# Define test categories and their tests
# This makes it easier to manage and group tests logically

# Create categories
BASIC_TESTS = TestCategory("basic", "Basic functionality tests")
ADVANCED_TESTS = TestCategory("advanced", "Advanced functionality tests")
POLARS_TESTS = TestCategory("polars", "Polars engine and migration tests")
COMPREHENSIVE_TESTS = TestCategory("comprehensive", "Comprehensive tests covering multiple configurations")
FALLBACK_TESTS = TestCategory("fallback", "Tests for detecting silent fallbacks")
BENCHMARK_TESTS = TestCategory("benchmark", "Performance benchmarks and comparisons")

# Basic tests
BASIC_TESTS.add_test(
    "basic_scenarios", 
    ["tests/test_basic_scenarios.py"], 
    "Running basic scenario tests"
)
BASIC_TESTS.add_test(
    "conversion_window", 
    ["tests/test_conversion_window.py"], 
    "Running conversion window tests"
)
BASIC_TESTS.add_test(
    "counting_methods", 
    ["tests/test_counting_methods.py"], 
    "Running counting method tests"
)

# Advanced tests
ADVANCED_TESTS.add_test(
    "edge_cases", 
    ["tests/test_edge_cases.py"], 
    "Running edge case tests"
)
ADVANCED_TESTS.add_test(
    "segmentation", 
    ["tests/test_segmentation.py"], 
    "Running segmentation tests"
)
ADVANCED_TESTS.add_test(
    "integration", 
    ["tests/test_integration_flow.py"], 
    "Running integration tests for complete workflow"
)
ADVANCED_TESTS.add_test(
    "no_reload", 
    ["tests/test_no_reload_improvements.py"], 
    "Running no-reload improvements tests"
)

# Polars tests
POLARS_TESTS.add_test(
    "polars_engine", 
    [
        "tests/test_polars_engine.py",
        "tests/test_polars_path_analysis.py",
        "tests/test_polars_pandas_comparison.py"
    ], 
    "Running Polars engine and migration tests"
)

# Comprehensive tests
COMPREHENSIVE_TESTS.add_test(
    "comprehensive_all", 
    ["tests/test_funnel_calculator_comprehensive.py"], 
    "Running comprehensive funnel calculator tests"
)
COMPREHENSIVE_TESTS.add_test(
    "config_combinations", 
    ["tests/test_funnel_calculator_comprehensive.py"], 
    "Running tests for all configuration combinations",
    "config_combinations"
)
COMPREHENSIVE_TESTS.add_test(
    "comprehensive_edge", 
    ["tests/test_funnel_calculator_comprehensive.py"],
    "Running comprehensive edge case tests",
    "edge_case"
)

# Fallback tests
FALLBACK_TESTS.add_test(
    "fallback_detection", 
    [
        "tests/test_polars_fallback_detection.py",
        "tests/test_lazy_frame_bug.py"
    ], 
    "Running improved fallback detection tests"
)
FALLBACK_TESTS.add_test(
    "path_analysis_fix", 
    ["tests/test_path_analysis_fix.py"], 
    "Running path analysis fix tests"
)
FALLBACK_TESTS.add_test(
    "comprehensive_fallback", 
    ["tests/test_fallback_comprehensive.py"], 
    "Running comprehensive fallback detection tests"
)
FALLBACK_TESTS.add_test(
    "all_fallback", 
    [
        "tests/test_polars_fallback_detection.py",
        "tests/test_lazy_frame_bug.py",
        "tests/test_path_analysis_fix.py",
        "tests/test_fallback_comprehensive.py"
    ], 
    "Running all fallback detection tests and generating report",
    "documentation"
)
FALLBACK_TESTS.add_test(
    "fallback_report", 
    ["tests/test_fallback_comprehensive.py"], 
    "Generating fallback report from comprehensive tests"
)

# Benchmark tests
BENCHMARK_TESTS.add_test(
    "performance", 
    ["tests/"], 
    "Running performance tests",
    "performance"
)
BENCHMARK_TESTS.add_test(
    "comprehensive_performance", 
    ["tests/test_funnel_calculator_comprehensive.py"],
    "Running comprehensive performance tests",
    "large_dataset"
)
BENCHMARK_TESTS.add_test(
    "data_integrity", 
    ["tests/"], 
    "Running data integrity tests comparing engines on large datasets",
    "data_integrity"
)
BENCHMARK_TESTS.add_test(
    "polars_fallback", 
    [
        "tests/test_funnel_calculator_comprehensive.py"
    ], 
    "Running tests to detect silent Polars fallbacks",
    "fallback"
)

# Utility tests
UTILITY_TESTS = TestCategory("utility", "Utility tests")
UTILITY_TESTS.add_test(
    "smoke", 
    ["tests/test_basic_scenarios.py"], 
    "Running smoke test",
    "smoke"
)

# Combine all categories
TEST_CATEGORIES = {
    "basic": BASIC_TESTS,
    "advanced": ADVANCED_TESTS,
    "polars": POLARS_TESTS,
    "comprehensive": COMPREHENSIVE_TESTS,
    "fallback": FALLBACK_TESTS,
    "benchmark": BENCHMARK_TESTS,
    "utility": UTILITY_TESTS
}


def run_all_tests(parallel=False, coverage=False, markers=None) -> bool:
    """Run all tests with optional configurations."""
    # Collect all test files from all categories
    all_test_files = []
    for category in TEST_CATEGORIES.values():
        all_test_files.extend(list(category.get_all_test_files()))
    
    # Remove any duplicates while preserving order
    unique_files = []
    for file in all_test_files:
        if file not in unique_files:
            unique_files.append(file)
    
    description = "Running all tests"
    success, _, _ = run_pytest(unique_files, description, parallel, coverage, markers)
    return success


def run_tests_by_category(category_name: str, parallel=False, coverage=False, markers=None) -> bool:
    """Run all tests in a specific category."""
    if category_name not in TEST_CATEGORIES:
        print(f"❌ Unknown test category: {category_name}")
        return False
    
    category = TEST_CATEGORIES[category_name]
    
    # Get all test files from this category
    all_test_files = list(category.get_all_test_files())
    
    description = f"Running all {category.description.lower()}"
    success, _, _ = run_pytest(all_test_files, description, parallel, coverage, markers)
    return success


def run_specific_test(category_name: str, test_name: str, 
                     parallel=False, coverage=False, markers=None) -> bool:
    """Run a specific test from a category."""
    if category_name not in TEST_CATEGORIES:
        print(f"❌ Unknown test category: {category_name}")
        return False
    
    category = TEST_CATEGORIES[category_name]
    
    if test_name not in category.test_functions:
        print(f"❌ Unknown test in category {category_name}: {test_name}")
        return False
    
    # Special case for fallback report
    if category_name == "fallback" and test_name == "fallback_report":
        return run_fallback_report()
    
    test_files, description, specific_marker = category.test_functions[test_name]
    success, _, _ = run_pytest(test_files, description, parallel, coverage, markers, specific_marker)
    return success


def run_all_benchmarks(parallel=False) -> bool:
    """Run all benchmark tests."""
    return run_tests_by_category("benchmark", parallel=parallel)


def print_test_summary(categories=None):
    """Print a summary of available tests for reference."""
    print("\n📋 Available Test Categories and Tests:")
    
    if categories is None:
        categories = TEST_CATEGORIES.keys()
    
    for category_name in categories:
        category = TEST_CATEGORIES[category_name]
        print(f"\n▶ {category.name.upper()}: {category.description}")
        
        for test_name, (_, description, marker) in category.test_functions.items():
            marker_info = f" (marker: {marker})" if marker else ""
            print(f"   - {test_name}: {description}{marker_info}")
    
    print("\n💡 Examples:")
    print("   python run_tests.py --basic-all")
    print("   python run_tests.py --fallback comprehensive_fallback")
    print("   python run_tests.py --benchmark performance --parallel")
    print("   python run_tests.py --fallback fallback_report")


def main():
    """Main function to handle command line arguments and run tests."""
    parser = argparse.ArgumentParser(
        description="Funnel Analytics Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Categories:
    - Basic tests: Basic scenarios, conversion window, counting methods
    - Advanced tests: Edge cases, segmentation, integration flow, no-reload improvements
    - Polars tests: Polars engine, path analysis, pandas comparison 
    - Comprehensive tests: All configuration combinations, edge cases, performance
    - Fallback detection tests: Tests that detect silent fallbacks to slower implementations
    - Benchmarks: Performance tests and comparisons between implementations

Examples:
  python run_tests.py                     # Run all tests
  python run_tests.py --basic-all         # Run all basic tests
  python run_tests.py --basic scenarios   # Run specific basic test
  python run_tests.py --advanced-all      # Run all advanced tests
  python run_tests.py --polars-all        # Run all Polars tests
  python run_tests.py --comprehensive-all # Run all comprehensive tests
  python run_tests.py --fallback-all      # Run all fallback detection tests
  python run_tests.py --benchmarks        # Run all benchmarks
  python run_tests.py --data-integrity    # Run data integrity tests
  python run_tests.py --coverage          # Run all tests with coverage
  python run_tests.py --parallel          # Run tests in parallel
  python run_tests.py --marker edge_case  # Run edge case tests only
  python run_tests.py --smoke             # Run quick smoke test
  python run_tests.py --report            # Generate comprehensive report
  python run_tests.py --list              # List all available tests
  python run_tests.py --fallback fallback_report  # Generate fallback detection report
        """
    )
    
    # Category group options
    parser.add_argument("--all", action="store_true", help="Run all tests (default action)")
    parser.add_argument("--basic-all", action="store_true", help="Run all basic tests")
    parser.add_argument("--advanced-all", action="store_true", help="Run all advanced tests")
    parser.add_argument("--polars-all", action="store_true", help="Run all Polars tests")
    parser.add_argument("--comprehensive-all", action="store_true", help="Run all comprehensive tests")
    parser.add_argument("--fallback-all", action="store_true", help="Run all fallback detection tests")
    parser.add_argument("--benchmarks", action="store_true", help="Run all benchmark tests")
    
    # Category specific test options
    parser.add_argument("--basic", metavar="TEST", help="Run specific basic test: scenarios, conversion_window, counting_methods")
    parser.add_argument("--advanced", metavar="TEST", help="Run specific advanced test: edge_cases, segmentation, integration, no_reload")
    parser.add_argument("--polars", metavar="TEST", help="Run specific Polars test: polars_engine")
    parser.add_argument("--comprehensive", metavar="TEST", help="Run specific comprehensive test: comprehensive_all, config_combinations, comprehensive_edge")
    parser.add_argument("--fallback", metavar="TEST", help="Run specific fallback test: fallback_detection, path_analysis_fix, comprehensive_fallback, all_fallback, fallback_report")
    parser.add_argument("--benchmark", metavar="TEST", help="Run specific benchmark: performance, comprehensive_performance, data_integrity, polars_fallback")
    
    # Specific named tests for backward compatibility
    parser.add_argument("--smoke", action="store_true", help="Run a quick smoke test")
    parser.add_argument("--data-integrity", action="store_true", help="Run data integrity tests")
    parser.add_argument("--check", action="store_true", help="Check test dependencies")
    parser.add_argument("--validate", action="store_true", help="Validate test files")
    parser.add_argument("--report", action="store_true", help="Generate a comprehensive test report")
    parser.add_argument("--list", action="store_true", help="List all available tests with descriptions")
    parser.add_argument("--fallback-report", action="store_true", help="Generate fallback detection report")
    
    # Test execution options
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--marker", action="append", help="Run tests with specific marker (can be used multiple times)")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    parser.add_argument("--no-capture", action="store_true", help="Don't capture output, show all test output directly")
    
    args = parser.parse_args()
    
    print("🧪 Funnel Analytics Test Runner")
    print("=" * 50)
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    # List all available tests if requested
    if args.list:
        print_test_summary()
        sys.exit(0)
    
    # Check dependencies if requested
    if args.check:
        if not check_test_dependencies():
            sys.exit(1)
    
    # Validate test files if requested
    if args.validate:
        if not validate_test_files():
            sys.exit(1)
    
    success = True
    actions_performed = False
    
    # Generate fallback report if requested
    if args.fallback_report:
        success = run_fallback_report() and success
        actions_performed = True
    
    # Run category groups
    if args.basic_all:
        success = run_tests_by_category("basic", args.parallel, args.coverage, args.marker) and success
        actions_performed = True
    
    if args.advanced_all:
        success = run_tests_by_category("advanced", args.parallel, args.coverage, args.marker) and success
        actions_performed = True
    
    if args.polars_all:
        success = run_tests_by_category("polars", args.parallel, args.coverage, args.marker) and success
        actions_performed = True
    
    if args.comprehensive_all:
        success = run_tests_by_category("comprehensive", args.parallel, args.coverage, args.marker) and success
        actions_performed = True
    
    if args.fallback_all:
        success = run_tests_by_category("fallback", args.parallel, args.coverage, args.marker) and success
        actions_performed = True
    
    if args.benchmarks:
        success = run_all_benchmarks(args.parallel) and success
        actions_performed = True
    
    # Run specific tests from categories
    if args.basic:
        success = run_specific_test("basic", args.basic, args.parallel, args.coverage, args.marker) and success
        actions_performed = True
    
    if args.advanced:
        success = run_specific_test("advanced", args.advanced, args.parallel, args.coverage, args.marker) and success
        actions_performed = True
    
    if args.polars:
        success = run_specific_test("polars", args.polars, args.parallel, args.coverage, args.marker) and success
        actions_performed = True
    
    if args.comprehensive:
        success = run_specific_test("comprehensive", args.comprehensive, args.parallel, args.coverage, args.marker) and success
        actions_performed = True
    
    if args.fallback:
        success = run_specific_test("fallback", args.fallback, args.parallel, args.coverage, args.marker) and success
        actions_performed = True
    
    if args.benchmark:
        success = run_specific_test("benchmark", args.benchmark, args.parallel, args.coverage, args.marker) and success
        actions_performed = True
    
    # Handle specific named tests for backward compatibility
    if args.smoke:
        success = run_specific_test("utility", "smoke", args.parallel, args.coverage, args.marker) and success
        actions_performed = True
    
    if args.data_integrity:
        success = run_specific_test("benchmark", "data_integrity", args.parallel, args.coverage, args.marker) and success
        actions_performed = True
    
    if args.report:
        success = generate_test_report() and success
        actions_performed = True
    
    # If no specific tests are selected, run all tests (unless it's a marker run)
    if not actions_performed and not args.marker:
        success = run_all_tests(args.parallel, args.coverage)
    
    # Print summary
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print("❌ Some tests failed or encountered errors.")
        print("\nNext steps:")
        print("1. Review the test output above")
        print("2. Check for any missing dependencies")
        print("3. Verify test data and fixtures")
        print("4. Run individual test categories to isolate issues")
        sys.exit(1)


if __name__ == "__main__":
    main() 