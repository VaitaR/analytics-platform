#!/usr/bin/env python3
"""
Test Runner for Funnel Analytics Testing System

This script provides various ways to run the automated tests for the funnel analytics engine.
It supports different test categories, coverage reporting, and parallel execution.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --basic           # Run only basic scenario tests
    python run_tests.py --coverage        # Run tests with coverage report
    python run_tests.py --parallel        # Run tests in parallel
    python run_tests.py --marker edge_case # Run tests with specific marker
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


def setup_environment():
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


def run_command(cmd, description=""):
    """Run a command and return the result."""
    if description:
        print(f"🔄 {description}")
    
    print(f"   Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            print(f"✅ {description or 'Command'} completed successfully")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"❌ {description or 'Command'} failed")
            if result.stderr:
                print("STDERR:", result.stderr)
            if result.stdout:
                print("STDOUT:", result.stdout)
        
        return result.returncode == 0
    
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        print("   Make sure pytest is installed: pip install pytest")
        return False
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False


def run_basic_tests():
    """Run basic scenario tests."""
    cmd = ["python", "-m", "pytest", "tests/test_basic_scenarios.py", "-v"]
    return run_command(cmd, "Running basic scenario tests")


def run_conversion_window_tests():
    """Run conversion window tests."""
    cmd = ["python", "-m", "pytest", "tests/test_conversion_window.py", "-v"]
    return run_command(cmd, "Running conversion window tests")


def run_counting_method_tests():
    """Run counting method tests."""
    cmd = ["python", "-m", "pytest", "tests/test_counting_methods.py", "-v"]
    return run_command(cmd, "Running counting method tests")


def run_edge_case_tests():
    """Run edge case tests."""
    cmd = ["python", "-m", "pytest", "tests/test_edge_cases.py", "-v"]
    return run_command(cmd, "Running edge case tests")


def run_segmentation_tests():
    """Run segmentation tests."""
    cmd = ["python", "-m", "pytest", "tests/test_segmentation.py", "-v"]
    return run_command(cmd, "Running segmentation tests")


def run_integration_tests():
    """Run integration tests for complete workflow."""
    cmd = ["python", "-m", "pytest", "tests/test_integration_flow.py", "-v"]
    return run_command(cmd, "Running integration tests")


def run_no_reload_tests():
    """Run no-reload improvements tests."""
    cmd = ["python", "-m", "pytest", "tests/test_no_reload_improvements.py", "-v"]
    return run_command(cmd, "Running no-reload improvements tests")


def run_all_tests(parallel=False, coverage=False, markers=None):
    """Run all tests with optional configurations."""
    cmd = ["python", "-m", "pytest", "tests/", "-v"]
    
    if parallel:
        cmd.extend(["-n", "auto"])  # Run in parallel using all available CPUs
        
    if coverage:
        cmd.extend(["--cov=app", "--cov-report=html", "--cov-report=term-missing"])
        
    if markers:
        for marker in markers:
            cmd.extend(["-m", marker])
    
    description = "Running all tests"
    if parallel:
        description += " (parallel)"
    if coverage:
        description += " with coverage"
    if markers:
        description += f" with markers: {', '.join(markers)}"
        
    return run_command(cmd, description)


def run_performance_tests():
    """Run performance tests specifically."""
    cmd = ["python", "-m", "pytest", "tests/", "-v", "-m", "performance"]
    return run_command(cmd, "Running performance tests")


def run_smoke_tests():
    """Run a quick smoke test to verify basic functionality."""
    cmd = ["python", "-m", "pytest", "tests/test_basic_scenarios.py::TestBasicScenarios::test_linear_funnel_calculation", "-v"]
    return run_command(cmd, "Running smoke test")


def check_test_dependencies():
    """Check if all test dependencies are installed."""
    print("🔍 Checking test dependencies...")
    
    dependencies = ["pytest", "pandas", "numpy", "scipy"]
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


def validate_test_files():
    """Validate that all test files are present and importable."""
    print("🔍 Validating test files...")
    
    test_files = [
        "tests/conftest.py",
        "tests/test_basic_scenarios.py",
        "tests/test_conversion_window.py",
        "tests/test_counting_methods.py",
        "tests/test_edge_cases.py",
        "tests/test_segmentation.py",
        "tests/test_integration_flow.py",
        "tests/test_no_reload_improvements.py"
    ]
    
    missing = []
    for test_file in test_files:
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


def generate_test_report():
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
    
    success = run_command(cmd, "Generating test report")
    
    if success:
        print("\n📋 Test Report Generated:")
        print("   📄 HTML Coverage Report: htmlcov/index.html")
        print("   📄 XML Coverage Report: coverage.xml")
        print("   📄 JUnit XML Report: test-results.xml")
    
    return success


def main():
    """Main function to handle command line arguments and run tests."""
    parser = argparse.ArgumentParser(
        description="Funnel Analytics Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                     # Run all tests
  python run_tests.py --basic            # Run basic scenario tests only
  python run_tests.py --coverage         # Run all tests with coverage
  python run_tests.py --parallel         # Run tests in parallel
  python run_tests.py --marker edge_case # Run edge case tests only
  python run_tests.py --smoke            # Run quick smoke test
  python run_tests.py --report           # Generate comprehensive report
        """
    )
    
    # Test category options
    parser.add_argument("--basic", action="store_true", help="Run basic scenario tests")
    parser.add_argument("--conversion-window", action="store_true", help="Run conversion window tests")
    parser.add_argument("--counting-methods", action="store_true", help="Run counting method tests")
    parser.add_argument("--edge-cases", action="store_true", help="Run edge case tests")
    parser.add_argument("--segmentation", action="store_true", help="Run segmentation tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests for complete workflow")
    parser.add_argument("--no-reload", action="store_true", help="Run no-reload improvements tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    
    # Test execution options
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--marker", action="append", help="Run tests with specific marker (can be used multiple times)")
    
    # Utility options
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test")
    parser.add_argument("--report", action="store_true", help="Generate comprehensive test report")
    parser.add_argument("--check", action="store_true", help="Check dependencies and test files")
    
    args = parser.parse_args()
    
    print("🧪 Funnel Analytics Test Runner")
    print("=" * 50)
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    # Check dependencies if requested
    if args.check or len(sys.argv) == 1:  # Run check by default if no args
        if not check_test_dependencies() or not validate_test_files():
            sys.exit(1)
    
    success = True
    
    # Run specific test categories
    if args.smoke:
        success = run_smoke_tests()
    elif args.basic:
        success = run_basic_tests()
    elif args.conversion_window:
        success = run_conversion_window_tests()
    elif args.counting_methods:
        success = run_counting_method_tests()
    elif args.edge_cases:
        success = run_edge_case_tests()
    elif args.segmentation:
        success = run_segmentation_tests()
    elif args.integration:
        success = run_integration_tests()
    elif args.no_reload:
        success = run_no_reload_tests()
    elif args.performance:
        success = run_performance_tests()
    elif args.report:
        success = generate_test_report()
    elif any([args.parallel, args.coverage, args.marker]):
        # Run all tests with specific options
        success = run_all_tests(
            parallel=args.parallel,
            coverage=args.coverage,
            markers=args.marker
        )
    else:
        # Default: run all tests
        success = run_all_tests()
    
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