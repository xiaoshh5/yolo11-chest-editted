#!/usr/bin/env python
"""One-click test runner. Generates test data, runs all tests.

Usage:
    python run_tests.py              # run all tests
    python run_tests.py --quick      # skip data generation (use cached)
    python run_tests.py -k "test_ctr"  # run matching tests only
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "tests" / "data"

args = sys.argv[1:]

if "--quick" not in args:
    print(":: Generating test data...")
    subprocess.run([sys.executable, str(ROOT / "tests" / "generate_test_data.py")], check=True)

# Remove our custom flags before passing to pytest
pytest_args = [a for a in args if a != "--quick"]
cmd = [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-v"] + pytest_args

print(f":: Running: {' '.join(cmd)}")
sys.exit(subprocess.run(cmd).returncode)
