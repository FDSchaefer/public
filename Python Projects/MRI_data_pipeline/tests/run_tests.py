#!/usr/bin/env python3
"""
Test runner for all unit tests
"""
import sys
import unittest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Discover and run all tests
loader = unittest.TestLoader()
start_dir = Path(__file__).parent
suite = loader.discover(start_dir, pattern='test_*.py')

runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

# Exit with error code if tests failed
sys.exit(0 if result.wasSuccessful() else 1)
