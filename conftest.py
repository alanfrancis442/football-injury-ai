# conftest.py
#
# Pytest root configuration — adds the project root to sys.path so that
# 'module1', 'module2', 'utils', etc. are importable without installation.
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
