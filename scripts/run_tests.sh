#!/bin/bash

# Activate virtual environment and run tests
source venv/bin/activate

# Run all tests with verbose output
echo "Running test suite for csv-to-llm..."
python -m pytest test_csv_to_llm.py -v

# Optional: run tests with coverage if coverage is installed
# python -m pytest test_csv_to_llm.py --cov=csv_to_llm --cov-report=html -v