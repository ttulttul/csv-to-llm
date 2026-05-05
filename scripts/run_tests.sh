#!/bin/bash

# Run all tests with verbose output
echo "Running test suite for csv-to-llm..."
uv run pytest tests/ -v

# Optional: run tests with coverage if coverage is installed
# uv run pytest tests/ --cov=csv_to_llm --cov-report=html -v
