#!/bin/bash

# Exit on error
set -e

echo "Detecting highest available Python 3 version..."

# Find available python3 versions (checking 3.13 down to 3.8)
PYTHON_CMD=""
for version in {13..8}; do
    if command -v "python3.$version" &> /dev/null; then
        PYTHON_CMD="python3.$version"
        echo "Found specific version: $PYTHON_CMD"
        break
    fi
done

# Fallback to generic python3
if [ -z "$PYTHON_CMD" ]; then
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        echo "Using generic python3 command."
    fi
fi

# Check if any python3 was found
if [ -z "$PYTHON_CMD" ]; then
    echo "Error: No suitable Python 3 installation found (checked
python3.12 down to python3.8 and generic python3)."
    exit 1
fi

echo "Creating virtual environment using $PYTHON_CMD..."

# Create a virtual environment using the detected command
$PYTHON_CMD -m venv venv

source venv/bin/activate

pip install -r requirements.txt

