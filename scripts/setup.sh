#!/bin/bash

# Exit on error
set -e

if ! command -v uv &> /dev/null; then
    echo "Error: uv is required. Install it from https://docs.astral.sh/uv/."
    exit 1
fi

uv sync
