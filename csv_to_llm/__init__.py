"""CSV to LLM - Process CSV files with LLM APIs using prompt templates."""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core import process_csv_with_claude, process_csv_with_llm

__all__ = ["process_csv_with_claude", "process_csv_with_llm"]
