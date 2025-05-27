# csv-to-llm

A Python package that processes CSV files by sending templated prompts to Claude API in parallel and storing responses in a new column.

## Features

- **Parallel Processing**: Process multiple rows concurrently for faster execution
- **Template Support**: Use flexible prompt templates with column references
- **Caching**: Built-in caching to avoid re-processing identical prompts
- **Progress Tracking**: Real-time progress updates and error handling
- **Flexible Column References**: Support both named columns (`{column_name}`) and positional columns (`{COL1}`, `{COL2}`, etc.)
- **Row Skipping**: Skip rows based on regex patterns
- **Resume Support**: Continue processing from where you left off
- **Embeddings Generation**: Generate text embeddings with `--embeddings`, using providers such as OpenAI

## Installation

### From Source

```bash
git clone <repository-url>
cd csv-to-llm
pip install -e .
```

### For Development

```bash
git clone <repository-url>
cd csv-to-llm
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

## Configuration

Create a `.env` file in your project directory with your Anthropic API key:

```
ANTHROPIC_API_KEY=your_api_key_here
```

## Usage

### Command Line Interface

```bash
csv-to-llm --input input.csv --output output.csv --prompt-template "Summarize: {text_column}" --output-col summary
```

### Basic Example

```bash
csv-to-llm \
  --input data.csv \
  --output results.csv \
  --prompt-template "Translate to French: {english_text}" \
  --output-col french_translation \
  --system "You are a professional translator"
```

### Advanced Example with Parallel Processing

```bash
csv-to-llm \
  --input large_dataset.csv \
  --output processed_data.csv \
  --prompt-template-file examples/sample_prompt.txt \
  --output-col ai_response \
  --parallel 4 \
  --verbose \
  --max-tokens 500
```

### Using Positional Column References

```bash
csv-to-llm \
  --input data.csv \
  --output results.csv \
  --prompt-template "Compare {COL1} and {COL2}" \
  --output-col comparison
```

### Skip Rows Based on Patterns

```bash
csv-to-llm \
  --input data.csv \
  --output results.csv \
  --prompt-template "Process: {text}" \
  --output-col result \
  --skip-rows status "completed|processed"
```

## Command Line Options

- `--input`: Input CSV file path (required)
- `--output`: Output CSV file path (required)
- `--prompt-template`: Prompt template string with column placeholders
- `--prompt-template-file`: Path to file containing prompt template
- `--output-col`: Column name to store Claude's responses (required)
- `--system`: System prompt for Claude (default: "You are a helpful assistant.")
- `--model`: Claude model to use (default: "claude-3-7-sonnet-20250219")
- `--max-tokens`: Maximum tokens for response (default: 1000)
- `--temperature`: Temperature setting (default: 1.0)
- `--parallel`: Number of parallel processes (default: 1)
- `--verbose`: Enable verbose logging
- `--test-first-row`: Process only the first row for testing
- `--skip-rows`: Skip rows matching regex pattern in specified column
- `--embeddings`: Generate embeddings instead of Claude responses
- `--embeddings-provider`: Embedding model provider (default: OpenAI)
- `--embeddings-model`: Embedding model name (default: text-embedding-3-large)

## Python API

You can also use the package programmatically:

```python
from csv_to_llm import process_csv_with_claude

process_csv_with_claude(
    input_csv_path="input.csv",
    output_csv_path="output.csv",
    prompt_template="Summarize: {text_column}",
    output_column="summary",
    system_prompt="You are a helpful assistant.",
    parallel=2
)
```

## Development

### Running Tests

```bash
source venv/bin/activate
python -m pytest tests/
```

### Project Structure

```
csv-to-llm/
├── csv_to_llm/
│   ├── __init__.py
│   ├── core.py          # Main processing logic
│   └── cli.py           # Command line interface
├── tests/
│   ├── __init__.py
│   └── test_csv_to_llm.py
├── examples/
│   └── sample_prompt.txt
├── scripts/
│   ├── setup.sh
│   └── run_tests.sh
├── requirements.txt
├── setup.py
├── pyproject.toml
└── README.md
```

## License

MIT License - see LICENSE file for details.
