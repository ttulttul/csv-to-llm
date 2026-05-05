# csv-to-llm

A Python package that processes CSV files by sending templated prompts to OpenAI, Anthropic, or Perplexity APIs in parallel and storing responses in a new column.

## Features

- **Parallel Processing**: Process multiple rows concurrently for faster execution
- **Template Support**: Use flexible prompt templates with column references
- **Caching**: Built-in caching to avoid re-processing identical prompts
- **Progress Tracking**: Real-time progress updates and error handling
- **Flexible Column References**: Support both named columns (`{column_name}`) and positional columns (`{COL1}`, `{COL2}`, etc.)
- **Row Skipping**: Skip rows based on regex patterns
- **Resume Support**: Continue processing from where you left off
- **Embeddings Generation**: Generate text embeddings with `--embeddings`, using providers such as OpenAI
- **Automatic Retries**: Re-attempt failed or empty generations up to a configurable number of times
- **Provider Selection**: Generate text with Anthropic Messages, OpenAI Responses, or Perplexity Sonar
- **Structured Outputs**: Enforce schemas defined in user-supplied Pydantic models via OpenAI's structured responses
- **Auto Schema Mode**: Provide a single instruction via `--auto` and let the tool synthesize a Pydantic model and prompt automatically

## Installation

### From Source

```bash
git clone <repository-url>
cd csv-to-llm
pip install -e .
```

### With uv

```bash
git clone <repository-url>
cd csv-to-llm
uv sync
```

## Configuration

Create a `.env` file in your project directory with the API keys you need:

```
ANTHROPIC_API_KEY=your_api_key_here
OPENAI_API_KEY=your_openai_key_here
PERPLEXITY_API_KEY=your_perplexity_key_here
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

By default the tool targets Anthropic's **claude-sonnet-4-20250514** model. Use `--provider openai` for OpenAI's Responses API or `--provider perplexity` for Perplexity Sonar. When you enable structured outputs via `--pydantic-model`, the default automatically switches to OpenAI's **gpt-5.2** (you can still override `--model` explicitly at any time).

```bash
csv-to-llm \
  --provider openai \
  --model gpt-5.2 \
  --input data.csv \
  --output results.csv \
  --prompt-template "Summarize: {text}" \
  --output-col summary
```

```bash
csv-to-llm \
  --provider perplexity \
  --model sonar-pro \
  --input data.csv \
  --output results.csv \
  --prompt-template "Research and enrich this company: {company_name}" \
  --output-col enrichment
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

### Structured Outputs with Pydantic

Provide a Pydantic model file (or module) and tell the CLI which field to copy into the output column. The request will be routed through OpenAI's Structured Outputs API and validated against the schema.

The `examples/` directory includes two ready-to-use email classification schemas:

- `examples/email_category_model.py`: richer classification with category, action-required flag, project, and custom labels.
- `examples/email_classification_broad_model.py`: broader high-level category plus optional labels.

```bash
csv-to-llm \
  --input input.csv \
  --output output.csv \
  --prompt-template "Categorize this email using the supplied schema. Subject: {subject}" \
  --output-col content_category \
  --provider openai \
  --model gpt-5.2 \
  --pydantic-model examples/email_category_model.py \
  --pydantic-model-class EmailCategory \
  --pydantic-model-field content_category
```

If you instead want every Pydantic field to land in the CSV, skip `--pydantic-model-field` and supply a prefix:

```bash
csv-to-llm \
  --input input.csv \
  --output output.csv \
  --prompt-template "Categorize the subject using the supplied schema: {subject}" \
  --output-col structured_payload \
  --provider openai \
  --pydantic-model examples/email_classification_broad_model.py \
  --pydantic-model-class EmailClassification \
  --pydantic-model-column-prefix llm_
```

Notes:

- `--pydantic-model-field` is required and must exist on the BaseModel class.
- `--model` should reference an OpenAI model that supports Structured Outputs.
- Structured outputs cannot be combined with `--embeddings`.
- Use `--pydantic-model-column-prefix` (for example, `--pydantic-model-column-prefix llm_`) to populate every field from the Pydantic model into its own column such as `llm_category`, `llm_explanation`, etc.; the entire structured response is stored (as JSON) in `--output-col`. This option is mutually exclusive with `--pydantic-model-field`.

### Auto Mode

Tired of hand-writing schemas? Use `--auto` with a plain-English instruction and let the tool design a Pydantic model, prompt template, and output column for you. The CLI samples a few rows from your CSV, asks the LLM to craft a schema, writes that schema to disk, and then runs the structured-output pipeline automatically.

```bash
csv-to-llm \
  --auto "Classify the email subjects into sensible categories" \
  --input input.csv \
  --output output.csv
```

Options:

- `--auto-sample-size`: Number of rows (default 5) included in the schema-design request.
- Auto mode always uses OpenAI structured outputs, so `OPENAI_API_KEY` must be set. Any manual prompt or Pydantic arguments are ignored/forbidden in this mode.

## Command Line Options

- `--input`: Input CSV file path (required)
- `--output`: Output CSV file path (required)
- `--prompt-template`: Prompt template string with column placeholders
- `--prompt-template-file`: Path to file containing prompt template
- `--output-col`: Column name to store LLM responses (required)
- `--system`: System prompt for the selected LLM (default: "You are a helpful assistant.")
- `--provider`: LLM provider (`anthropic`, `openai`, or `perplexity`; default: `anthropic`)
- `--model`: Model to use (defaults to `claude-sonnet-4-20250514`, `gpt-5.2`, or `sonar-pro` depending on provider)
- `--max-tokens`: Maximum tokens for response (default: 1000)
- `--temperature`: Temperature setting (default: 1.0)
- `--max-retries`: Number of retries after the initial attempt if the LLM response fails validation or is empty (default: 2 retries)
- `--parallel`: Number of parallel processes (default: 1)
- `--verbose`: Enable verbose logging
- `--test-first-row`: Process only the first row for testing
- `--skip-rows`: Skip rows matching regex pattern in specified column
- `--embeddings`: Generate embeddings instead of Claude responses
- `--embeddings-provider`: Embedding model provider (default: OpenAI)
- `--embeddings-model`: Embedding model name (default: text-embedding-3-large)
- `--pydantic-model`: Path or module reference for a Pydantic BaseModel to enable structured outputs
- `--pydantic-model-class`: BaseModel class name when multiple models are defined in the same module
- `--pydantic-model-field`: Field on the BaseModel whose value is stored in `--output-col` (required unless `--pydantic-model-column-prefix` is provided)
- `--pydantic-model-column-prefix`: When set, populate additional columns for every BaseModel field using the provided prefix, and store the entire structured response (as JSON) in `--output-col`; mutually exclusive with `--pydantic-model-field`
- `--auto`: One-shot instruction to auto-generate a Pydantic model, prompt template, and output column
- `--auto-sample-size`: Number of rows to include when designing the auto schema (default: 5)

## Python API

You can also use the package programmatically:

```python
from csv_to_llm import process_csv_with_llm

process_csv_with_llm(
    input_csv_path="input.csv",
    output_csv_path="output.csv",
    prompt_template="Summarize: {text_column}",
    output_column="summary",
    system_prompt="You are a helpful assistant.",
    provider="openai",
    model="gpt-5.2",
    parallel=2,
    max_retries=2,
    # pydantic_model_path="models/schema.py",
    # pydantic_model_class="StructuredResponse",
    # pydantic_model_field="answer",
)
```

When the optional Pydantic arguments are supplied, OpenAI Structured Outputs are used automatically and `OPENAI_API_KEY` must be set.

## Development

### Running Tests

```bash
uv run pytest tests/
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
├── uv.lock
├── requirements.txt
├── setup.py
├── pyproject.toml
└── README.md
```

## License

MIT License - see LICENSE file for details.
