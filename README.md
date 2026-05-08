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
- **Iterative Structured Outputs**: Optionally fill large nested Pydantic models one leaf field at a time
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

By default the tool targets Anthropic's **claude-sonnet-4-20250514** model. Use `--provider openai` for OpenAI's Responses API or `--provider perplexity` for Perplexity Sonar. When you enable structured outputs via `--pydantic-model`, the default automatically switches to OpenAI's **gpt-5.4-mini**, or Perplexity's **pro-search** preset when `--provider perplexity` is set (you can still override `--model` explicitly at any time).

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

Provide a Pydantic model file (or module) and tell the CLI which field to copy into the output column. The request will be routed through OpenAI's Structured Outputs API or Perplexity's JSON Schema structured output API and validated against the schema.

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
- With `--provider perplexity`, `--model` is treated as the Perplexity Responses preset for structured outputs, defaulting to `pro-search`.
- Structured outputs cannot be combined with `--embeddings`.
- Use `--pydantic-model-column-prefix` (for example, `--pydantic-model-column-prefix llm_`) to populate every field from the Pydantic model into its own column such as `llm_category`, `llm_explanation`, etc. Nested objects are flattened into column names such as `llm_pricing_and_provisioning_cost_structure`. Lists and other compound values are serialized as JSON. The entire structured response is also stored as JSON in `--output-col`. This option is mutually exclusive with `--pydantic-model-field`.
- Use `--pydantic-model-iterate` for large or deeply nested schemas with OpenAI. The tool will ask the LLM for each leaf field separately, then reassemble and validate the original Pydantic model. This increases API calls but can improve reliability for complex schemas. Iterative field calls share the `--parallel` budget: when there are fewer active rows than workers, unused worker capacity is used to fill fields concurrently.

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
- `--auto-multi-column`: Let auto mode choose a multi-field schema and populate multiple prefixed output columns when the instruction naturally calls for several values.
- Auto mode supports `--provider openai` and `--provider perplexity`. With no provider, it defaults to OpenAI for backward compatibility.
- With `--provider perplexity`, schema design and structured extraction use the Perplexity Responses API preset from `--model`, defaulting to `pro-search`.
- `--model-websearch` is available with OpenAI and Perplexity auto mode. Any manual prompt or Pydantic arguments are ignored/forbidden in auto mode.
- With `--verbose`, auto mode prints the generated prompt template before row processing starts.
- Auto-generated Pydantic model files are written next to the input CSV under `.csv_to_llm_auto/` and include common typing imports so optional fields can be validated during structured extraction.
- Auto mode validates the generated primary field against the generated model and repairs obvious mismatches before row processing starts.
- Existing blank output columns are normalized before assignment so resumed runs can write string, JSON, or error values even when pandas inferred a numeric dtype.
- Auto-generated validators are normalized for Pydantic v2 when providers emit common v1-style `values.get(...)` field validator code.
- Perplexity structured schemas are normalized for strict validation, including optional fields and nested objects.

### Google Sheets Apps Script

The `apps_script/` folder contains a standalone Google Apps Script version for
Google Sheets custom functions. After installing the script and setting
`PERPLEXITY_API_KEY`, you can call Perplexity directly from cells:

```text
=PERPLEXITY("What is this company's homepage URL? The company name is " & A2, TRUE)
```

The folder also includes `PERPLEXITY_AUTO(headers, sample_rows, input_row,
instruction)`, which adapts the CLI auto-mode idea to Sheets ranges by using
headers and sample rows as context for a structured Perplexity result.

### Caching

LLM calls are cached automatically in `./llm_cache` using joblib. The cache covers plain text generation, Pydantic structured outputs, iterative Pydantic field extraction, and auto-mode schema design. Retries after a failed or empty response bypass the cache for the retry attempt. Embeddings are not cached by this layer.

## Command Line Options

- `--input`: Input CSV file path (required)
- `--output`: Output CSV file path (required)
- `--prompt-template`: Prompt template string with column placeholders
- `--prompt-template-file`: Path to file containing prompt template
- `--output-col`: Column name to store LLM responses (required)
- `--system`: System prompt for the selected LLM (default: "You are a helpful assistant.")
- `--provider`: LLM provider (`anthropic`, `openai`, or `perplexity`; default: `anthropic`)
- `--model`: Model or preset to use (defaults to `claude-sonnet-4-20250514`, `gpt-5.4-mini`, or `sonar-pro` depending on provider; Perplexity structured outputs default to `pro-search`)
- `--model-websearch`: Enable provider web search tools for OpenAI or Perplexity model calls
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
- `--pydantic-model-iterate`: Fill structured output leaf fields one at a time with OpenAI, recursing into nested BaseModel fields before validating the full model
- `--auto`: One-shot instruction to auto-generate a Pydantic model, prompt template, and output column
- `--auto-sample-size`: Number of rows to include when designing the auto schema (default: 5)
- `--auto-multi-column`: Allow auto mode to synthesize multiple output columns when appropriate

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
