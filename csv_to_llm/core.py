import os
import pandas as pd
import anthropic
import time
from dotenv import load_dotenv
import concurrent.futures
import re
import logging
import importlib
import importlib.util
import inspect
import hashlib
import json
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Type, Tuple, List, Callable, Dict, Any, Iterable
from joblib import Memory
from tqdm import tqdm
from colorama import Fore, Style, init
from openai import OpenAI
from pydantic import BaseModel
from .embeddings import get_embedding

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Module-level logger
logger = logging.getLogger(__name__)

# --- Joblib Cache Setup ---
# Define cache directory (you might want to make this configurable or use a temporary dir)
cachedir = './llm_cache'
memory = Memory(cachedir, verbose=0)
# --- End Cache Setup ---

PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI = "openai"
PROVIDER_PERPLEXITY = "perplexity"
SUPPORTED_LLM_PROVIDERS = (PROVIDER_ANTHROPIC, PROVIDER_OPENAI, PROVIDER_PERPLEXITY)

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
DEFAULT_OPENAI_MODEL = "gpt-5.2"
DEFAULT_PERPLEXITY_MODEL = "sonar-pro"
DEFAULT_PROVIDER = PROVIDER_ANTHROPIC
PERPLEXITY_BASE_URL = "https://api.perplexity.ai"

_thread_local_clients = threading.local()


@dataclass(frozen=True)
class StructuredOutputConfig:
    """Runtime configuration for Pydantic-based structured outputs."""

    model_reference: str
    class_name: Optional[str]
    output_field: Optional[str]
    llm_model: str
    max_output_tokens: int
    temperature: float
    system_prompt: str


@dataclass
class RowProcessingArgs:
    """Data passed to worker processes/sequential loop for each row."""

    index: int
    row_data: dict
    required_columns: List[str]
    prompt_template: str
    model: str
    max_tokens: int
    temperature: float
    system_prompt: str
    output_column: str
    structured_config: Optional[StructuredOutputConfig]
    max_retries: int
    column_prefix: Optional[str]
    provider: str = DEFAULT_PROVIDER


def _get_thread_local_openai_client() -> OpenAI:
    """Fetch (or create) a thread-local OpenAI client for reuse."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment variables")

    client = getattr(_thread_local_clients, "openai_client", None)
    if client is None:
        client = OpenAI(api_key=api_key)
        _thread_local_clients.openai_client = client
    return client


def _get_thread_local_perplexity_client() -> OpenAI:
    """Fetch (or create) a thread-local Perplexity client for reuse."""

    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise RuntimeError("PERPLEXITY_API_KEY not found in environment variables")

    client = getattr(_thread_local_clients, "perplexity_client", None)
    if client is None:
        client = OpenAI(api_key=api_key, base_url=PERPLEXITY_BASE_URL)
        _thread_local_clients.perplexity_client = client
    return client


def _get_thread_local_anthropic_client() -> anthropic.Anthropic:
    """Fetch (or create) a thread-local Anthropic client for reuse."""

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not found in environment variables")

    client = getattr(_thread_local_clients, "anthropic_client", None)
    if client is None:
        client = anthropic.Anthropic(api_key=api_key)
        _thread_local_clients.anthropic_client = client
    return client


def _normalize_model_reference(model_reference: str) -> str:
    """Ensure file-based references are absolute to keep cache keys stable."""

    if os.path.exists(model_reference):
        return os.path.abspath(model_reference)
    return model_reference


def _split_model_reference(model_reference: str, explicit_class_name: Optional[str]) -> Tuple[str, Optional[str]]:
    """Break `path:ClassName` references into components and enforce overrides."""

    reference = model_reference
    class_name = explicit_class_name

    if ':' in model_reference:
        reference, embedded_class = model_reference.rsplit(':', 1)
        if class_name and class_name != embedded_class:
            raise ValueError(
                "Conflicting class definitions supplied via --pydantic-model and --pydantic-model-class."
            )
        class_name = class_name or embedded_class

    normalized_reference = _normalize_model_reference(reference.strip())
    return normalized_reference, class_name


def _import_module_from_reference(reference: str):
    """Import a module either from a filesystem path or dot-path string."""

    if os.path.exists(reference):
        module_name = f"csv_to_llm_user_model_{hashlib.sha1(reference.encode('utf-8')).hexdigest()}"
        spec = importlib.util.spec_from_file_location(module_name, reference)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module from '{reference}'")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return importlib.import_module(reference)


def _find_pydantic_model_class(module, class_name: Optional[str]) -> Type[BaseModel]:
    """Locate the desired BaseModel subclass inside an imported module."""

    candidates = [
        obj for obj in vars(module).values()
        if inspect.isclass(obj) and issubclass(obj, BaseModel)
    ]

    if class_name:
        for candidate in candidates:
            if candidate.__name__ == class_name:
                return candidate
        raise ValueError(
            f"Pydantic model '{class_name}' was not found in module '{module.__name__}'"
        )

    if not candidates:
        raise ValueError("No Pydantic BaseModel subclasses found in the provided module")

    if len(candidates) > 1:
        raise ValueError(
            "Multiple Pydantic BaseModel subclasses found. Specify --pydantic-model-class explicitly."
        )

    return candidates[0]


@lru_cache(maxsize=None)
def _get_pydantic_model_class(reference: str, class_name: Optional[str]) -> Type[BaseModel]:
    """Import and cache the requested Pydantic BaseModel."""

    module = _import_module_from_reference(reference)
    return _find_pydantic_model_class(module, class_name)


def build_structured_output_config(
    model_reference: str,
    model_class_name: Optional[str],
    output_field: Optional[str],
    llm_model: str,
    max_tokens: int,
    temperature: float,
    system_prompt: str,
) -> StructuredOutputConfig:
    """Validate user-supplied structured output arguments and build config."""

    if not model_reference:
        raise ValueError("--pydantic-model must point to a valid Python module or file")

    normalized_reference, resolved_class = _split_model_reference(model_reference, model_class_name)
    config = StructuredOutputConfig(
        model_reference=normalized_reference,
        class_name=resolved_class,
        output_field=output_field,
        llm_model=llm_model,
        max_output_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
    )

    model_cls = _get_pydantic_model_class(config.model_reference, config.class_name)
    if output_field and output_field not in model_cls.model_fields:
        raise ValueError(
            f"Field '{output_field}' not found on Pydantic model '{model_cls.__name__}'"
        )

    return config


def call_openai_structured(prompt_value: str, structured_config: StructuredOutputConfig, openai_client: Optional[OpenAI] = None) -> str:
    """Call OpenAI with Structured Outputs enabled and return the parsed BaseModel."""

    model_cls = _get_pydantic_model_class(structured_config.model_reference, structured_config.class_name)
    client = openai_client or OpenAI()

    response = client.responses.parse(
        model=structured_config.llm_model,
        input=[
            {"role": "system", "content": structured_config.system_prompt},
            {"role": "user", "content": prompt_value},
        ],
        text_format=model_cls,
        temperature=structured_config.temperature,
        max_output_tokens=structured_config.max_output_tokens,
    )

    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError("Structured output parsing failed: no parsed content returned")

    return parsed


def _extract_openai_response_text(response: Any) -> str:
    """Extract text from an OpenAI Responses API response."""

    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    fragments: List[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                fragments.append(text)
    return "".join(fragments)


def _serialize_structured_value(value: Any) -> str:
    """Convert arbitrary structured field data to a CSV-friendly string."""

    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def _flatten_structured_fields(value: Any, prefix: str) -> Dict[str, str]:
    """Flatten nested structured output values into CSV column names."""

    if isinstance(value, BaseModel):
        value = value.model_dump()

    if isinstance(value, dict):
        flattened: Dict[str, str] = {}
        for key, nested_value in value.items():
            key_prefix = f"{prefix}{key}"
            if isinstance(nested_value, BaseModel):
                nested_value = nested_value.model_dump()
            if isinstance(nested_value, dict):
                flattened.update(_flatten_structured_fields(nested_value, f"{key_prefix}_"))
            else:
                flattened[key_prefix] = _serialize_structured_value(nested_value)
        return flattened

    return {prefix.rstrip("_"): _serialize_structured_value(value)}


def invoke_with_retries(max_retries: int, request_fn: Callable[[int], str]) -> str:
    """Execute request_fn with retry semantics. request_fn receives the attempt index."""

    attempts = max(max_retries, 0) + 1
    last_error: Optional[Exception] = None

    for attempt in range(attempts):
        try:
            response = request_fn(attempt)
            if response is None or (isinstance(response, str) and not response.strip()):
                raise RuntimeError("LLM returned an empty response")
            return response
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"LLM call failed after {attempts} attempts: {last_error}"
    )

def _call_claude_api(client, model, max_tokens, temperature, system_prompt, prompt_value):
    """Internal helper that performs the actual Anthropic API call."""

    logger.info("Sending request to Claude model '%s' with system prompt '%s'. Prompt: %s",
                model, system_prompt, prompt_value)

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_value
                    }
                ]
            }
        ]
    )
    if message.content and hasattr(message.content[0], 'text'):
        response_text = message.content[0].text
        logger.info("Received response from Claude. Response: %s", response_text)
        return response_text

    warning_msg = f"Warning: Unexpected API response structure. Content: {message.content}"
    print(warning_msg)
    logger.warning("Unexpected API response structure. Content: %s", message.content)
    return ""


def _call_openai_api(client, model, max_tokens, temperature, system_prompt, prompt_value):
    """Internal helper that performs an OpenAI Responses API call."""

    logger.info("Sending request to OpenAI model '%s'. Prompt: %s", model, prompt_value)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_value},
        ],
        max_output_tokens=max_tokens,
        temperature=temperature,
    )
    response_text = _extract_openai_response_text(response)
    logger.info("Received response from OpenAI. Response: %s", response_text)
    return response_text


def _call_perplexity_api(client, model, max_tokens, temperature, system_prompt, prompt_value):
    """Internal helper that performs a Perplexity Sonar chat completion call."""

    logger.info("Sending request to Perplexity model '%s'. Prompt: %s", model, prompt_value)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_value},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if response.choices and response.choices[0].message.content:
        response_text = response.choices[0].message.content
        logger.info("Received response from Perplexity. Response: %s", response_text)
        return response_text

    warning_msg = f"Warning: Unexpected API response structure. Choices: {response.choices}"
    print(warning_msg)
    logger.warning("Unexpected Perplexity API response structure. Choices: %s", response.choices)
    return ""


# --- Cached API Call ---
@memory.cache(ignore=['client'])
def call_claude_api_cached(client, model, max_tokens, temperature, system_prompt, prompt_value):
    return _call_claude_api(client, model, max_tokens, temperature, system_prompt, prompt_value)


def call_claude_api_uncached(client, model, max_tokens, temperature, system_prompt, prompt_value):
    return _call_claude_api(client, model, max_tokens, temperature, system_prompt, prompt_value)


@memory.cache(ignore=['client'])
def call_openai_api_cached(client, model, max_tokens, temperature, system_prompt, prompt_value):
    return _call_openai_api(client, model, max_tokens, temperature, system_prompt, prompt_value)


def call_openai_api_uncached(client, model, max_tokens, temperature, system_prompt, prompt_value):
    return _call_openai_api(client, model, max_tokens, temperature, system_prompt, prompt_value)


@memory.cache(ignore=['client'])
def call_perplexity_api_cached(client, model, max_tokens, temperature, system_prompt, prompt_value):
    return _call_perplexity_api(client, model, max_tokens, temperature, system_prompt, prompt_value)


def call_perplexity_api_uncached(client, model, max_tokens, temperature, system_prompt, prompt_value):
    return _call_perplexity_api(client, model, max_tokens, temperature, system_prompt, prompt_value)
# --- End Cached API Call ---


def _get_provider_client(provider: str):
    """Return a thread-local client for the selected provider."""

    if provider == PROVIDER_ANTHROPIC:
        return _get_thread_local_anthropic_client()
    if provider == PROVIDER_OPENAI:
        return _get_thread_local_openai_client()
    if provider == PROVIDER_PERPLEXITY:
        return _get_thread_local_perplexity_client()
    raise RuntimeError(f"Unsupported provider '{provider}'")


def call_llm_api(
    provider: str,
    client: Any,
    model: str,
    max_tokens: int,
    temperature: float,
    system_prompt: str,
    prompt_value: str,
    attempt: int,
) -> str:
    """Call the selected LLM provider, using the cache only for the first attempt."""

    cache = attempt == 0
    if provider == PROVIDER_ANTHROPIC:
        request_fn = call_claude_api_cached if cache else call_claude_api_uncached
    elif provider == PROVIDER_OPENAI:
        request_fn = call_openai_api_cached if cache else call_openai_api_uncached
    elif provider == PROVIDER_PERPLEXITY:
        request_fn = call_perplexity_api_cached if cache else call_perplexity_api_uncached
    else:
        raise ValueError(f"Unsupported provider '{provider}'")

    return request_fn(
        client=client,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        prompt_value=prompt_value,
    )


def process_single_row(task: RowProcessingArgs):
    """Processes a single row, supporting both Anthropic and OpenAI structured flows."""

    structured_config = task.structured_config
    column_prefix = task.column_prefix

    try:
        if structured_config:
            openai_client = _get_thread_local_openai_client()
            provider_client = None
        else:
            openai_client = None
            provider_client = _get_provider_client(task.provider)
    except RuntimeError as exc:
        return task.index, None, str(exc)

    format_dict = {col: task.row_data.get(col) for col in task.required_columns}
    if any(pd.isna(format_dict.get(col)) for col in task.required_columns):
        return task.index, None, "Missing data for prompt template"

    try:
        prompt_value = task.prompt_template.format(**format_dict)
    except KeyError as e:
        return task.index, None, f"Formatting error (likely missing column in template): {e}"
    except TypeError as e:
        return task.index, None, f"Formatting error (likely type issue): {e}"

    try:
        def _request(attempt: int):
            if structured_config:
                parsed = call_openai_structured(
                    prompt_value=prompt_value,
                    structured_config=structured_config,
                    openai_client=openai_client,
                )
                if structured_config.output_field:
                    target_value = getattr(parsed, structured_config.output_field, None)
                else:
                    target_value = parsed
                value = _serialize_structured_value(target_value)
                if not value.strip():
                    raise RuntimeError("Structured output field empty")
                return parsed, value
            return call_llm_api(
                provider=task.provider,
                client=provider_client,
                model=task.model,
                max_tokens=task.max_tokens,
                temperature=task.temperature,
                system_prompt=task.system_prompt,
                prompt_value=prompt_value,
                attempt=attempt,
            )

        response = invoke_with_retries(task.max_retries, _request)

        extra_fields: Optional[Dict[str, str]] = None
        output_value: str

        if structured_config:
            parsed_model, output_value = response
            if column_prefix:
                dumped = parsed_model.model_dump()
                extra_fields = _flatten_structured_fields(dumped, column_prefix)
        else:
            output_value = response

        return task.index, {"output_value": output_value, "extra_fields": extra_fields}, None
    except Exception as e:
        return task.index, None, f"API Error: {e}"


def _iter_row_processing_args(
    df: pd.DataFrame,
    indices: List[int],
    positional_cols: List[str],
    required_columns: List[str],
    prompt_template: str,
    provider: str,
    model: str,
    max_tokens: int,
    temperature: float,
    system_prompt: str,
    output_column: str,
    structured_output_config: Optional[StructuredOutputConfig],
    max_retries: int,
    column_prefix: Optional[str],
) -> Iterable[RowProcessingArgs]:
    """Yield RowProcessingArgs lazily so tasks can be streamed to workers."""

    for index in indices:
        row = df.loc[index]
        row_dict = row.to_dict()
        for col in positional_cols:
            col_idx = int(col[3:]) - 1  # COL1 -> index 0
            if 0 <= col_idx < len(row):
                row_dict[col] = row.iloc[col_idx]
            else:
                row_dict[col] = pd.NA

        yield RowProcessingArgs(
            index=index,
            row_data=row_dict,
            required_columns=required_columns,
            prompt_template=prompt_template,
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            output_column=output_column,
            structured_config=structured_output_config,
            max_retries=max_retries,
            column_prefix=column_prefix,
        )


def process_csv_with_claude(
    input_csv_path,
    output_csv_path,
    prompt_template,
    output_column,
    system_prompt="You are a world-class poet. Respond only with short poems.",
    provider=None,
    model=None,
    max_tokens=1000,
    temperature=1,
    test_first_row=False,
    parallel=1,
    skip_column=None,
    skip_regex=None,
    pydantic_model_path=None,
    pydantic_model_class=None,
    pydantic_model_field=None,
    pydantic_model_column_prefix=None,
    max_retries=2,
):
    """
    Process a CSV file by sending templated row data to an LLM API and saving responses to another column.
    
    Args:
        input_csv_path (str): Path to the input CSV file
        output_csv_path (str): Path to save the output CSV file
        prompt_template (str): A template string for the prompt, potentially containing column names in curly braces (e.g., "Summarize: {text_column}")
        output_column (str): Name of the column to store Claude's responses
        system_prompt (str): System prompt for the LLM API
        provider (str): LLM provider to use: anthropic, openai, or perplexity
        model (str): Model to use
        max_tokens (int): Maximum tokens for Claude's response
        temperature (float): Temperature setting for response randomness
        test_first_row (bool): If True, only process the first valid row and exit.
        parallel (int): Number of parallel processes to use.
        skip_column (str, optional): Column name to check for skipping rows. Defaults to None.
        skip_regex (str, optional): Regex pattern to match in skip_column for skipping. Defaults to None.
        pydantic_model_path (str, optional): Path or module reference to a Python file containing a Pydantic BaseModel for structured outputs.
        pydantic_model_class (str, optional): Name of the BaseModel subclass to use when the module contains multiple models.
        pydantic_model_field (str, optional): Field on the Pydantic model whose value should populate the output column (required unless column prefix is provided).
        pydantic_model_column_prefix (str, optional): When provided, populate additional columns for every Pydantic field using this prefix and serialize the entire model into the output column.
        max_retries (int): Number of retries after the initial attempt if the LLM call fails or returns empty output.
    """
    # Load environment variables from .env file (needed for main process checks)
    load_dotenv()

    if provider is None and pydantic_model_path:
        provider = PROVIDER_OPENAI
    provider = (provider or DEFAULT_PROVIDER).lower()
    if provider not in SUPPORTED_LLM_PROVIDERS:
        raise ValueError(
            f"Unsupported provider '{provider}'. Choose one of: {', '.join(SUPPORTED_LLM_PROVIDERS)}"
        )
    if pydantic_model_path and provider != PROVIDER_OPENAI:
        raise ValueError("Structured outputs currently require provider='openai'")
    
    if model is None:
        if pydantic_model_path or provider == PROVIDER_OPENAI:
            model = DEFAULT_OPENAI_MODEL
        elif provider == PROVIDER_PERPLEXITY:
            model = DEFAULT_PERPLEXITY_MODEL
        else:
            model = DEFAULT_ANTHROPIC_MODEL

    max_retries = max(0, int(max_retries))

    if pydantic_model_column_prefix and not pydantic_model_path:
        raise ValueError("--pydantic-model-column-prefix requires --pydantic-model")

    if pydantic_model_path:
        if pydantic_model_column_prefix and pydantic_model_field:
            raise ValueError("--pydantic-model-field cannot be used with --pydantic-model-column-prefix")
        if not pydantic_model_column_prefix and not pydantic_model_field:
            raise ValueError("--pydantic-model-field is required unless --pydantic-model-column-prefix is provided")

    # Read the CSV file
    try:
        df = pd.read_csv(input_csv_path)
        print(f"{Fore.GREEN}✓{Style.RESET_ALL} Successfully loaded CSV with {Fore.CYAN}{len(df)}{Style.RESET_ALL} rows")
    except Exception as e:
        print(f"{Fore.RED}✗{Style.RESET_ALL} Error loading CSV: {e}")
        return

    # Extract required placeholders from the prompt template
    required_columns = re.findall(r'\{([^}]+)\}', prompt_template)
    if not required_columns:
        raise ValueError(
            "Prompt template must contain at least one column identifier enclosed in curly braces, "
            "e.g., {column_name}. None were found."
        )

    # Split placeholders into named vs positional (COL\d+)
    positional_cols = [col for col in required_columns if re.fullmatch(r'COL\d+', col)]
    named_cols = [col for col in required_columns if col not in positional_cols]

    # Validate only the named columns against the dataframe headers
    missing_columns = [col for col in named_cols if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in CSV for prompt template: {', '.join(missing_columns)}"
        )

    # Create the output column if it doesn't exist
    if output_column not in df.columns:
        df[output_column] = pd.NA # Use pandas NA for better type handling

    # Identify rows to process (where output column is null/NA)
    rows_to_process_indices = df[df[output_column].isna()].index
    total_rows_to_process = len(rows_to_process_indices)
    total_rows = len(df)
    print(f"{Fore.YELLOW}📊{Style.RESET_ALL} Found {Fore.CYAN}{total_rows_to_process}{Style.RESET_ALL} rows needing processing out of {Fore.CYAN}{total_rows}{Style.RESET_ALL} total rows")

    processed_count = 0
    skipped_count = 0
    skip_pattern = None

    # --- Skip Rows Logic ---
    if skip_column and skip_regex:
        if skip_column not in df.columns:
            raise ValueError(f"Skip column '{skip_column}' not found in the CSV.")
        try:
            skip_pattern = re.compile(skip_regex)
            print(f"{Fore.YELLOW}⚠️{Style.RESET_ALL} Will skip rows where column '{skip_column}' matches regex: '{skip_regex}'")
        except re.error as e:
            raise ValueError(f"Invalid regex pattern provided for skipping: {e}")

        rows_to_actually_process_indices = []
        for index in rows_to_process_indices:
            value_to_check = df.loc[index, skip_column]
            # Ensure value is string for regex search, handle NA/None safely
            value_str = str(value_to_check)
            if pd.isna(value_to_check): # Treat actual NA/None as non-matching
                 rows_to_actually_process_indices.append(index)
                 continue

            if skip_pattern.search(value_str):
                df.at[index, output_column] = "" # Set output to empty string for skipped rows
                skipped_count += 1
            else:
                rows_to_actually_process_indices.append(index) # Keep this row for processing

        if skipped_count > 0:
            print(f"{Fore.YELLOW}⏭️{Style.RESET_ALL} Skipped {Fore.CYAN}{skipped_count}{Style.RESET_ALL} rows based on regex match")
            # Update the list of indices to only those not skipped
            rows_to_process_indices = rows_to_actually_process_indices
            print(f"{Fore.GREEN}🚀{Style.RESET_ALL} Proceeding to process {Fore.CYAN}{len(rows_to_process_indices)}{Style.RESET_ALL} remaining rows")
    # --- End Skip Rows Logic ---

    structured_output_config = None
    provider_client = None
    openai_client = None

    if pydantic_model_path:
        structured_output_config = build_structured_output_config(
            model_reference=pydantic_model_path,
            model_class_name=pydantic_model_class,
            output_field=pydantic_model_field,
            llm_model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables in main process")
        if parallel == 1:
            openai_client = OpenAI(api_key=openai_api_key)
    else:
        required_key = {
            PROVIDER_ANTHROPIC: "ANTHROPIC_API_KEY",
            PROVIDER_OPENAI: "OPENAI_API_KEY",
            PROVIDER_PERPLEXITY: "PERPLEXITY_API_KEY",
        }[provider]
        api_key = os.getenv(required_key)
        if not api_key:
            raise ValueError(f"{required_key} not found in environment variables in main process")
        if parallel == 1:
            if provider == PROVIDER_ANTHROPIC:
                provider_client = anthropic.Anthropic(api_key=api_key)
            elif provider == PROVIDER_OPENAI:
                provider_client = OpenAI(api_key=api_key)
            else:
                provider_client = OpenAI(api_key=api_key, base_url=PERPLEXITY_BASE_URL)

    rows_to_process_indices = list(rows_to_process_indices)

    if test_first_row and rows_to_process_indices:
        print(f"{Fore.CYAN}🧪{Style.RESET_ALL} Test mode: Preparing only the first valid row for processing")
        rows_to_process_indices = [rows_to_process_indices[0]]

    total_tasks = len(rows_to_process_indices)
    if total_tasks == 0:
        print(f"{Fore.GREEN}✓{Style.RESET_ALL} No rows require processing")
        return

    if parallel > 1:
        print(f"{Fore.BLUE}⚡{Style.RESET_ALL} Starting parallel processing with {Fore.CYAN}{parallel}{Style.RESET_ALL} workers")

        def handle_future(fut, pbar):
            nonlocal processed_count
            idx: Optional[int] = None
            try:
                idx, response_payload, error = fut.result()
                if error:
                    tqdm.write(f"{Fore.RED}✗{Style.RESET_ALL} Error processing row {idx + 1}: {error}")
                    df.at[idx, output_column] = f"ERROR: {error}"
                else:
                    df.at[idx, output_column] = response_payload["output_value"]
                    extra_fields = response_payload.get("extra_fields")
                    if extra_fields:
                        for col_name, value in extra_fields.items():
                            if col_name not in df.columns:
                                df[col_name] = pd.NA
                            df.at[idx, col_name] = value
                    processed_count += 1
            except Exception as exc:
                fallback_idx = idx if idx is not None else getattr(fut, "row_index", None)
                label = f"row {fallback_idx + 1}" if fallback_idx is not None else "a row"
                tqdm.write(f"{Fore.RED}✗{Style.RESET_ALL} Error processing {label}: {exc}")
                if fallback_idx is not None:
                    df.at[fallback_idx, output_column] = f"ERROR: Future execution failed - {exc}"
            finally:
                pbar.update(1)

        task_iterator = _iter_row_processing_args(
            df=df,
            indices=rows_to_process_indices,
            positional_cols=positional_cols,
            required_columns=required_columns,
            prompt_template=prompt_template,
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            output_column=output_column,
            structured_output_config=structured_output_config,
            max_retries=max_retries,
            column_prefix=pydantic_model_column_prefix,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            pending = set()
            with tqdm(total=total_tasks, desc="Processing rows", unit="row",
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                for task in task_iterator:
                    future = executor.submit(process_single_row, task)
                    setattr(future, "row_index", task.index)
                    pending.add(future)
                    if len(pending) >= parallel:
                        done_future = next(concurrent.futures.as_completed(pending))
                        pending.remove(done_future)
                        handle_future(done_future, pbar)

                while pending:
                    done_future = next(concurrent.futures.as_completed(pending))
                    pending.remove(done_future)
                    handle_future(done_future, pbar)

        print(f"{Fore.GREEN}💾{Style.RESET_ALL} Parallel processing finished. Saving results...")
        df.to_csv(output_csv_path, index=False)

    else: # Sequential processing (parallel == 1)
        print(f"{Fore.BLUE}🔄{Style.RESET_ALL} Starting sequential processing")
        if structured_output_config:
            if openai_client is None:
                openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            if provider_client is None:
                provider_client = _get_provider_client(provider)

        # Use tqdm for progress tracking
        task_iterator = _iter_row_processing_args(
            df=df,
            indices=rows_to_process_indices,
            positional_cols=positional_cols,
            required_columns=required_columns,
            prompt_template=prompt_template,
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            output_column=output_column,
            structured_output_config=structured_output_config,
            max_retries=max_retries,
            column_prefix=pydantic_model_column_prefix,
        )

        with tqdm(total=total_tasks, desc="Processing rows", unit="row", 
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            for task in task_iterator:
                format_dict = {col: task.row_data.get(col) for col in task.required_columns}
                if any(pd.isna(val) for val in format_dict.values()):
                    tqdm.write(f"{Fore.YELLOW}⏭️{Style.RESET_ALL} Skipping row {task.index+1} (missing data for prompt template)")
                    pbar.update(1)
                    continue

                try:
                    prompt_value = task.prompt_template.format(**format_dict)
                except KeyError as e:
                    tqdm.write(f"{Fore.YELLOW}⏭️{Style.RESET_ALL} Skipping row {task.index+1} due to formatting error: {e}")
                    pbar.update(1)
                    continue

                pbar.set_postfix_str(f"Row {task.index+1}")

                def _request(attempt: int):
                    if structured_output_config:
                        parsed = call_openai_structured(
                            prompt_value=prompt_value,
                            structured_config=structured_output_config,
                            openai_client=openai_client,
                        )
                        if structured_output_config.output_field:
                            target_value = getattr(parsed, structured_output_config.output_field, None)
                        else:
                            target_value = parsed
                        value = _serialize_structured_value(target_value)
                        if not value.strip():
                            raise RuntimeError("Structured output field empty")
                        return parsed, value
                    return call_llm_api(
                        provider=provider,
                        client=provider_client,
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system_prompt=system_prompt,
                        prompt_value=prompt_value,
                        attempt=attempt,
                    )

                try:
                    response = invoke_with_retries(task.max_retries, _request)

                    extra_fields: Optional[Dict[str, str]] = None
                    output_value: str

                    if structured_output_config:
                        parsed_model, output_value = response
                        if task.column_prefix:
                            dumped = parsed_model.model_dump()
                            extra_fields = _flatten_structured_fields(dumped, task.column_prefix)
                    else:
                        output_value = response

                    df.at[task.index, output_column] = output_value
                    if extra_fields:
                        for col_name, value in extra_fields.items():
                            if col_name not in df.columns:
                                df[col_name] = pd.NA
                            df.at[task.index, col_name] = value
                    processed_count += 1
                    df.to_csv(output_csv_path, index=False)

                    if test_first_row:
                        tqdm.write(f"{Fore.CYAN}🧪{Style.RESET_ALL} Test mode: Processed first valid row. Exiting")
                        break
                except Exception as e:
                    tqdm.write(f"{Fore.RED}✗{Style.RESET_ALL} Error processing row {task.index+1}: {e}")
                    df.at[task.index, output_column] = f"ERROR: {e}"
                    df.to_csv(output_csv_path, index=False)
                finally:
                    pbar.update(1)

        print(f"{Fore.GREEN}✓{Style.RESET_ALL} Sequential processing finished")

    processed_rows_final = len(df[df[output_column].notna()]) - (total_rows - total_rows_to_process) # Count non-NA in output col minus initially skipped
    print(f"\n{Fore.GREEN}🎉{Style.RESET_ALL} Completed processing. Total rows with output: {Fore.CYAN}{processed_rows_final}/{total_rows}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}💾{Style.RESET_ALL} Results saved to {Fore.CYAN}{output_csv_path}{Style.RESET_ALL}")


def process_csv_with_llm(*args: Any, **kwargs: Any) -> None:
    """Compatibility-friendly alias for processing CSV rows with any supported LLM provider."""

    process_csv_with_claude(*args, **kwargs)


# --------------------------------------------------------------------------- #
# Embeddings processing                                                       #
# --------------------------------------------------------------------------- #
def process_csv_with_embeddings(
    input_csv_path,
    output_csv_path,
    prompt_template,
    output_column,
    embeddings_provider="OpenAI",
    embeddings_model="text-embedding-3-large",
    test_first_row=False,
    skip_column=None,
    skip_regex=None,
):
    """
    Generate embeddings for each row using `prompt_template` and store them
    (as JSON serialised lists) in `output_column`.

    The implementation is deliberately sequential and cached-agnostic for now.
    """
    import json

    # Read CSV
    try:
        df = pd.read_csv(input_csv_path)
        print(f"{Fore.GREEN}✓{Style.RESET_ALL} Loaded CSV with {Fore.CYAN}{len(df)}{Style.RESET_ALL} rows")
    except Exception as e:
        print(f"{Fore.RED}✗{Style.RESET_ALL} Error loading CSV: {e}")
        return

    required_columns = re.findall(r"\{([^}]+)\}", prompt_template)
    if not required_columns:
        raise ValueError("Prompt template requires at least one {column} placeholder.")

    # Validate columns (named placeholders only)
    positional_cols = [c for c in required_columns if re.fullmatch(r"COL\\d+", c)]
    named_cols = [c for c in required_columns if c not in positional_cols]
    missing = [c for c in named_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {', '.join(missing)}")

    if output_column not in df.columns:
        df[output_column] = pd.NA

    rows_to_process = df[df[output_column].isna()].index
    total = len(rows_to_process)
    if total == 0:
        print(f"{Fore.GREEN}✓{Style.RESET_ALL} No rows require processing")
        return

    # Skip logic (reuse from Claude flow, simplified)
    if skip_column and skip_regex:
        if skip_column not in df.columns:
            raise ValueError(f"Skip column '{skip_column}' not found in CSV.")
        pattern = re.compile(skip_regex)
        keep = []
        skipped = 0
        for idx in rows_to_process:
            val = df.at[idx, skip_column]
            if pd.isna(val) or not pattern.search(str(val)):
                keep.append(idx)
            else:
                df.at[idx, output_column] = ""
                skipped += 1
        rows_to_process = keep
        if skipped:
            print(f"{Fore.YELLOW}⏭️{Style.RESET_ALL} Skipped {skipped} rows via regex")

    # Sequential embedding generation
    for count, idx in enumerate(rows_to_process, start=1):
        row = df.loc[idx]
        row_dict = row.to_dict()

        # Add positional helpers
        for col in positional_cols:
            col_idx = int(col[3:]) - 1
            row_dict[col] = row.iloc[col_idx] if 0 <= col_idx < len(row) else pd.NA

        try:
            prompt_value = prompt_template.format(**row_dict)
        except Exception as e:
            print(f"{Fore.YELLOW}⏭️{Style.RESET_ALL} Row {idx+1} formatting error: {e}")
            continue

        try:
            embedding_vec = get_embedding(
                prompt_value,
                model_provider=embeddings_provider,
                model=embeddings_model,
            )
            df.at[idx, output_column] = json.dumps(embedding_vec)
        except Exception as e:
            print(f"{Fore.RED}✗{Style.RESET_ALL} Error embedding row {idx+1}: {e}")
            df.at[idx, output_column] = f"ERROR: {e}"

        # Early exit for test mode
        if test_first_row:
            print(f"{Fore.CYAN}🧪{Style.RESET_ALL} Test mode: processed first row")
            break

        if count % 10 == 0 or count == len(rows_to_process):
            print(f"{Fore.BLUE}🔄{Style.RESET_ALL} {count}/{len(rows_to_process)} rows embedded")

    df.to_csv(output_csv_path, index=False)
    print(f"{Fore.GREEN}💾{Style.RESET_ALL} Saved embeddings to {Fore.CYAN}{output_csv_path}{Style.RESET_ALL}")
