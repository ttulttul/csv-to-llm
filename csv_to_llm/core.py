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
import types
from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache
from string import Formatter
from typing import Optional, Type, Tuple, List, Callable, Dict, Any, Iterable, get_args, get_origin, Union
from joblib import Memory
from tqdm import tqdm
from colorama import Fore, Style, init
from openai import OpenAI
from perplexity import Perplexity
from pydantic import BaseModel, create_model
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
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
DEFAULT_PERPLEXITY_MODEL = "sonar-pro"
DEFAULT_PERPLEXITY_STRUCTURED_PRESET = "pro-search"
DEFAULT_PROVIDER = PROVIDER_ANTHROPIC
PERPLEXITY_BASE_URL = "https://api.perplexity.ai"

_thread_local_clients = threading.local()
_UNION_TYPES = tuple(
    union_type for union_type in (Union, getattr(types, "UnionType", None))
    if union_type is not None
)


def _openai_web_search_tools(enabled: bool) -> Optional[List[Dict[str, str]]]:
    """Return the OpenAI Responses web search tool list when enabled."""

    if not enabled:
        return None
    return [{"type": "web_search"}]


def _perplexity_web_search_tools(enabled: bool) -> Optional[List[Dict[str, str]]]:
    """Return the Perplexity Responses web search tool list when enabled."""

    if not enabled:
        return None
    return [{"type": "web_search"}, {"type": "fetch_url"}]


def _response_schema_name(raw_name: str) -> str:
    """Build a 1-64 character alphanumeric JSON schema name for provider APIs."""

    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_name) or "StructuredOutput"
    if len(cleaned) <= 64:
        return cleaned
    digest = hashlib.sha1(raw_name.encode("utf-8")).hexdigest()[:12]
    return f"{cleaned[:52]}{digest}"[:64]


def _extract_prompt_fields(prompt_template: str, valid_fields: Optional[set[str]] = None) -> List[str]:
    """Extract format fields while ignoring escaped literal braces."""

    if valid_fields is not None:
        fields = []
        placeholder_tokens = sorted(
            ((f"{{{field}}}", field) for field in valid_fields),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        i = 0
        while i < len(prompt_template):
            if prompt_template.startswith("{{", i) or prompt_template.startswith("}}", i):
                i += 2
                continue

            matched_placeholder = False
            for token, field in placeholder_tokens:
                if prompt_template.startswith(token, i):
                    fields.append(field)
                    i += len(token)
                    matched_placeholder = True
                    break
            if matched_placeholder:
                continue

            if prompt_template[i] == "{":
                end = prompt_template.find("}", i + 1)
                if end != -1:
                    field_name = prompt_template[i + 1:end].split("!", 1)[0].split(":", 1)[0]
                    if field_name:
                        fields.append(field_name)
                    i = end + 1
                    continue
            i += 1
        return fields

    fields = []
    for _, field_name, _, _ in Formatter().parse(prompt_template):
        if field_name is None:
            continue
        field_name = field_name.split("!", 1)[0].split(":", 1)[0]
        if field_name:
            fields.append(field_name)
    return fields


def _render_prompt_template(prompt_template: str, values: Dict[str, Any], required_fields: List[str]) -> str:
    """Render a prompt template using exact CSV column placeholder tokens."""

    placeholder_tokens = sorted(
        ((f"{{{field}}}", field) for field in required_fields),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    result = []
    i = 0
    while i < len(prompt_template):
        if prompt_template.startswith("{{", i):
            result.append("{")
            i += 2
            continue
        if prompt_template.startswith("}}", i):
            result.append("}")
            i += 2
            continue

        matched_placeholder = False
        for token, field in placeholder_tokens:
            if prompt_template.startswith(token, i):
                result.append(_stringify_prompt_value(values[field]))
                i += len(token)
                matched_placeholder = True
                break
        if matched_placeholder:
            continue

        result.append(prompt_template[i])
        i += 1
    return "".join(result)


def _stringify_prompt_value(value: Any) -> str:
    """Convert a CSV cell value to prompt text, treating blank cells as empty."""

    is_missing = False
    try:
        missing_check = pd.isna(value)
        if isinstance(missing_check, bool):
            is_missing = missing_check
    except (TypeError, ValueError):
        is_missing = False

    if is_missing:
        return ""
    return str(value)


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
    provider: str = PROVIDER_OPENAI
    iterate_fields: bool = False
    iterate_parallelism: int = 1
    model_websearch: bool = False


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
    model_websearch: bool = False


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


def _get_perplexity_api_key() -> str:
    """Load and return the Perplexity API key from the environment."""

    load_dotenv()
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise RuntimeError("PERPLEXITY_API_KEY not found in environment variables")
    return api_key


def _get_thread_local_perplexity_client() -> OpenAI:
    """Fetch (or create) a thread-local Perplexity client for reuse."""

    api_key = _get_perplexity_api_key()

    client = getattr(_thread_local_clients, "perplexity_client", None)
    if client is None:
        client = OpenAI(api_key=api_key, base_url=PERPLEXITY_BASE_URL)
        _thread_local_clients.perplexity_client = client
    return client


def _get_thread_local_perplexity_responses_client() -> Perplexity:
    """Fetch (or create) a thread-local Perplexity SDK client for Responses API calls."""

    api_key = _get_perplexity_api_key()

    client = getattr(_thread_local_clients, "perplexity_responses_client", None)
    if client is None:
        client = Perplexity(api_key=api_key)
        _thread_local_clients.perplexity_responses_client = client
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
    provider: str = PROVIDER_OPENAI,
    iterate_fields: bool = False,
    iterate_parallelism: int = 1,
    model_websearch: bool = False,
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
        provider=provider,
        iterate_fields=iterate_fields,
        iterate_parallelism=max(1, int(iterate_parallelism)),
        model_websearch=model_websearch,
    )

    model_cls = _get_pydantic_model_class(config.model_reference, config.class_name)
    if output_field and output_field not in model_cls.model_fields:
        raise ValueError(
            f"Field '{output_field}' not found on Pydantic model '{model_cls.__name__}'"
        )

    return config


def _call_openai_structured_json(
    client: OpenAI,
    prompt_value: str,
    model_reference: str,
    class_name: Optional[str],
    llm_model: str,
    max_output_tokens: int,
    temperature: float,
    system_prompt: str,
    model_websearch: bool,
) -> str:
    """Call OpenAI with Structured Outputs enabled and return JSON."""

    model_cls = _get_pydantic_model_class(model_reference, class_name)

    response = client.responses.parse(
        model=llm_model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_value},
        ],
        text_format=model_cls,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        tools=_openai_web_search_tools(model_websearch),
    )

    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError("Structured output parsing failed: no parsed content returned")

    return parsed.model_dump_json()


@memory.cache(ignore=["client"])
def _call_openai_structured_json_cached(
    client: OpenAI,
    prompt_value: str,
    model_reference: str,
    class_name: Optional[str],
    llm_model: str,
    max_output_tokens: int,
    temperature: float,
    system_prompt: str,
    model_websearch: bool,
) -> str:
    return _call_openai_structured_json(
        client,
        prompt_value,
        model_reference,
        class_name,
        llm_model,
        max_output_tokens,
        temperature,
        system_prompt,
        model_websearch,
    )


def call_openai_structured(prompt_value: str, structured_config: StructuredOutputConfig, openai_client: Optional[OpenAI] = None) -> BaseModel:
    """Call OpenAI with Structured Outputs enabled and return the parsed BaseModel."""

    model_cls = _get_pydantic_model_class(structured_config.model_reference, structured_config.class_name)
    client = openai_client or OpenAI()
    output_json = _call_openai_structured_json_cached(
        client=client,
        prompt_value=prompt_value,
        model_reference=structured_config.model_reference,
        class_name=structured_config.class_name,
        llm_model=structured_config.llm_model,
        max_output_tokens=structured_config.max_output_tokens,
        temperature=structured_config.temperature,
        system_prompt=structured_config.system_prompt,
        model_websearch=structured_config.model_websearch,
    )
    return model_cls.model_validate_json(output_json)


def _perplexity_response_format(model_cls: Type[BaseModel], schema_name: str) -> Dict[str, Any]:
    """Build a Perplexity JSON Schema response format for a Pydantic model."""

    schema = model_cls.model_json_schema()
    schema["required"] = list(model_cls.model_fields.keys())
    schema["additionalProperties"] = False
    return {
        "type": "json_schema",
        "json_schema": {
            "name": _response_schema_name(schema_name),
            "schema": schema,
        },
    }


def _call_perplexity_structured_json(
    client: Perplexity,
    prompt_value: str,
    model_reference: str,
    class_name: Optional[str],
    llm_model: str,
    max_output_tokens: int,
    system_prompt: str,
    model_websearch: bool,
) -> str:
    """Call Perplexity with JSON Schema structured outputs and return JSON."""

    model_cls = _get_pydantic_model_class(model_reference, class_name)
    request_kwargs = {
        "preset": llm_model,
        "input": prompt_value,
        "instructions": system_prompt,
        "max_output_tokens": max_output_tokens,
        "response_format": _perplexity_response_format(model_cls, model_cls.__name__),
    }
    tools = _perplexity_web_search_tools(model_websearch)
    if tools:
        request_kwargs["tools"] = tools
    response = client.responses.create(**request_kwargs)

    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise RuntimeError("Perplexity structured output failed: no output_text returned")
    return model_cls.model_validate_json(output_text).model_dump_json()


@memory.cache(ignore=["client"])
def _call_perplexity_structured_json_cached(
    client: Perplexity,
    prompt_value: str,
    model_reference: str,
    class_name: Optional[str],
    llm_model: str,
    max_output_tokens: int,
    system_prompt: str,
    model_websearch: bool,
) -> str:
    return _call_perplexity_structured_json(
        client,
        prompt_value,
        model_reference,
        class_name,
        llm_model,
        max_output_tokens,
        system_prompt,
        model_websearch,
    )


def call_perplexity_structured(
    prompt_value: str,
    structured_config: StructuredOutputConfig,
    perplexity_client: Optional[Perplexity] = None,
) -> BaseModel:
    """Call Perplexity with JSON Schema structured outputs and return the parsed BaseModel."""

    model_cls = _get_pydantic_model_class(structured_config.model_reference, structured_config.class_name)
    client = perplexity_client or Perplexity(api_key=_get_perplexity_api_key())
    output_json = _call_perplexity_structured_json_cached(
        client=client,
        prompt_value=prompt_value,
        model_reference=structured_config.model_reference,
        class_name=structured_config.class_name,
        llm_model=structured_config.llm_model,
        max_output_tokens=structured_config.max_output_tokens,
        system_prompt=structured_config.system_prompt,
        model_websearch=structured_config.model_websearch,
    )
    return model_cls.model_validate_json(output_json)


def _humanize_identifier(value: str) -> str:
    """Convert Python identifiers into short prompt-friendly words."""

    return value.replace("_", " ")


def _annotation_model_class(annotation: Any) -> Optional[Type[BaseModel]]:
    """Return a nested BaseModel class for direct or optional model annotations."""

    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return annotation

    origin = get_origin(annotation)
    if origin not in _UNION_TYPES:
        return None

    model_args = [
        arg for arg in get_args(annotation)
        if inspect.isclass(arg) and issubclass(arg, BaseModel)
    ]
    non_none_args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if len(model_args) == 1 and len(non_none_args) == 1:
        return model_args[0]
    return None


def _annotation_display_name(annotation: Any) -> str:
    """Return a compact type label for per-field extraction prompts."""

    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _iter_structured_leaf_fields(
    model_cls: Type[BaseModel],
    path: Optional[List[str]] = None,
    owner_models: Optional[List[Type[BaseModel]]] = None,
) -> Iterable[Tuple[List[str], List[Type[BaseModel]], str, Any, Optional[str]]]:
    """Yield leaf fields from a model, recursing into nested BaseModel fields."""

    current_path = path or []
    current_owner_models = owner_models or [model_cls]

    for field_name, field_info in model_cls.model_fields.items():
        field_path = current_path + [field_name]
        nested_model = _annotation_model_class(field_info.annotation)
        if nested_model is not None:
            yield from _iter_structured_leaf_fields(
                nested_model,
                path=field_path,
                owner_models=current_owner_models + [nested_model],
            )
        else:
            yield field_path, current_owner_models, field_name, field_info.annotation, field_info.description


def _set_nested_value(target: Dict[str, Any], path: List[str], value: Any) -> None:
    """Set a nested value inside a dictionary using a field path."""

    cursor = target
    for segment in path[:-1]:
        cursor = cursor.setdefault(segment, {})
    cursor[path[-1]] = value


def _build_iterative_field_prompt(
    base_prompt: str,
    owner_models: List[Type[BaseModel]],
    field_name: str,
    annotation: Any,
    field_description: Optional[str],
) -> str:
    """Build the prompt used when extracting one structured field."""

    owner_names = [model.__name__ for model in owner_models]
    owner_label = " ".join(_humanize_identifier(name) for name in owner_names)
    prompt = (
        f"Respond with the {field_name} (of type {_annotation_display_name(annotation)}) "
        f"of this {owner_label}.\n\nInput:\n{base_prompt}"
    )
    context_lines = []
    for model in owner_models:
        doc = inspect.getdoc(model)
        if doc:
            context_lines.append(f"{model.__name__}: {doc}")
    if field_description:
        context_lines.append(f"{field_name}: {field_description}")
    if context_lines:
        prompt += "\n\nSchema context:\n" + "\n".join(f"- {line}" for line in context_lines)
    return prompt


def _iterative_field_model_name(owner_names: List[str], field_name: str) -> str:
    """Build an OpenAI-compatible temporary model name for one field schema."""

    readable_name = "CsvToLlm" + "".join(part.title().replace("_", "") for part in owner_names + [field_name])
    if len(readable_name) <= 64:
        return readable_name

    digest_source = ".".join(owner_names + [field_name])
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:12]
    suffix = field_name.title().replace("_", "")[:32]
    compact_name = f"CsvToLlm{suffix}{digest}"
    return compact_name[:64]


def _call_openai_structured_field(
    prompt_value: str,
    structured_config: StructuredOutputConfig,
    field_name: str,
    field_annotation: Any,
    owner_models: List[Type[BaseModel]],
    field_description: Optional[str],
    openai_client: OpenAI,
) -> Any:
    """Call OpenAI for a single leaf field using a temporary one-field model."""

    owner_names = [model.__name__ for model in owner_models]
    field_model_name = _iterative_field_model_name(owner_names, field_name)
    field_model = create_model(field_model_name, **{field_name: (field_annotation, ...)})
    field_prompt = _build_iterative_field_prompt(
        base_prompt=prompt_value,
        owner_models=owner_models,
        field_name=field_name,
        annotation=field_annotation,
        field_description=field_description,
    )

    response = openai_client.responses.parse(
        model=structured_config.llm_model,
        input=[
            {"role": "system", "content": structured_config.system_prompt},
            {"role": "user", "content": field_prompt},
        ],
        text_format=field_model,
        temperature=structured_config.temperature,
        max_output_tokens=structured_config.max_output_tokens,
        tools=_openai_web_search_tools(structured_config.model_websearch),
    )

    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError(f"Structured output parsing failed for field '{field_name}'")
    return getattr(parsed, field_name)


def _call_openai_structured_iterative_json(
    client: OpenAI,
    prompt_value: str,
    model_reference: str,
    class_name: Optional[str],
    llm_model: str,
    max_output_tokens: int,
    temperature: float,
    system_prompt: str,
    model_websearch: bool,
    iterate_parallelism: int,
) -> str:
    """Fill a Pydantic model iteratively and return JSON."""

    model_cls = _get_pydantic_model_class(model_reference, class_name)
    structured_config = StructuredOutputConfig(
        model_reference=model_reference,
        class_name=class_name,
        output_field=None,
        llm_model=llm_model,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        provider=PROVIDER_OPENAI,
        iterate_fields=True,
        iterate_parallelism=iterate_parallelism,
        model_websearch=model_websearch,
    )
    payload: Dict[str, Any] = {}
    leaf_fields = list(_iter_structured_leaf_fields(model_cls))

    def fetch_field(field_spec: Tuple[List[str], List[Type[BaseModel]], str, Any, Optional[str]]) -> Tuple[List[str], Any]:
        field_path, owner_models, field_name, field_annotation, field_description = field_spec
        return field_path, _call_openai_structured_field(
            prompt_value=prompt_value,
            structured_config=structured_config,
            field_name=field_name,
            field_annotation=field_annotation,
            owner_models=owner_models,
            field_description=field_description,
            openai_client=client,
        )

    if structured_config.iterate_parallelism <= 1 or len(leaf_fields) <= 1:
        for field_spec in leaf_fields:
            field_path, field_value = fetch_field(field_spec)
            _set_nested_value(payload, field_path, field_value)
    else:
        max_workers = min(structured_config.iterate_parallelism, len(leaf_fields))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_field = {
                executor.submit(fetch_field, field_spec): field_spec
                for field_spec in leaf_fields
            }
            for future in concurrent.futures.as_completed(future_to_field):
                field_path, field_value = future.result()
                _set_nested_value(payload, field_path, field_value)

    return model_cls.model_validate(payload).model_dump_json()


@memory.cache(ignore=["client"])
def _call_openai_structured_iterative_json_cached(
    client: OpenAI,
    prompt_value: str,
    model_reference: str,
    class_name: Optional[str],
    llm_model: str,
    max_output_tokens: int,
    temperature: float,
    system_prompt: str,
    model_websearch: bool,
    iterate_parallelism: int,
) -> str:
    return _call_openai_structured_iterative_json(
        client,
        prompt_value,
        model_reference,
        class_name,
        llm_model,
        max_output_tokens,
        temperature,
        system_prompt,
        model_websearch,
        iterate_parallelism,
    )


def call_openai_structured_iterative(
    prompt_value: str,
    structured_config: StructuredOutputConfig,
    openai_client: Optional[OpenAI] = None,
) -> BaseModel:
    """Fill a Pydantic model by asking OpenAI for one leaf field at a time."""

    model_cls = _get_pydantic_model_class(structured_config.model_reference, structured_config.class_name)
    client = openai_client or OpenAI()
    output_json = _call_openai_structured_iterative_json_cached(
        client=client,
        prompt_value=prompt_value,
        model_reference=structured_config.model_reference,
        class_name=structured_config.class_name,
        llm_model=structured_config.llm_model,
        max_output_tokens=structured_config.max_output_tokens,
        temperature=structured_config.temperature,
        system_prompt=structured_config.system_prompt,
        model_websearch=structured_config.model_websearch,
        iterate_parallelism=structured_config.iterate_parallelism,
    )
    return model_cls.model_validate_json(output_json)


def call_structured_openai(
    prompt_value: str,
    structured_config: StructuredOutputConfig,
    openai_client: Optional[OpenAI] = None,
) -> BaseModel:
    """Call OpenAI structured outputs in normal or per-field iterative mode."""

    if structured_config.iterate_fields:
        return call_openai_structured_iterative(
            prompt_value=prompt_value,
            structured_config=structured_config,
            openai_client=openai_client,
        )
    return call_openai_structured(
        prompt_value=prompt_value,
        structured_config=structured_config,
        openai_client=openai_client,
    )


def call_structured_perplexity(
    prompt_value: str,
    structured_config: StructuredOutputConfig,
    perplexity_client: Optional[Perplexity] = None,
) -> BaseModel:
    """Call Perplexity structured outputs."""

    if structured_config.iterate_fields:
        raise RuntimeError("--pydantic-model-iterate is not supported for provider='perplexity'")
    return call_perplexity_structured(
        prompt_value=prompt_value,
        structured_config=structured_config,
        perplexity_client=perplexity_client,
    )


def call_structured_model(
    prompt_value: str,
    structured_config: StructuredOutputConfig,
    openai_client: Optional[OpenAI] = None,
    perplexity_client: Optional[Perplexity] = None,
) -> BaseModel:
    """Call the configured structured output provider."""

    if structured_config.provider == PROVIDER_OPENAI:
        return call_structured_openai(
            prompt_value=prompt_value,
            structured_config=structured_config,
            openai_client=openai_client,
        )
    if structured_config.provider == PROVIDER_PERPLEXITY:
        return call_structured_perplexity(
            prompt_value=prompt_value,
            structured_config=structured_config,
            perplexity_client=perplexity_client,
        )
    raise RuntimeError(f"Structured outputs are not supported for provider='{structured_config.provider}'")


def _iterative_field_parallelism(parallel: int, total_tasks: int) -> int:
    """Allocate unused row-worker capacity to iterative field extraction."""

    parallel_budget = max(1, int(parallel))
    active_row_workers = max(1, min(parallel_budget, max(1, total_tasks)))
    return max(1, parallel_budget // active_row_workers)


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
        value = value.model_dump(mode="json")
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def _flatten_structured_fields(value: Any, prefix: str) -> Dict[str, str]:
    """Flatten nested structured output values into CSV column names."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")

    if isinstance(value, dict):
        flattened: Dict[str, str] = {}
        for key, nested_value in value.items():
            key_prefix = f"{prefix}{key}"
            if isinstance(nested_value, BaseModel):
                nested_value = nested_value.model_dump(mode="json")
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


def _call_openai_api(client, model, max_tokens, temperature, system_prompt, prompt_value, model_websearch=False):
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
        tools=_openai_web_search_tools(model_websearch),
    )
    response_text = _extract_openai_response_text(response)
    logger.info("Received response from OpenAI. Response: %s", response_text)
    return response_text


def _call_perplexity_api(client, model, max_tokens, temperature, system_prompt, prompt_value, model_websearch=False):
    """Internal helper that performs a Perplexity Sonar chat completion call."""

    logger.info("Sending request to Perplexity model '%s'. Prompt: %s", model, prompt_value)
    if model_websearch:
        response = client.responses.create(
            model=model,
            input=prompt_value,
            instructions=system_prompt,
            max_output_tokens=max_tokens,
            tools=_perplexity_web_search_tools(model_websearch),
        )
        response_text = getattr(response, "output_text", "")
        if response_text:
            logger.info("Received response from Perplexity. Response: %s", response_text)
            return response_text

        warning_msg = "Warning: Unexpected Perplexity Responses API response structure."
        print(warning_msg)
        logger.warning("Unexpected Perplexity Responses API response: %s", response)
        return ""

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
def call_openai_api_cached(client, model, max_tokens, temperature, system_prompt, prompt_value, model_websearch=False):
    return _call_openai_api(client, model, max_tokens, temperature, system_prompt, prompt_value, model_websearch)


def call_openai_api_uncached(client, model, max_tokens, temperature, system_prompt, prompt_value, model_websearch=False):
    return _call_openai_api(client, model, max_tokens, temperature, system_prompt, prompt_value, model_websearch)


@memory.cache(ignore=['client'])
def call_perplexity_api_cached(client, model, max_tokens, temperature, system_prompt, prompt_value, model_websearch=False):
    return _call_perplexity_api(client, model, max_tokens, temperature, system_prompt, prompt_value, model_websearch)


def call_perplexity_api_uncached(client, model, max_tokens, temperature, system_prompt, prompt_value, model_websearch=False):
    return _call_perplexity_api(client, model, max_tokens, temperature, system_prompt, prompt_value, model_websearch)
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
    model_websearch: bool = False,
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

    request_kwargs = {
        "client": client,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system_prompt": system_prompt,
        "prompt_value": prompt_value,
    }
    if provider in (PROVIDER_OPENAI, PROVIDER_PERPLEXITY):
        request_kwargs["model_websearch"] = model_websearch

    return request_fn(**request_kwargs)


def process_single_row(task: RowProcessingArgs):
    """Processes a single row, supporting both Anthropic and OpenAI structured flows."""

    structured_config = task.structured_config
    column_prefix = task.column_prefix

    try:
        if structured_config:
            openai_client = _get_thread_local_openai_client() if structured_config.provider == PROVIDER_OPENAI else None
            perplexity_structured_client = (
                _get_thread_local_perplexity_responses_client()
                if structured_config.provider == PROVIDER_PERPLEXITY
                else None
            )
            provider_client = None
        else:
            openai_client = None
            perplexity_structured_client = None
            provider_client = _get_provider_client(task.provider)
    except RuntimeError as exc:
        return task.index, None, str(exc)

    try:
        format_dict = {col: task.row_data[col] for col in task.required_columns}
        prompt_value = _render_prompt_template(task.prompt_template, format_dict, task.required_columns)
    except KeyError as e:
        return task.index, None, f"Formatting error (likely missing column in template): {e}"
    except TypeError as e:
        return task.index, None, f"Formatting error (likely type issue): {e}"

    try:
        def _request(attempt: int):
            if structured_config:
                parsed = call_structured_model(
                    prompt_value=prompt_value,
                    structured_config=structured_config,
                    openai_client=openai_client,
                    perplexity_client=perplexity_structured_client,
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
                model_websearch=task.model_websearch,
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
    model_websearch: bool,
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
            model_websearch=model_websearch,
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
    pydantic_model_iterate=False,
    model_websearch=False,
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
        pydantic_model_iterate (bool): If True, ask the LLM to fill one leaf Pydantic field at a time and reassemble the full model.
        model_websearch (bool): If True, enable provider web search tools for OpenAI or Perplexity model calls.
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
    if pydantic_model_path and provider not in (PROVIDER_OPENAI, PROVIDER_PERPLEXITY):
        raise ValueError("Structured outputs currently require provider='openai' or provider='perplexity'")
    if model_websearch and provider not in (PROVIDER_OPENAI, PROVIDER_PERPLEXITY):
        raise ValueError("--model-websearch currently requires provider='openai' or provider='perplexity'")
    if pydantic_model_iterate and provider != PROVIDER_OPENAI:
        raise ValueError("--pydantic-model-iterate currently requires provider='openai'")
    
    if model is None:
        if provider == PROVIDER_OPENAI:
            model = DEFAULT_OPENAI_MODEL
        elif provider == PROVIDER_PERPLEXITY:
            model = DEFAULT_PERPLEXITY_STRUCTURED_PRESET if pydantic_model_path else DEFAULT_PERPLEXITY_MODEL
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
    required_columns = _extract_prompt_fields(prompt_template, set(df.columns))
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
            provider=provider,
            iterate_fields=pydantic_model_iterate,
            model_websearch=model_websearch,
        )
        required_key = "OPENAI_API_KEY" if provider == PROVIDER_OPENAI else "PERPLEXITY_API_KEY"
        api_key = os.getenv(required_key)
        if not api_key:
            raise ValueError(f"{required_key} not found in environment variables in main process")
        if parallel == 1:
            if provider == PROVIDER_OPENAI:
                openai_client = OpenAI(api_key=api_key)
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

    if structured_output_config and structured_output_config.iterate_fields:
        structured_output_config = replace(
            structured_output_config,
            iterate_parallelism=_iterative_field_parallelism(parallel, total_tasks),
        )

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
            model_websearch=model_websearch,
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
        perplexity_structured_client = None
        if structured_output_config:
            if structured_output_config.provider == PROVIDER_OPENAI and openai_client is None:
                openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            elif structured_output_config.provider == PROVIDER_PERPLEXITY:
                perplexity_structured_client = Perplexity(api_key=os.getenv("PERPLEXITY_API_KEY"))
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
            model_websearch=model_websearch,
        )

        with tqdm(total=total_tasks, desc="Processing rows", unit="row", 
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            for task in task_iterator:
                try:
                    format_dict = {col: task.row_data[col] for col in task.required_columns}
                    prompt_value = _render_prompt_template(task.prompt_template, format_dict, task.required_columns)
                except KeyError as e:
                    tqdm.write(f"{Fore.YELLOW}⏭️{Style.RESET_ALL} Skipping row {task.index+1} due to formatting error: {e}")
                    pbar.update(1)
                    continue

                pbar.set_postfix_str(f"Row {task.index+1}")

                def _request(attempt: int):
                    if structured_output_config:
                        parsed = call_structured_model(
                            prompt_value=prompt_value,
                            structured_config=structured_output_config,
                            openai_client=openai_client,
                            perplexity_client=perplexity_structured_client,
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
                        model_websearch=model_websearch,
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

    required_columns = _extract_prompt_fields(prompt_template, set(df.columns))
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
            prompt_value = _render_prompt_template(prompt_template, row_dict, required_columns)
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
