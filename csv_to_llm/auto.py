import json
import os
import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from openai import OpenAI
from perplexity import Perplexity
from pydantic import BaseModel, Field

from .core import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_PERPLEXITY_STRUCTURED_PRESET,
    PROVIDER_OPENAI,
    PROVIDER_PERPLEXITY,
    _extract_prompt_fields,
    _get_perplexity_api_key,
    _openai_web_search_tools,
    _perplexity_response_format,
    _perplexity_web_search_tools,
)


class AutoModelDesign(BaseModel):
    """Schema returned by the LLM when auto-generating models."""

    model_name: str = Field(description="Name of the Pydantic BaseModel class")
    python_code: str = Field(description="Python code defining the BaseModel, including imports")
    primary_field: Optional[str] = Field(
        description="Field whose value should populate the CLI output column; nullable if none",
        default=None,
    )
    prompt_template: str = Field(
        description="Prompt template string that references CSV columns via {column_name}",
    )
    output_column_name: Optional[str] = Field(
        description="Suggested name for the csv output column",
        default=None,
    )
    reasoning: Optional[str] = Field(
        description="Brief rationale for the chosen schema",
        default=None,
    )


@dataclass
class AutoPlan:
    prompt_template: str
    pydantic_model_path: str
    pydantic_model_class: str
    primary_field: Optional[str]
    output_column: str


def _sanitize_identifier(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", value).strip("_")
    return cleaned or fallback


def _ensure_python_file(code: str, class_name: str, directory: str) -> str:
    os.makedirs(directory, exist_ok=True)
    module_name = _sanitize_identifier(class_name.lower(), "auto_model")
    file_path = os.path.join(directory, f"{module_name}.py")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code.strip())
        if not code.endswith("\n"):
            f.write("\n")
    return file_path


def _auto_design_system_prompt() -> str:
    return (
        "You design Pydantic BaseModel schemas for CSV data. "
        "Given column metadata and sample rows, produce Python code for a Pydantic BaseModel. "
        "Also provide a prompt template that references CSV columns via {column_name}."
    )


def _auto_design_user_text(instruction: str, columns_meta: list[dict], sample_payload: list[dict]) -> str:
    return "Task description: {task}\n\nColumns:\n{cols}\n\nSample rows:\n{rows}".format(
        task=instruction,
        cols=json.dumps(columns_meta, indent=2),
        rows=json.dumps(sample_payload, indent=2),
    )


def _run_openai_auto_design(
    instruction: str,
    columns_meta: list[dict],
    sample_payload: list[dict],
    model_name: str,
    temperature: float,
    model_websearch: bool,
    openai_client: Optional[OpenAI],
) -> AutoModelDesign:
    client = openai_client or OpenAI()
    response = client.responses.parse(
        model=model_name,
        temperature=temperature,
        input=[
            {
                "role": "system",
                "content": _auto_design_system_prompt(),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": _auto_design_user_text(instruction, columns_meta, sample_payload),
                    }
                ],
            },
        ],
        text_format=AutoModelDesign,
        tools=_openai_web_search_tools(model_websearch),
    )

    design = response.output_parsed
    if design is None:
        raise RuntimeError("Auto design response parsing failed")
    return design


def _run_perplexity_auto_design(
    instruction: str,
    columns_meta: list[dict],
    sample_payload: list[dict],
    model_name: str,
    model_websearch: bool,
    perplexity_client: Optional[Perplexity],
) -> AutoModelDesign:
    client = perplexity_client or Perplexity(api_key=_get_perplexity_api_key())
    request_kwargs = {
        "preset": model_name,
        "input": _auto_design_user_text(instruction, columns_meta, sample_payload),
        "instructions": _auto_design_system_prompt(),
        "response_format": _perplexity_response_format(AutoModelDesign, AutoModelDesign.__name__),
    }
    tools = _perplexity_web_search_tools(model_websearch)
    if tools:
        request_kwargs["tools"] = tools
    response = client.responses.create(**request_kwargs)

    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise RuntimeError("Perplexity auto design failed: no output_text returned")
    return AutoModelDesign.model_validate_json(output_text)


def _escape_non_column_braces(prompt_template: str, valid_fields: set[str]) -> str:
    """Escape generated literal braces while preserving CSV column placeholders."""

    result = []
    i = 0
    while i < len(prompt_template):
        char = prompt_template[i]
        if char == "{":
            if i + 1 < len(prompt_template) and prompt_template[i + 1] == "{":
                result.append("{{")
                i += 2
                continue
            end = prompt_template.find("}", i + 1)
            if end == -1:
                result.append("{{")
                i += 1
                continue
            field_name = prompt_template[i + 1:end]
            if field_name in valid_fields:
                result.append(prompt_template[i:end + 1])
            else:
                result.append("{{")
                result.append(field_name)
                result.append("}}")
            i = end + 1
            continue
        if char == "}":
            if i + 1 < len(prompt_template) and prompt_template[i + 1] == "}":
                result.append("}}")
                i += 2
                continue
            result.append("}}")
            i += 1
            continue
        result.append(char)
        i += 1
    return "".join(result)


def run_auto_mode(
    instruction: str,
    input_csv_path: str,
    sample_size: int,
    provider: str = PROVIDER_OPENAI,
    model: Optional[str] = None,
    temperature: float = 0,
    model_websearch: bool = False,
    output_column: Optional[str] = None,
    openai_client: Optional[OpenAI] = None,
    perplexity_client: Optional[Perplexity] = None,
) -> AutoPlan:
    """Generate a Pydantic model + prompt using a single instruction."""

    df = pd.read_csv(input_csv_path)
    if df.empty:
        raise ValueError("Input CSV is empty; cannot run --auto")

    sample_size = max(1, min(sample_size, len(df)))
    sample_df = df.head(sample_size)
    sample_payload = sample_df.to_dict(orient="records")

    columns_meta = [
        {"name": col, "dtype": str(df[col].dtype)}
        for col in df.columns
    ]

    if provider == PROVIDER_OPENAI:
        model_name = model or DEFAULT_OPENAI_MODEL
        design = _run_openai_auto_design(
            instruction=instruction,
            columns_meta=columns_meta,
            sample_payload=sample_payload,
            model_name=model_name,
            temperature=temperature,
            model_websearch=model_websearch,
            openai_client=openai_client,
        )
    elif provider == PROVIDER_PERPLEXITY:
        model_name = model or DEFAULT_PERPLEXITY_STRUCTURED_PRESET
        design = _run_perplexity_auto_design(
            instruction=instruction,
            columns_meta=columns_meta,
            sample_payload=sample_payload,
            model_name=model_name,
            model_websearch=model_websearch,
            perplexity_client=perplexity_client,
        )
    else:
        raise ValueError("--auto currently requires --provider openai or --provider perplexity")

    class_name = design.model_name or "AutoSchema"
    python_code = design.python_code.strip()
    if "BaseModel" not in python_code:
        python_code = (
            "from pydantic import BaseModel\n\n"
            f"class {class_name}(BaseModel):\n    summary: str\n"
        )

    auto_dir = os.path.join(os.path.dirname(os.path.abspath(input_csv_path)), ".csv_to_llm_auto")
    model_path = _ensure_python_file(python_code, class_name, auto_dir)

    suggested_output = output_column or design.output_column_name or _sanitize_identifier(
        (design.primary_field or "auto_output"), "auto_output"
    )

    prompt_template = _escape_non_column_braces(design.prompt_template.strip(), set(df.columns))
    placeholders = _extract_prompt_fields(prompt_template)
    missing = [ph for ph in placeholders if ph not in df.columns]
    if missing:
        raise ValueError(
            "Auto-generated prompt references unknown columns: " + ", ".join(missing)
        )

    return AutoPlan(
        prompt_template=prompt_template,
        pydantic_model_path=model_path,
        pydantic_model_class=class_name,
        primary_field=design.primary_field,
        output_column=suggested_output,
    )
