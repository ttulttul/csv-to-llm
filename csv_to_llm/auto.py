import json
import os
import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

from .core import DEFAULT_OPENAI_MODEL


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


def run_auto_mode(
    instruction: str,
    input_csv_path: str,
    sample_size: int,
    model: Optional[str] = None,
    temperature: float = 0,
    output_column: Optional[str] = None,
    openai_client: Optional[OpenAI] = None,
) -> AutoPlan:
    """Generate a Pydantic model + prompt using a single instruction."""

    df = pd.read_csv(input_csv_path)
    if df.empty:
        raise ValueError("Input CSV is empty; cannot run --auto")

    model_name = model or DEFAULT_OPENAI_MODEL
    client = openai_client or OpenAI()

    sample_size = max(1, min(sample_size, len(df)))
    sample_df = df.head(sample_size)
    sample_payload = sample_df.to_dict(orient="records")

    columns_meta = [
        {"name": col, "dtype": str(df[col].dtype)}
        for col in df.columns
    ]

    task_payload = {
        "user_instruction": instruction,
        "columns": columns_meta,
        "sample_rows": sample_payload,
    }

    response = client.responses.parse(
        model=model_name,
        temperature=temperature,
        input=[
            {
                "role": "system",
                "content": (
                    "You design Pydantic BaseModel schemas for CSV data. "
                    "Given column metadata and sample rows, produce Python code for a Pydantic BaseModel. "
                    "Also provide a prompt template that references CSV columns via {column_name}."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Task description: {task}\n\nColumns:\n{cols}\n\nSample rows:\n{rows}".format(
                                task=instruction,
                                cols=json.dumps(columns_meta, indent=2),
                                rows=json.dumps(sample_payload, indent=2),
                            )
                        ),
                    }
                ],
            },
        ],
        text_format=AutoModelDesign,
    )

    design = response.output_parsed
    if design is None:
        raise RuntimeError("Auto design response parsing failed")

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

    prompt_template = design.prompt_template.strip()
    placeholders = re.findall(r"\{([^}]+)\}", prompt_template)
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
