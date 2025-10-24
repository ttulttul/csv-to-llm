import argparse
import logging
from colorama import Fore, Style, init
from .core import (
    process_csv_with_claude,
    process_csv_with_embeddings,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OPENAI_MODEL,
)
from .auto import run_auto_mode
from .embeddings import list_available_embedding_models

# Initialize colorama for cross-platform colored output
init(autoreset=True)


def main():
    """Main CLI entry point for csv-to-llm."""
    parser = argparse.ArgumentParser(
        description=f"{Fore.BLUE}csv-to-llm{Style.RESET_ALL}: Process CSV with Claude API using a prompt template.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--prompt-template", help="Prompt template string (e.g., 'Summarize: {text}')")
    parser.add_argument("--prompt-template-file", help="Path to a text file containing the prompt template. Overrides --prompt-template if provided.")
    parser.add_argument("--output-col", help="Column to store responses (required unless --auto provides a default)")
    parser.add_argument("--system", default="You are a helpful assistant.",
                        help="System prompt for Claude")
    parser.add_argument("--model", help="Model to use (defaults differ for Anthropic vs OpenAI modes)")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Maximum tokens for response")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature setting")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging (INFO level)")
    parser.add_argument("--test-first-row", action="store_true", help="Only process the first valid row for testing")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel processes to use (default: 1 for sequential)")
    parser.add_argument("--skip-rows", nargs=2, metavar=('COLUMN', 'REGEX'), help="Column name and regex pattern. Skip processing rows where the column value matches the regex, setting output to empty string.")
    parser.add_argument("--max-retries", type=int, default=2, help="Number of times to retry a failed/empty LLM response (default: 2 retries)")
    parser.add_argument("--pydantic-model", help="Path or module reference to a Pydantic BaseModel for structured outputs")
    parser.add_argument("--pydantic-model-class", help="Class name of the BaseModel to use when multiple models are defined")
    parser.add_argument("--pydantic-model-field", help="Field on the Pydantic model to copy into --output-col")
    parser.add_argument("--pydantic-model-column-prefix", help="Prefix for extra columns populated from every Pydantic field (mutually exclusive with --pydantic-model-field)")

    # --- Embedding specific options ---
    provider_choices = list(list_available_embedding_models().keys())
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Generate embeddings (instead of Claude responses) based on the prompt template",
    )
    parser.add_argument(
        "--embeddings-provider",
        default="OpenAI",
        choices=provider_choices,
        help="Embedding model provider",
    )
    parser.add_argument(
        "--embeddings-model",
        default="text-embedding-3-large",
        help="Embedding model name (see list_available_embedding_models())",
    )
    parser.add_argument(
        "--auto",
        help="Automatically synthesize a Pydantic model, prompt template, and output column from a single instruction",
    )
    parser.add_argument(
        "--auto-sample-size",
        type=int,
        default=5,
        help="Number of sample rows to include when designing the auto schema (default: 5)",
    )

    args = parser.parse_args()
    auto_instruction = getattr(args, "auto", None)
    auto_sample_size = getattr(args, "auto_sample_size", 5)

    # --- Logging Configuration ---
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")
    # --- End Logging Configuration ---
    
    # Welcome message
    print(f"{Fore.BLUE}🚀{Style.RESET_ALL} Starting {Fore.CYAN}csv-to-llm{Style.RESET_ALL} processing...")

    skip_col_arg = None
    skip_regex_arg = None
    if args.skip_rows:
        skip_col_arg = args.skip_rows[0]
        skip_regex_arg = args.skip_rows[1]

    prompt_template_value = None
    auto_plan = None

    if auto_instruction:
        if auto_sample_size <= 0:
            parser.error("--auto-sample-size must be positive")
        if args.embeddings:
            parser.error("--auto cannot be combined with --embeddings")
        if args.prompt_template or args.prompt_template_file:
            parser.error("--auto cannot be combined with prompt template arguments")
        if args.pydantic_model or args.pydantic_model_class or args.pydantic_model_field or args.pydantic_model_column_prefix:
            parser.error("--auto cannot be combined with manual Pydantic arguments")

        auto_plan = run_auto_mode(
            instruction=auto_instruction,
            input_csv_path=args.input,
            sample_size=auto_sample_size,
            model=args.model,
            temperature=args.temperature,
            output_column=args.output_col,
        )
        prompt_template_value = auto_plan.prompt_template
        args.pydantic_model = auto_plan.pydantic_model_path
        args.pydantic_model_class = auto_plan.pydantic_model_class
        args.pydantic_model_field = auto_plan.primary_field
        if auto_plan.primary_field is None and not args.pydantic_model_column_prefix:
            args.pydantic_model_column_prefix = "auto_"
        args.output_col = args.output_col or auto_plan.output_column
        print(
            f"{Fore.GREEN}✓{Style.RESET_ALL} Auto mode synthesized model '{auto_plan.pydantic_model_class}' "
            f"and prompt for column '{args.output_col}'"
        )

    # Determine the prompt template value; file overrides direct string unless auto provided
    if prompt_template_value is None:
        if args.prompt_template_file:
            try:
                with open(args.prompt_template_file, "r", encoding="utf-8") as f:
                    prompt_template_value = f.read()
            except Exception as e:
                print(f"{Fore.RED}✗{Style.RESET_ALL} Failed to read prompt template file '{args.prompt_template_file}': {e}")
                raise ValueError(f"Failed to read prompt template file '{args.prompt_template_file}': {e}")
        else:
            prompt_template_value = args.prompt_template

    if prompt_template_value is None:
        print(f"{Fore.RED}✗{Style.RESET_ALL} You must specify either --prompt-template, --prompt-template-file, or use --auto")
        raise ValueError("You must specify either --prompt-template, --prompt-template-file, or use --auto.")

    if not args.output_col:
        parser.error("--output-col is required (auto mode can provide one automatically)")

    if args.pydantic_model and args.embeddings:
        parser.error("--pydantic-model cannot be used together with --embeddings")

    if args.pydantic_model_column_prefix and not args.pydantic_model:
        parser.error("--pydantic-model-column-prefix requires --pydantic-model")

    if args.pydantic_model and args.pydantic_model_column_prefix and args.pydantic_model_field:
        parser.error("--pydantic-model-field cannot be combined with --pydantic-model-column-prefix")

    if args.pydantic_model and not args.pydantic_model_column_prefix and not args.pydantic_model_field:
        parser.error("--pydantic-model-field is required unless --pydantic-model-column-prefix is provided")

    if args.max_retries < 0:
        parser.error("--max-retries must be zero or a positive integer")

    resolved_model = args.model
    if args.pydantic_model:
        resolved_model = resolved_model or DEFAULT_OPENAI_MODEL
    else:
        resolved_model = resolved_model or DEFAULT_ANTHROPIC_MODEL

    if args.embeddings:
        process_csv_with_embeddings(
            input_csv_path=args.input,
            output_csv_path=args.output,
            prompt_template=prompt_template_value,
            output_column=args.output_col,
            embeddings_provider=args.embeddings_provider,
            embeddings_model=args.embeddings_model,
            test_first_row=args.test_first_row,
            skip_column=skip_col_arg,
            skip_regex=skip_regex_arg,
        )
    else:
        process_csv_with_claude(
            input_csv_path=args.input,
            output_csv_path=args.output,
            prompt_template=prompt_template_value,
            output_column=args.output_col,
            system_prompt=args.system,
            model=resolved_model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            test_first_row=args.test_first_row,
            parallel=args.parallel,
            skip_column=skip_col_arg,
            skip_regex=skip_regex_arg,
            pydantic_model_path=args.pydantic_model,
            pydantic_model_class=args.pydantic_model_class,
            pydantic_model_field=args.pydantic_model_field,
            pydantic_model_column_prefix=args.pydantic_model_column_prefix,
            max_retries=args.max_retries,
        )


if __name__ == "__main__":
    main()
