import argparse
import logging
from .core import process_csv_with_claude


def main():
    """Main CLI entry point for csv-to-llm."""
    parser = argparse.ArgumentParser(description="csv-to-llm: Process CSV with Claude API using a prompt template.")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--prompt-template", help="Prompt template string (e.g., 'Summarize: {text}')")
    parser.add_argument("--prompt-template-file", help="Path to a text file containing the prompt template. Overrides --prompt-template if provided.")
    parser.add_argument("--output-col", required=True, help="Column to store responses")
    parser.add_argument("--system", default="You are a helpful assistant.",
                        help="System prompt for Claude")
    parser.add_argument("--model", default="claude-3-7-sonnet-20250219", help="Claude model to use")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Maximum tokens for response")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature setting")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging (INFO level)")
    parser.add_argument("--test-first-row", action="store_true", help="Only process the first valid row for testing")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel processes to use (default: 1 for sequential)")
    parser.add_argument("--skip-rows", nargs=2, metavar=('COLUMN', 'REGEX'), help="Column name and regex pattern. Skip processing rows where the column value matches the regex, setting output to empty string.")

    args = parser.parse_args()

    # --- Logging Configuration ---
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")
    # --- End Logging Configuration ---

    skip_col_arg = None
    skip_regex_arg = None
    if args.skip_rows:
        skip_col_arg = args.skip_rows[0]
        skip_regex_arg = args.skip_rows[1]

    # Determine the prompt template value; file overrides direct string
    if args.prompt_template_file:
        try:
            with open(args.prompt_template_file, "r", encoding="utf-8") as f:
                prompt_template_value = f.read()
        except Exception as e:
            raise ValueError(f"Failed to read prompt template file '{args.prompt_template_file}': {e}")
    else:
        prompt_template_value = args.prompt_template

    if prompt_template_value is None:
        raise ValueError("You must specify either --prompt-template or --prompt-template-file.")

    process_csv_with_claude(
        input_csv_path=args.input,
        output_csv_path=args.output,
        prompt_template=prompt_template_value,
        output_column=args.output_col,
        system_prompt=args.system,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        test_first_row=args.test_first_row,
        parallel=args.parallel,
        skip_column=skip_col_arg,
        skip_regex=skip_regex_arg
    )


if __name__ == "__main__":
    main()