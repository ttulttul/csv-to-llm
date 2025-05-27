import argparse
import logging
from colorama import Fore, Style, init
from .core import process_csv_with_claude, process_csv_with_embeddings
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

    args = parser.parse_args()

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

    # Determine the prompt template value; file overrides direct string
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
        print(f"{Fore.RED}✗{Style.RESET_ALL} You must specify either --prompt-template or --prompt-template-file")
        raise ValueError("You must specify either --prompt-template or --prompt-template-file.")

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
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            test_first_row=args.test_first_row,
            parallel=args.parallel,
            skip_column=skip_col_arg,
            skip_regex=skip_regex_arg,
        )


if __name__ == "__main__":
    main()
