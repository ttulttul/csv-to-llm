import os
import pandas as pd
import anthropic
import time
from dotenv import load_dotenv
import concurrent.futures
import re
import logging
from joblib import Memory
from tqdm import tqdm
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Module-level logger
logger = logging.getLogger(__name__)

# --- Joblib Cache Setup ---
# Define cache directory (you might want to make this configurable or use a temporary dir)
cachedir = './claude_cache' 
memory = Memory(cachedir, verbose=0)
# --- End Cache Setup ---

# --- Cached API Call ---
@memory.cache(ignore=['client']) # Ignore the client object for caching purposes
def call_claude_api_cached(client, model, max_tokens, temperature, system_prompt, prompt_value):
    """
    Wrapper function to call the Claude API, designed for caching with joblib.
    The 'client' object is passed but ignored by the cache.
    """
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
    # Extract response content safely
    if message.content and hasattr(message.content[0], 'text'):
        response_text = message.content[0].text
        logger.info("Received response from Claude. Response: %s", response_text)
        return response_text
    else:
        # Log or handle the unexpected response structure
        warning_msg = f"Warning: Unexpected API response structure. Content: {message.content}"
        print(warning_msg)
        logger.warning("Unexpected API response structure. Content: %s", message.content)
        return ""
# --- End Cached API Call ---


# Worker function for parallel processing
def process_single_row(args_tuple):
    """Processes a single row by calling the Claude API via a cached wrapper."""
    index, row_data, required_columns, prompt_template, model, max_tokens, temperature, system_prompt, output_column = args_tuple

    # Load environment variables and initialize client within the worker process
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        # Cannot raise exception easily across processes, return error status
        return index, None, "ANTHROPIC_API_KEY not found in environment variables"
    
    client = anthropic.Anthropic(api_key=api_key)

    # Prepare the data for formatting the prompt template
    # Convert row_data (dict) back to Series temporarily for easier NA checks if needed, or handle dict directly
    format_dict = {col: row_data.get(col) for col in required_columns} 

    # Check if any required values for the template are missing in this row
    # Ensure robust checking for various forms of missing data (None, NaN)
    if any(pd.isna(format_dict.get(col)) for col in required_columns):
         return index, None, "Missing data for prompt template"

    # Format the prompt using the template and row data
    try:
        prompt_value = prompt_template.format(**format_dict)
    except KeyError as e:
        return index, None, f"Formatting error (likely missing column in template): {e}"
    except TypeError as e: # Handle potential type errors during formatting if data isn't string
        return index, None, f"Formatting error (likely type issue): {e}"


    try:
        # Call Claude API via the cached wrapper
        response = call_claude_api_cached(
            client=client, # Pass client, but it's ignored by cache
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            prompt_value=prompt_value
        )
        return index, response, None # index, result, error=None

    except Exception as e:
        # Return error without verbose logging
        return index, None, f"API Error: {e}" # index, result=None, error


def process_csv_with_claude(
    input_csv_path,
    output_csv_path,
    prompt_template,
    output_column,
    system_prompt="You are a world-class poet. Respond only with short poems.",
    model="claude-3-7-sonnet-20250219",
    max_tokens=1000,
    temperature=1,
    test_first_row=False,
    parallel=1,
    skip_column=None,
    skip_regex=None
):
    """
    Process a CSV file by sending values from one column to Claude API and saving responses to another column.
    
    Args:
        input_csv_path (str): Path to the input CSV file
        output_csv_path (str): Path to save the output CSV file
        prompt_template (str): A template string for the prompt, potentially containing column names in curly braces (e.g., "Summarize: {text_column}")
        output_column (str): Name of the column to store Claude's responses
        system_prompt (str): System prompt for Claude API
        model (str): Claude model to use
        max_tokens (int): Maximum tokens for Claude's response
        temperature (float): Temperature setting for response randomness
        test_first_row (bool): If True, only process the first valid row and exit.
        parallel (int): Number of parallel processes to use.
        skip_column (str, optional): Column name to check for skipping rows. Defaults to None.
        skip_regex (str, optional): Regex pattern to match in skip_column for skipping. Defaults to None.
    """
    # Load environment variables from .env file (needed for main process checks)
    load_dotenv()
    
    # Initialize the Anthropic client with API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables in main process")
    
    # Client is initialized here for sequential mode or pre-checks
    # For parallel mode, each worker initializes its own client
    client = None
    if parallel == 1:
         client = anthropic.Anthropic(api_key=api_key)

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
    tasks = []
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

    # Prepare tasks for processing
    for index in rows_to_process_indices:
        row = df.loc[index]

        # Build an augmented dict that includes positional placeholders (COL\d+)
        row_dict = row.to_dict()
        for col in positional_cols:
            col_idx = int(col[3:]) - 1  # COL1 → index 0
            if 0 <= col_idx < len(row):
                row_dict[col] = row.iloc[col_idx]
            else:
                row_dict[col] = pd.NA  # Out-of-range index

        # Prepare data tuple for the worker function
        task_args = (
            index, row_dict, required_columns, prompt_template, model,
            max_tokens, temperature, system_prompt, output_column
        )
        tasks.append(task_args)
        
        # Handle test_first_row: only prepare the first valid task
        if test_first_row:
             print(f"{Fore.CYAN}🧪{Style.RESET_ALL} Test mode: Preparing only the first valid row for processing")
             break # Only add the first task

    if not tasks:
        print(f"{Fore.GREEN}✓{Style.RESET_ALL} No rows require processing")
        return

    if parallel > 1:
        print(f"{Fore.BLUE}⚡{Style.RESET_ALL} Starting parallel processing with {Fore.CYAN}{parallel}{Style.RESET_ALL} workers")
        results = {} # Store results keyed by index
        with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as executor:
            # Submit tasks
            future_to_index = {executor.submit(process_single_row, task): task[0] for task in tasks}

            # Use tqdm for progress tracking
            with tqdm(total=len(tasks), desc="Processing rows", unit="row", 
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        idx, response, error = future.result()
                        if error:
                            tqdm.write(f"{Fore.RED}✗{Style.RESET_ALL} Error processing row {idx + 1}: {error}")
                            df.at[idx, output_column] = f"ERROR: {error}" 
                        else:
                            df.at[idx, output_column] = response
                            processed_count += 1
                    except Exception as exc:
                        tqdm.write(f"{Fore.RED}✗{Style.RESET_ALL} Row {index + 1} generated an exception: {exc}")
                        df.at[index, output_column] = f"ERROR: Future execution failed - {exc}"
                    
                    pbar.update(1)

        # Save the entire DataFrame once after all parallel tasks are done
        print(f"{Fore.GREEN}💾{Style.RESET_ALL} Parallel processing finished. Saving results...")
        df.to_csv(output_csv_path, index=False)

    else: # Sequential processing (parallel == 1)
        print(f"{Fore.BLUE}🔄{Style.RESET_ALL} Starting sequential processing")
        if client is None: # Should have been initialized earlier if parallel==1
             client = anthropic.Anthropic(api_key=api_key)

        # Use tqdm for progress tracking
        with tqdm(tasks, desc="Processing rows", unit="row", 
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            for task_args in pbar:
                index, row_data_dict, _, _, _, _, _, _, _ = task_args  # Unpack needed args

                # Prepare the data for formatting the prompt template
                format_dict = {col: row_data_dict.get(col) for col in required_columns}

                # Check if any required values for the template are missing in this row
                if any(pd.isna(val) for val in format_dict.values()):
                     tqdm.write(f"{Fore.YELLOW}⏭️{Style.RESET_ALL} Skipping row {index+1} (missing data for prompt template)")
                     continue

                # Format the prompt using the template and row data
                try:
                    prompt_value = prompt_template.format(**format_dict)
                except KeyError as e:
                    tqdm.write(f"{Fore.YELLOW}⏭️{Style.RESET_ALL} Skipping row {index+1} due to formatting error: {e}")
                    continue

                # Update progress bar description with current row
                pbar.set_postfix_str(f"Row {index+1}")

                try:
                    # Call Claude API via the cached wrapper (using the single client)
                    response = call_claude_api_cached(
                        client=client, # Pass client, but it's ignored by cache
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system_prompt=system_prompt,
                        prompt_value=prompt_value
                    )
                    
                    # Save response to the dataframe
                    df.at[index, output_column] = response
                    processed_count += 1
                    
                    # Save progress after each successful API call (only in sequential mode)
                    df.to_csv(output_csv_path, index=False)

                    # If testing the first row, break after successful processing (already handled by task list size)
                    if test_first_row:
                        tqdm.write(f"{Fore.CYAN}🧪{Style.RESET_ALL} Test mode: Processed first valid row. Exiting")
                        break # Exit loop after first task

                except Exception as e:
                    tqdm.write(f"{Fore.RED}✗{Style.RESET_ALL} Error processing row {index+1}: {e}")
                    df.at[index, output_column] = f"ERROR: {e}"
                    # Save progress even if there was an error (only in sequential mode)
                    df.to_csv(output_csv_path, index=False)

        print(f"{Fore.GREEN}✓{Style.RESET_ALL} Sequential processing finished")

    processed_rows_final = len(df[df[output_column].notna()]) - (total_rows - total_rows_to_process) # Count non-NA in output col minus initially skipped
    print(f"\n{Fore.GREEN}🎉{Style.RESET_ALL} Completed processing. Total rows with output: {Fore.CYAN}{processed_rows_final}/{total_rows}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}💾{Style.RESET_ALL} Results saved to {Fore.CYAN}{output_csv_path}{Style.RESET_ALL}")
