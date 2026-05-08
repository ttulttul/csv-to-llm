# Google Sheets Apps Script

This folder contains a standalone Google Apps Script implementation of the
spreadsheet-facing Perplexity workflow.

## Files

- `Code.gs`: Custom functions and helper functions.
- `appsscript.json`: Apps Script manifest with the required scopes.

## Setup

1. Open a Google Sheet.
2. Go to **Extensions > Apps Script**.
3. Copy `Code.gs` into the Apps Script editor.
4. Copy `appsscript.json` into the manifest file. If the manifest is hidden,
   enable **Project Settings > Show "appsscript.json" manifest file in editor**.
5. Run `promptForPerplexityApiKey` once from the Apps Script editor, or reload
   the spreadsheet and use **csv-to-llm > Set Perplexity API key**.
6. Optionally run `promptForPerplexityModel` to override the default Agent API
   model. The default is `openai/gpt-5.4`.

You can also run `promptForPerplexityPreset` if you prefer using a Perplexity
preset such as `pro-search` instead of a model name.

## Custom Functions

Plain prompt:

```text
=PERPLEXITY("What is this company's homepage URL? The company name is " & A2, TRUE)
```

The second argument enables Perplexity `web_search` and `fetch_url` tools.

Auto mode:

```text
=PERPLEXITY_AUTO(
  A1:F1,
  A2:F6,
  A33:F33,
  "Categorize this company into one of the following categories: web hosting provider, telecom company, other"
)
```

`PERPLEXITY_AUTO` returns a single result string. Use `PERPLEXITY_AUTO_JSON`
with the same arguments to return the full JSON payload:

```json
{
  "result": "web hosting provider",
  "confidence": "high",
  "rationale": "The target row describes hosting infrastructure services."
}
```

Both functions use `CacheService` for six-hour request caching keyed by the
full Perplexity payload. This reduces duplicate paid API calls when Google
Sheets recalculates unchanged formulas.

## Notes

- Google Sheets custom functions must finish quickly. For large batches, prefer
  filling a small number of cells at a time or use the Python CLI in this repo.
- Custom functions cannot write outside their own result cell or spill range.
- The first structured-output call for a new JSON schema can be slower while
  Perplexity prepares the schema.
- Perplexity tool use can add separate `web_search` and `fetch_url` charges.
