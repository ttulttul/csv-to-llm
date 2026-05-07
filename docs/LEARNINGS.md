# Learnings

- 2026-05-05: Current provider interfaces differ by vendor. OpenAI's recommended text generation path is the Responses API, Anthropic uses the Messages API, and Perplexity Sonar supports OpenAI-compatible chat completions at `https://api.perplexity.ai`.
- 2026-05-05: Structured outputs remain OpenAI-specific in this project because the existing implementation uses `client.responses.parse(...)` with Pydantic models.
- 2026-05-05: Client construction should happen after CSV and argument validation so local input errors do not require valid SDK transport setup or live API credentials.
- 2026-05-05: The old root-level `my_model.py` and `my_model_broad.py` files were example Pydantic schemas for email structured extraction. They now live under `examples/` with README usage.
- 2026-05-05: Users expect `--pydantic-model-column-prefix` to create scalar columns for nested structured output fields, not JSON blobs for each top-level nested object.
- 2026-05-05: Large nested Pydantic schemas can be more reliable when filled one leaf field at a time, then reassembled and validated against the original model.
- 2026-05-05: Iterative field extraction should consume spare `--parallel` capacity without multiplying row workers by field workers into unbounded API concurrency.
- 2026-05-05: OpenAI Responses web search is opt-in via `tools=[{"type": "web_search"}]`; Perplexity Responses supports native web search with `tools=[{"type": "web_search"}, {"type": "fetch_url"}]`.
- 2026-05-05: Perplexity structured output uses the official `perplexityai` SDK, `responses.create`, and JSON Schema `response_format`, then validates `response.output_text` through the requested Pydantic model.
- 2026-05-05: Auto mode needs provider-aware schema design because Perplexity structured extraction expects a Perplexity-generated compatible Pydantic schema and preset model before the normal row-processing pipeline starts.
- 2026-05-05: Structured-output caching should store raw JSON and re-validate through the requested Pydantic model on each read, avoiding pickled SDK clients or dynamic model instances in the cache.
- 2026-05-07: Provider-generated auto-mode schemas may use postponed annotations like `Optional[int]` without importing `Optional`; generated files now include common typing imports and the dynamic Pydantic loader rebuilds models with a fallback typing namespace.
- 2026-05-07: Perplexity structured outputs require every object schema to set `required` to every key in `properties`, even nullable or defaulted Pydantic fields; schema normalization now applies that strict shape recursively.
