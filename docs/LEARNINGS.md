# Learnings

- 2026-05-05: Current provider interfaces differ by vendor. OpenAI's recommended text generation path is the Responses API, Anthropic uses the Messages API, and Perplexity Sonar supports OpenAI-compatible chat completions at `https://api.perplexity.ai`.
- 2026-05-05: Structured outputs remain OpenAI-specific in this project because the existing implementation uses `client.responses.parse(...)` with Pydantic models.
- 2026-05-05: Client construction should happen after CSV and argument validation so local input errors do not require valid SDK transport setup or live API credentials.
- 2026-05-05: The old root-level `my_model.py` and `my_model_broad.py` files were example Pydantic schemas for email structured extraction. They now live under `examples/` with README usage.
- 2026-05-05: Users expect `--pydantic-model-column-prefix` to create scalar columns for nested structured output fields, not JSON blobs for each top-level nested object.
- 2026-05-05: Large nested Pydantic schemas can be more reliable when filled one leaf field at a time, then reassembled and validated against the original model.
- 2026-05-05: Iterative field extraction should consume spare `--parallel` capacity without multiplying row workers by field workers into unbounded API concurrency.
