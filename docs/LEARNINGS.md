# Learnings

- 2026-05-05: Current provider interfaces differ by vendor. OpenAI's recommended text generation path is the Responses API, Anthropic uses the Messages API, and Perplexity Sonar supports OpenAI-compatible chat completions at `https://api.perplexity.ai`.
- 2026-05-05: Structured outputs remain OpenAI-specific in this project because the existing implementation uses `client.responses.parse(...)` with Pydantic models.
- 2026-05-05: Client construction should happen after CSV and argument validation so local input errors do not require valid SDK transport setup or live API credentials.
