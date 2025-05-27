"""
Utility module for obtaining text embeddings from different providers.

Currently implemented provider(s):
- OpenAI   (requires `openai` Python package and an `OPENAI_API_KEY` env var)

The public interface intentionally hides provider-specific details
behind two helper functions:

    get_embedding(text, model_provider='OpenAI', model='text-embedding-3-large')
        → List[float]

    list_available_embedding_models()
        → Dict[str, List[str]]
"""

from typing import Dict, List

# --------------------------------------------------------------------------- #
# Registry of supported providers and models                                  #
# --------------------------------------------------------------------------- #
_AVAILABLE_EMBEDDINGS: Dict[str, List[str]] = {
    "OpenAI": [
        "text-embedding-3-small",
        "text-embedding-3-large",
    ]
}


def list_available_embedding_models() -> Dict[str, List[str]]:
    """
    Return a mapping of provider → list of supported models.

    This can be used by callers (e.g. CLI) to populate `choices=` in argparse.
    """
    return _AVAILABLE_EMBEDDINGS


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def get_embedding(
    input_string: str,
    model_provider: str = "OpenAI",
    model: str = "text-embedding-3-large",
) -> List[float]:
    """
    Return the embedding vector for `input_string` using the selected provider.

    Args:
        input_string: Text to embed.
        model_provider: Name of the provider (e.g. "OpenAI").
        model: Model name supported by the provider.

    Raises:
        ValueError: If the provider / model is not supported.
        ImportError: If the provider's SDK is not installed.
        Exception: Any error raised by the provider's client.
    """
    if model_provider not in _AVAILABLE_EMBEDDINGS:
        raise ValueError(
            f"Unknown embedding provider '{model_provider}'. "
            f"Known providers: {', '.join(_AVAILABLE_EMBEDDINGS)}"
        )

    if model not in _AVAILABLE_EMBEDDINGS[model_provider]:
        raise ValueError(
            f"Model '{model}' not recognised for provider '{model_provider}'. "
            f"Available models: {', '.join(_AVAILABLE_EMBEDDINGS[model_provider])}"
        )

    if model_provider == "OpenAI":
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise ImportError(
                "The 'openai' Python package is required for OpenAI embeddings. "
                "Install with: pip install openai"
            ) from exc

        client = OpenAI()
        response = client.embeddings.create(input=input_string, model=model)
        return response.data[0].embedding  # type: ignore[return-value]

    # Fallback for not-yet-implemented providers
    raise NotImplementedError(
        f"Embedding provider '{model_provider}' is recognised but not implemented."
    )
