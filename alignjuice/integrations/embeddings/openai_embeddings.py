"""
OpenAI embedding backend for AlignJuice.

Provides embedding generation using OpenAI API.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import numpy as np

from alignjuice.integrations.embeddings.base import EmbeddingBackend, EmbeddingResult
from alignjuice.core.registry import register_embedding


@register_embedding("openai_embeddings")
class OpenAIEmbeddingBackend(EmbeddingBackend):
    """
    OpenAI backend for cloud embedding generation.

    Uses OpenAI's embedding API (text-embedding-3-small, text-embedding-3-large, etc.).
    """

    name = "openai_embeddings"

    # Model dimensions
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,  # OpenAI supports larger batches
        normalize: bool = True,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        """
        Initialize OpenAI embedding backend.

        Args:
            model: OpenAI embedding model name
            batch_size: Batch size for API calls
            normalize: Whether to L2-normalize embeddings
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            base_url: API base URL
            timeout: Request timeout in seconds
        """
        super().__init__(model, batch_size, normalize, **kwargs)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key.")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        self._dimension = self.MODEL_DIMENSIONS.get(model)

    def embed(self, texts: list[str]) -> EmbeddingResult:
        """Generate embeddings using OpenAI API."""
        payload = {
            "model": self.model,
            "input": texts,
        }

        try:
            response = self._client.post(
                f"{self.base_url}/embeddings",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            # Extract embeddings in order
            embeddings_data = sorted(data["data"], key=lambda x: x["index"])
            embeddings = np.array([e["embedding"] for e in embeddings_data])

            # Normalize if requested
            if self.normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / (norms + 1e-9)

            usage = data.get("usage", {})

            return EmbeddingResult(
                embeddings=embeddings,
                model=data.get("model", self.model),
                usage={
                    "total_tokens": usage.get("total_tokens", 0),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                },
            )
        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            raise RuntimeError(f"OpenAI embedding API error ({e.response.status_code}): {error_body}") from e
        except httpx.HTTPError as e:
            raise RuntimeError(f"OpenAI embedding API connection error: {e}") from e

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            result = self.embed(["test"])
            self._dimension = result.dimension
        return self._dimension

    def health_check(self) -> bool:
        """Check if OpenAI API is accessible."""
        try:
            result = self.embed(["test"])
            return result.embeddings.shape[0] == 1
        except Exception:
            return False

    def __del__(self) -> None:
        """Clean up HTTP client."""
        if hasattr(self, "_client"):
            self._client.close()
