from __future__ import annotations

import os
import requests
from typing import List, Protocol


class Embeddings(Protocol):
    def embed(self, texts: List[str]) -> List[List[float]]:  # pragma: no cover
        ...


class OllamaEmbeddings:
    def __init__(self, model: str = "nomic-embed-text", host: str | None = None):
        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        url = f"{self.host}/api/embeddings"
        for t in texts:
            resp = requests.post(url, json={"model": self.model, "input": t}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            out.append(data["embedding"])  # type: ignore[index]
        return out


class OpenAIEmbeddings:
    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        from openai import OpenAI  # lazy import

        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def embed(self, texts: List[str]) -> List[List[float]]:
        # Batch for efficiency and token limits
        res = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in res.data]


class AnthropicEmbeddings:
    def __init__(self, model: str = "claude-embedding-1", api_key: str | None = None):
        # Some Anthropic accounts have an embeddings endpoint; keep this flexible.
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

    def embed(self, texts: List[str]) -> List[List[float]]:
        # Use HTTP directly to avoid tight SDK coupling; if not available, raise helpful error.
        headers = {
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set for embeddings")
        url = f"{self.base_url}/v1/embeddings"
        out: List[List[float]] = []
        for t in texts:
            resp = requests.post(url, headers=headers, json={"model": self.model, "input": t}, timeout=60)
            if resp.status_code == 404:
                raise RuntimeError("Anthropic embeddings endpoint unavailable for this account/model")
            resp.raise_for_status()
            data = resp.json()
            # Expect data like {"data": {"embedding": [...]}}
            emb = data.get("data", {}).get("embedding") or data.get("embedding")
            if not emb:
                raise RuntimeError(f"Anthropic embeddings unexpected response: {data}")
            out.append(emb)
        return out


def resolve_embeddings(provider: str, model: str) -> Embeddings:
    p = provider.lower()
    if p == "ollama":
        return OllamaEmbeddings(model=model)
    if p == "openai":
        return OpenAIEmbeddings(model=model)
    if p == "anthropic":
        return AnthropicEmbeddings(model=model)
    raise ValueError(f"Unknown embeddings provider: {provider}")

