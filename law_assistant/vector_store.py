from __future__ import annotations

import os
import faiss  # type: ignore
import orjson
import numpy as np
from typing import List, Dict, Tuple


class VectorStore:
    def __init__(self, dim: int, index: faiss.Index, meta_path: str):
        self.dim = dim
        self.index = index
        self.meta_path = meta_path
        self._meta: List[Dict] = []

    @property
    def size(self) -> int:
        return self.index.ntotal

    @staticmethod
    def _ensure_parent(path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    @classmethod
    def create(cls, dim: int, meta_path: str) -> "VectorStore":
        index = faiss.IndexFlatIP(dim)
        return cls(dim, index, meta_path)

    @classmethod
    def load(cls, index_path: str, meta_path: str) -> "VectorStore":
        index = faiss.read_index(index_path)
        # dim is stored in index
        dim = index.d
        vs = cls(dim, index, meta_path)
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                vs._meta = orjson.loads(f.read())
        return vs

    def add(self, embeddings: List[List[float]], metadatas: List[Dict]) -> None:
        arr = np.array(embeddings, dtype=np.float32)
        # Normalize for cosine similarity when using IP index
        faiss.normalize_L2(arr)
        self.index.add(arr)
        self._meta.extend(metadatas)

    def search(self, query: List[float], k: int = 5) -> List[Tuple[int, float, Dict]]:
        q = np.array([query], dtype=np.float32)
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, k)
        results: List[Tuple[int, float, Dict]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            md = self._meta[idx] if 0 <= idx < len(self._meta) else {}
            results.append((int(idx), float(score), md))
        return results

    def save(self, index_path: str, meta_path: str | None = None) -> None:
        meta_path = meta_path or self.meta_path
        self._ensure_parent(index_path)
        self._ensure_parent(meta_path)
        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            f.write(orjson.dumps(self._meta))

