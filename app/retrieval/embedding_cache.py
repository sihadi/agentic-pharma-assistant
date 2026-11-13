"""Content-addressed embedding cache.

Key = sha256(model + text). Changing the embedding model or a chunk's text
invalidates exactly the affected vectors. The cache is an optimisation only
(faster restarts / evaluation runs); the FAISS index itself is always built in
memory at start-up, and a missing or read-only cache is not an error.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)
EmbedFn = Callable[[list[str]], np.ndarray]


class EmbeddingCache:
    def __init__(self, directory: Path | None, model: str) -> None:
        self.model = model
        slug = re.sub(r"[^A-Za-z0-9_.-]", "_", model) or "model"
        self.path = directory / f"embeddings-{slug}.npz" if directory else None
        self._vectors: dict[str, np.ndarray] = {}
        if self.path and self.path.is_file():
            try:
                data = np.load(self.path, allow_pickle=False)
                self._vectors = dict(zip(data["keys"].tolist(), data["vectors"]))
            except (OSError, ValueError, KeyError):
                LOGGER.warning("embedding_cache_unreadable", extra={"path": str(self.path)})

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}\n{text}".encode()).hexdigest()

    def embed(self, texts: list[str], embed_fn: EmbedFn) -> tuple[np.ndarray, int]:
        """Return vectors for ``texts`` (cached or fresh) and the number computed."""

        keys = [self._key(t) for t in texts]
        missing = sorted({k: t for k, t in zip(keys, texts) if k not in self._vectors}.items())
        if missing:
            fresh = embed_fn([t for _, t in missing])
            for (key, _), vector in zip(missing, fresh):
                self._vectors[key] = vector
            self._save()
        return np.stack([self._vectors[k] for k in keys]).astype("float32"), len(missing)

    def _save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            keys = np.array(list(self._vectors), dtype=str)
            np.savez(self.path, keys=keys, vectors=np.stack(list(self._vectors.values())))
        except OSError:
            LOGGER.warning("embedding_cache_not_written", extra={"path": str(self.path)})
