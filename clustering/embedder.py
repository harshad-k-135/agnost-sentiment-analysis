"""Sentence-transformer embedding helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence, TYPE_CHECKING

import numpy as np

from config import SETTINGS
from utils.logger import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


logger = get_logger(__name__)


@lru_cache(maxsize=1)
def load_embedding_model() -> "SentenceTransformer":
    """Load and cache the configured sentence-transformer model.

    Returns:
        A lazily instantiated SentenceTransformer model.
    """

    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model %s", SETTINGS.embedding_model)
    return SentenceTransformer(SETTINGS.embedding_model)


def encode_conversations(conversations: Sequence[str], model: "SentenceTransformer" | None = None) -> np.ndarray:
    """Encode conversations into normalized embedding vectors.

    Args:
        conversations: Cleaned conversation strings.
        model: Optional model instance for dependency injection.

    Returns:
        A two-dimensional NumPy array of float embeddings.

    Raises:
        ValueError: If no conversations are supplied.
    """

    if not conversations:
        raise ValueError("At least one conversation is required for embedding.")

    active_model = model or load_embedding_model()
    embeddings = active_model.encode(list(conversations), show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=float)
