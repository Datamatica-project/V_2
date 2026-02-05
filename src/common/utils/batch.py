# src/common/utils/batch.py
from __future__ import annotations

from typing import Iterator, List, Tuple

# ✅ DTO 단일화: src/common/dto/dto.py
from ..dto.dto import InferItem


def make_batches(
    items: List[InferItem],
    batch_size: int,
    *,
    batch_prefix: str = "batch",
) -> Iterator[Tuple[str, List[InferItem]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    n = len(items)
    i = 0
    idx = 0
    while i < n:
        chunk = items[i : i + batch_size]
        batch_id = f"{batch_prefix}_{idx:05d}"
        yield batch_id, chunk
        i += batch_size
        idx += 1
