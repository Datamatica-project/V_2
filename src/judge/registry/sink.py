# src/judge/registry/sink.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Union

# ✅ DTO 단일화
from ...common.dto.dto import JudgeAttributionRecord, JudgeVerdictRecord

# ✅ 같은 폴더 기준 상대 import
from .manifest import ensure_segment_registered
from .run_paths import get_attributions_segment_path, get_verdicts_segment_path


def _append_jsonl(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def _segment_id(batch_index: int, segment_size_batches: int) -> int:
    if batch_index < 0:
        raise ValueError("batch_index must be >= 0")
    if segment_size_batches <= 0:
        raise ValueError("segment_size_batches must be > 0")
    return batch_index // segment_size_batches


def _to_json_line(record: Union[dict, Any]) -> str:
    """
    record -> JSON line (ensure_ascii=False)

    - dict면 그대로 dump
    - pydantic v2: model_dump_json / model_dump
    - pydantic v1: json() / dict()
    - 기타: repr로 감싸서 보존
    """
    if isinstance(record, dict):
        return json.dumps(record, ensure_ascii=False)

    # pydantic v2
    if hasattr(record, "model_dump_json"):
        try:
            return record.model_dump_json(ensure_ascii=False)  # type: ignore[attr-defined]
        except Exception:
            pass
    if hasattr(record, "model_dump"):
        try:
            d = record.model_dump()  # type: ignore[attr-defined]
            return json.dumps(d, ensure_ascii=False)
        except Exception:
            pass

    # pydantic v1
    if hasattr(record, "json"):
        try:
            return record.json(ensure_ascii=False)  # type: ignore[attr-defined]
        except TypeError:
            # v1 일부 환경은 ensure_ascii 인자 미지원일 수 있어 fallback
            try:
                return record.json()  # type: ignore[attr-defined]
            except Exception:
                pass
        except Exception:
            pass
    if hasattr(record, "dict"):
        try:
            d = record.dict()  # type: ignore[attr-defined]
            return json.dumps(d, ensure_ascii=False)
        except Exception:
            pass

    # 최후 보루
    return json.dumps({"_raw": repr(record)}, ensure_ascii=False)


def append_verdict(
    run_id: str,
    batch_index: int,
    record: Union[JudgeVerdictRecord, dict],
    *,
    segment_size_batches: int = 100,
    include_gateway_raw_zip: bool = False,
) -> None:
    seg_id = _segment_id(batch_index, segment_size_batches)

    ensure_segment_registered(
        run_id,
        seg_id,
        segment_size_batches=segment_size_batches,
        include_gateway_raw_zip=include_gateway_raw_zip,
    )

    p = get_verdicts_segment_path(run_id, seg_id)
    line = _to_json_line(record)
    _append_jsonl(p, line)


def append_attribution(
    run_id: str,
    batch_index: int,
    record: Union[JudgeAttributionRecord, dict],
    *,
    segment_size_batches: int = 100,
    include_gateway_raw_zip: bool = False,
) -> None:
    seg_id = _segment_id(batch_index, segment_size_batches)

    ensure_segment_registered(
        run_id,
        seg_id,
        segment_size_batches=segment_size_batches,
        include_gateway_raw_zip=include_gateway_raw_zip,
    )

    p = get_attributions_segment_path(run_id, seg_id)
    line = _to_json_line(record)
    _append_jsonl(p, line)
