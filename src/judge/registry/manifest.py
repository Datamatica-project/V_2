# src/judge/registry/manifest.py
from __future__ import annotations

import json
from typing import Any, Dict

# ✅ registry 내부 상대 import로 통일
from .run_paths import (
    get_manifest_path,
    get_verdicts_segment_path,
    get_attributions_segment_path,
    get_gateway_raw_zip_path,
)


def _load_or_init(run_id: str, segment_size_batches: int) -> Dict[str, Any]:
    p = get_manifest_path(run_id)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))

    m = {
        "run_id": run_id,
        "segment_size_batches": segment_size_batches,
        "segments": [],
    }
    p.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    return m


def ensure_segment_registered(
    run_id: str,
    segment_id: int,
    *,
    segment_size_batches: int,
    include_gateway_raw_zip: bool,
) -> None:
    """
    세그먼트 파일이 처음 쓰일 때 manifest에 등록.
    이미 등록되어 있으면 no-op.
    """
    m = _load_or_init(run_id, segment_size_batches)

    for s in m.get("segments", []):
        if int(s.get("segment_id")) == int(segment_id):
            return  # already registered

    start = segment_id * segment_size_batches
    end = start + segment_size_batches - 1

    entry = {
        "segment_id": segment_id,
        "batch_index_range": [start, end],
        "files": {
            "verdicts": str(
                get_verdicts_segment_path(run_id, segment_id).relative_to(
                    get_manifest_path(run_id).parent
                )
            ),
            "attributions": str(
                get_attributions_segment_path(run_id, segment_id).relative_to(
                    get_manifest_path(run_id).parent
                )
            ),
        },
    }

    if include_gateway_raw_zip:
        entry["files"]["gateway_raw_zip"] = str(
            get_gateway_raw_zip_path(run_id, segment_id).relative_to(
                get_manifest_path(run_id).parent
            )
        )

    m["segments"].append(entry)
    m["segments"].sort(key=lambda x: int(x["segment_id"]))

    get_manifest_path(run_id).write_text(
        json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8"
    )
