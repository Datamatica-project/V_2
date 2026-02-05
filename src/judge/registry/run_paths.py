# judge/registry/run_paths.py
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/workspace")).resolve()
JUDGE_ROOT = PROJECT_ROOT / "judge"
RUNS_ROOT = JUDGE_ROOT / "runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)


def get_run_dir(run_id: str) -> Path:
    d = RUNS_ROOT / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_status_path(run_id: str) -> Path:
    return get_run_dir(run_id) / "status.json"


def get_result_path(run_id: str) -> Path:
    # 최종 스냅샷(집계본) 1개 JSON (너가 원한 “계속 갱신되는 하나의 JSON”)
    return get_run_dir(run_id) / "result.json"


def get_segments_dir(run_id: str) -> Path:
    d = get_run_dir(run_id) / "segments"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_manifest_path(run_id: str) -> Path:
    return get_run_dir(run_id) / "manifest.json"


def get_verdicts_segment_path(run_id: str, segment_id: int) -> Path:
    return get_segments_dir(run_id) / f"verdicts_s{segment_id:05d}.jsonl"


def get_attributions_segment_path(run_id: str, segment_id: int) -> Path:
    return get_segments_dir(run_id) / f"attributions_s{segment_id:05d}.jsonl"


def get_gateway_raw_zip_path(run_id: str, segment_id: int) -> Path:
    # 옵션: raw request/response를 세그먼트 단위 zip으로 묶고 싶을 때
    return get_segments_dir(run_id) / f"gateway_raw_s{segment_id:05d}.zip"
