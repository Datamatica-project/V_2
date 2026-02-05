# src/judge/orchestrator/status.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

STATUS_DIR = Path("/workspace/logs/judge/status")
STATUS_DIR.mkdir(parents=True, exist_ok=True)

def _path(run_id: str) -> Path:
    return STATUS_DIR / f"{run_id}.json"

def init_status(run_id: str, total_images: int) -> None:
    p = _path(run_id)
    payload = {
        "run_id": run_id,
        "status": "RUNNING",
        "total_images": int(total_images),
        "processed_images": 0,
        "started_at": time.time(),
        "updated_at": time.time(),
        "error": None,
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def read_status(run_id: str) -> Dict[str, Any]:
    p = _path(run_id)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return json.loads(p.read_text(encoding="utf-8"))

def update_status(run_id: str, processed_images: int) -> None:
    s = read_status(run_id)
    s["processed_images"] = int(processed_images)
    s["updated_at"] = time.time()
    _path(run_id).write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")

def mark_done(run_id: str) -> None:
    s = read_status(run_id)
    s["status"] = "DONE"
    s["updated_at"] = time.time()
    s["finished_at"] = time.time()
    _path(run_id).write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")

def mark_failed(run_id: str, error: str) -> None:
    s = read_status(run_id)
    s["status"] = "FAIL"
    s["error"] = str(error)
    s["updated_at"] = time.time()
    s["finished_at"] = time.time()
    _path(run_id).write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
