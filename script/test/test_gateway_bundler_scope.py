
from __future__ import annotations

from pathlib import Path

from src.common.dto.dto import InferBatchRequest, InferItem
from src.gateway.bundler import _req_with_weight_scope


def _mk_req(params: dict):
    return InferBatchRequest(
        batch_id="b1",
        items=[InferItem(image_id="img1", image_path="/tmp/a.jpg")],
        params=params,
    )


def test_req_with_weight_scope_injects_model_dir_when_user_project_present(monkeypatch, tmp_path):
    # ensure WEIGHTS_DIR_PATH is stable by overriding env in runtime; bundler reads at import time
    # so we only test that returned params.weight_dir ends with /{model}
    req = _mk_req({"user_id": "u1", "project_id": "p1"})
    out = _req_with_weight_scope(req, "rtm")
    assert out.params["weight_dir"].endswith("/projects/u1/p1/rtm") or out.params["weight_dir"].endswith("\\projects\\u1\\p1\\rtm")


def test_req_with_weight_scope_respects_explicit_weight_file():
    req = _mk_req({"user_id": "u1", "project_id": "p1", "weight_path": "/weights/some.pt"})
    out = _req_with_weight_scope(req, "rtm")
    # should not overwrite / add weight_dir
    assert out is req


def test_req_with_weight_scope_appends_model_if_weight_dir_points_to_root():
    req = _mk_req({"user_id": "u1", "project_id": "p1", "weight_dir": "/weights/projects/u1/p1"})
    out = _req_with_weight_scope(req, "rtdetr")
    assert str(Path(out.params["weight_dir"]).name) == "rtdetr"
