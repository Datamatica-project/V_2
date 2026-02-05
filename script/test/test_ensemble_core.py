
from __future__ import annotations

from src.common.dto.dto import (
    Detection,
    GatewayBundle,
    InferBatchRequest,
    InferItem,
    InferItemResult,
    ModelPredResponse,
)

from src.ensemble.core import ensemble_bundle


def _req():
    return InferBatchRequest(
        batch_id="b1",
        items=[
            InferItem(image_id="img1", image_path="/tmp/img1.jpg", width=1280, height=720),
            InferItem(image_id="img2", image_path="/tmp/img2.jpg", width=1280, height=720),
        ],
        params={"run_id": "r1"},
    )


def _resp(model: str, batch_id: str, per_image: dict[str, list[Detection]]):
    results = []
    for image_id, dets in per_image.items():
        results.append(InferItemResult(image_id=image_id, detections=dets))
    return ModelPredResponse(model=model, batch_id=batch_id, ok=True, results=results, raw={"meta": {}})


def test_ensemble_bundle_pass3_and_miss():
    req = _req()

    # img1: 3 models agree (PASS_3 expected)
    detA = Detection(cls=0, conf=0.9, xyxy=[100, 100, 200, 200])
    detB = Detection(cls=0, conf=0.8, xyxy=[102, 98, 198, 205])
    detC = Detection(cls=0, conf=0.85, xyxy=[101, 101, 201, 199])

    # img2: no detections at all -> MISS
    empty = []

    bundle = GatewayBundle(
        batch_id=req.batch_id,
        items=req.items,
        responses={
            "yolov11": _resp("yolov11", req.batch_id, {"img1": [detA], "img2": empty}),
            "rtm":     _resp("rtm",     req.batch_id, {"img1": [detB], "img2": empty}),
            "rtdetr":  _resp("rtdetr",  req.batch_id, {"img1": [detC], "img2": empty}),
        },
        errors={},
    )

    verdicts, attrs = ensemble_bundle(
        bundle,
        iou_thr=0.5,
        conf_thr=0.25,
        min_agree=2,
        strong_agree=3,
        min_conf_p3=0.1,  # make PASS_3 easier for this test
        min_pair_iou_p3=0.1,
        min_conf_p2=0.1,
        min_pair_iou_p2=0.1,
    )

    vm = {v["image_id"]: v["verdict"] for v in verdicts}
    assert vm["img1"] == "PASS_3"
    assert vm["img2"] == "MISS"

    # MISS should create attribution
    am = {a["image_id"]: a for a in attrs}
    assert "img2" in am


def test_ensemble_bundle_fail_single_when_only_one_model_detects():
    req = _req()
    det = Detection(cls=1, conf=0.7, xyxy=[10, 10, 100, 100])

    bundle = GatewayBundle(
        batch_id=req.batch_id,
        items=req.items,
        responses={
            "yolov11": _resp("yolov11", req.batch_id, {"img1": [det], "img2": []}),
            "rtm":     _resp("rtm",     req.batch_id, {"img1": [], "img2": []}),
            "rtdetr":  _resp("rtdetr",  req.batch_id, {"img1": [], "img2": []}),
        },
        errors={},
    )

    verdicts, attrs = ensemble_bundle(bundle, iou_thr=0.5, conf_thr=0.25, min_agree=2, strong_agree=3)

    vm = {v["image_id"]: v["verdict"] for v in verdicts}
    assert vm["img1"] == "FAIL"
    # FAIL should create attribution
    assert any(a["image_id"] == "img1" for a in attrs)
