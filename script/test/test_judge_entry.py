
from __future__ import annotations

from src.common.dto.dto import (
    Detection,
    GatewayBundle,
    InferItem,
    InferItemResult,
    JudgeVerdictRecord,
    ModelPredResponse,
)

from src.judge.judge_core.judge_entry import judge_bundle


def _resp(model: str, batch_id: str, image_id: str, dets: list[Detection]):
    return ModelPredResponse(
        model=model,
        batch_id=batch_id,
        ok=True,
        results=[InferItemResult(image_id=image_id, detections=dets)],
        raw={"meta": {}},
    )


def test_judge_bundle_returns_dto_records():
    batch_id = "b1"
    items = [InferItem(image_id="img1", image_path="/tmp/a.jpg", width=1280, height=720)]

    det1 = Detection(cls=0, conf=0.9, xyxy=[100, 100, 200, 200])
    det2 = Detection(cls=0, conf=0.8, xyxy=[101, 99, 199, 201])

    bundle = GatewayBundle(
        batch_id=batch_id,
        items=items,
        responses={
            "yolov11": _resp("yolov11", batch_id, "img1", [det1]),
            "rtm":     _resp("rtm",     batch_id, "img1", [det2]),
            "rtdetr":  _resp("rtdetr",  batch_id, "img1", []),
        },
        errors={},
    )

    verdicts, attrs = judge_bundle(bundle, ensemble_cfg={"ensemble": {"iou_thr": 0.5, "min_agree": 2}})
    assert verdicts and isinstance(verdicts[0], JudgeVerdictRecord)
    assert verdicts[0].image_id == "img1"
    # Since two models agree, should be PASS_2 or FAIL depending on thresholds; ensure not MISS
    assert verdicts[0].verdict in ("PASS_2", "FAIL", "PASS_3")
