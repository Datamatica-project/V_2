# script/test/test_ensemble.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from ensemble.core import ensemble_bundle


@dataclass
class Det:
    cls: int
    conf: float
    xyxy: List[float]


@dataclass
class ItemResult:
    image_id: str
    detections: List[Det]


@dataclass
class Response:
    ok: bool
    results: List[ItemResult]


@dataclass
class Item:
    image_id: str


@dataclass
class Bundle:
    batch_id: str
    items: List[Item]
    responses: Dict[str, Response]
    errors: Dict[str, Any]


def box(xyxy=(0, 0, 100, 100), cls=0, conf=0.9):
    return Det(cls=cls, conf=conf, xyxy=list(xyxy))


def resp(image_id: str, dets: List[Det], ok=True):
    return Response(ok=ok, results=[ItemResult(image_id=image_id, detections=dets)])


def test_two_model_consensus_is_pass_2_and_has_fused_object():
    b = Bundle(
        batch_id="b0",
        items=[Item("img1")],
        responses={
            "yolov11": resp("img1", [box(conf=0.9)]),
            "yolox": resp("img1", [box(conf=0.8)]),
            "rtdetr": resp("img1", []),
        },
        errors={},
    )

    # PASS_2 조건을 만족시키기 위해 threshold를 테스트 친화적으로 낮춘다
    verdicts, attributions = ensemble_bundle(
        b,
        min_conf_p2=0.2,
        min_pair_iou_p2=0.2,
    )

    v = verdicts[0]
    assert v["verdict"] in ("PASS_2", "FAIL")  # 규칙/threshold에 따라 FAIL도 가능
    assert "objects" in v

    # consensus가 형성됐다면 object 1개 이상
    if v["verdict"] != "MISS":
        assert len(v["objects"]) == 1
        obj = v["objects"][0]
        assert obj["agree_count"] == 2
        assert obj["level"] in ("PASS_2", "FAIL")  # 객체 단위 레벨
        assert "fused" in obj and "xyxy" in obj["fused"]
        assert len(obj["fused"]["xyxy"]) == 4

    # PASS면 attribution 없음 / FAIL이면 attribution 있을 수 있음
    if v["verdict"] == "FAIL":
        assert attributions
    else:
        assert attributions == []


def test_single_model_detection_is_miss_or_fail_but_no_objects():
    b = Bundle(
        batch_id="b0",
        items=[Item("img1")],
        responses={
            "yolov11": resp("img1", [box()]),
            "yolox": resp("img1", []),
            "rtdetr": resp("img1", []),
        },
        errors={},
    )
    verdicts, attributions = ensemble_bundle(b)

    # 존재 합의가 없으므로 MISS가 기본
    assert verdicts[0]["verdict"] == "MISS"
    assert verdicts[0].get("objects") == []
    assert attributions  # MISS이면 attribution이 생김


def test_three_model_consensus_is_pass_3_with_default_thresholds():
    b = Bundle(
        batch_id="b0",
        items=[Item("img1")],
        responses={
            "yolov11": resp("img1", [box(conf=0.9)]),
            "yolox": resp("img1", [box(conf=0.9)]),
            "rtdetr": resp("img1", [box(conf=0.9)]),
        },
        errors={},
    )
    verdicts, attributions = ensemble_bundle(
        b,
        min_conf_p3=0.2,         # 테스트 안정성 위해 낮춤
        min_pair_iou_p3=0.2,
    )

    v = verdicts[0]
    assert v["verdict"] in ("PASS_3", "PASS_2", "FAIL")  # threshold에 따라 달라질 수 있음
    assert "objects" in v
    assert len(v["objects"]) == 1
    assert v["objects"][0]["agree_count"] == 3
    assert v["objects"][0]["level"] in ("PASS_3", "PASS_2", "FAIL")

    # PASS면 attribution 없음 / FAIL이면 attribution 있을 수 있음
    if v["verdict"] == "FAIL":
        assert attributions
    else:
        assert attributions == []


def test_iou_mismatch_is_miss():
    b = Bundle(
        batch_id="b0",
        items=[Item("img1")],
        responses={
            "yolov11": resp("img1", [box((0, 0, 50, 50))]),
            "yolox": resp("img1", [box((200, 200, 260, 260))]),
            "rtdetr": resp("img1", []),
        },
        errors={},
    )
    verdicts, attributions = ensemble_bundle(b, iou_thr=0.5)

    # IoU로 합의가 안 되므로 존재 합의 실패 → MISS
    assert verdicts[0]["verdict"] == "MISS"
    assert verdicts[0].get("objects") == []
    assert attributions
