
from __future__ import annotations

from math import isclose

import pytest

from src.ensemble.match import Box, cluster_consensus, iou_xyxy


def test_iou_xyxy_basic():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (0.0, 0.0, 10.0, 10.0)
    assert isclose(iou_xyxy(a, b), 1.0)

    c = (10.0, 10.0, 20.0, 20.0)  # touch at corner -> no overlap
    assert isclose(iou_xyxy(a, c), 0.0)

    d = (5.0, 5.0, 15.0, 15.0)  # 25 overlap / (100+100-25)=25/175
    assert isclose(iou_xyxy(a, d), 25.0 / 175.0)


def test_cluster_consensus_groups_by_class_and_iou():
    boxes = [
        Box(model="yolov11", cls=0, conf=0.9, xyxy=(0, 0, 10, 10)),
        Box(model="rtm",     cls=0, conf=0.8, xyxy=(1, 1, 11, 11)),  # iou high with first
        Box(model="rtdetr",  cls=1, conf=0.7, xyxy=(0, 0, 10, 10)),  # different class
    ]

    clusters = cluster_consensus(boxes, iou_thr=0.5, min_agree=2)
    assert len(clusters) == 1
    c = clusters[0]
    assert c.cls == 0
    assert c.agree_count() == 2
    assert set(c.members.keys()) == {"yolov11", "rtm"}


def test_cluster_consensus_no_duplicate_model_in_cluster():
    boxes = [
        Box(model="yolov11", cls=0, conf=0.9, xyxy=(0, 0, 10, 10)),
        Box(model="yolov11", cls=0, conf=0.8, xyxy=(1, 1, 11, 11)),  # same model
        Box(model="rtm",     cls=0, conf=0.7, xyxy=(0, 0, 10, 10)),
    ]
    clusters = cluster_consensus(boxes, iou_thr=0.5, min_agree=2)
    # Should still form a cluster with yolov11 + rtm (one yolov11 only)
    assert len(clusters) == 1
    assert clusters[0].agree_count() == 2
