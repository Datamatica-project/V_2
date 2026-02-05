
from __future__ import annotations

import json

from src.ensemble.export import export_coco_json, export_yolo_txt


def _verdict_record_pass():
    # minimal structure expected by export.py
    return {
        "image_id": "img1",
        "verdict": "PASS_2",
        "objects": [
            {
                "object_index": 0,
                "cls": 3,
                "agree_count": 2,
                "level": "PASS_2",
                "fused": {"xyxy": [10, 20, 110, 220], "conf": 0.77, "method": "conf_weighted_avg"},
                "members": {},
            }
        ],
    }


def test_export_yolo_txt(tmp_path):
    p = export_yolo_txt(_verdict_record_pass(), out_dir=tmp_path, image_wh=(1280, 720))
    assert p is not None and p.exists()

    txt = p.read_text(encoding="utf-8").strip().split()
    assert txt[0] == "3"  # cls id
    assert len(txt) == 5  # cls + 4 floats


def test_export_coco_json(tmp_path):
    p = export_coco_json(_verdict_record_pass(), out_dir=tmp_path, image_wh=(1280, 720))
    assert p is not None and p.exists()

    data = json.loads(p.read_text(encoding="utf-8"))
    assert "images" in data and "annotations" in data
    assert len(data["images"]) == 1
    assert len(data["annotations"]) == 1
    ann = data["annotations"][0]
    assert ann["category_id"] == 3
    assert len(ann["bbox"]) == 4
