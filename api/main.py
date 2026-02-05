from __future__ import annotations

import os
from fastapi import FastAPI

# 각각의 모델 앱을 "서브앱"으로 마운트 (원하면 prefix 없이도 가능)
from api.yolov11.server import app as yolov11_app
from api.rtdetr.server import app as rtdetr_app
from api.yolox.server import app as yolox_app


def create_app() -> FastAPI:
    app = FastAPI(title="V2 Model APIs", version="0.1.0")

    # 환경변수로 마운트 여부를 제어할 수도 있음
    mount_mode = os.getenv("V2_MOUNT_MODE", "subapps").lower().strip()
    # subapps: /yolov11, /rtdetr, /yolox 로 분리
    # flat: 각 앱을 따로 띄우는 운영(컨테이너 분리)이라면 main.py 안 써도 됨

    if mount_mode == "subapps":
        app.mount("/yolov11", yolov11_app)
        app.mount("/rtdetr", rtdetr_app)
        app.mount("/yolox", yolox_app)

        @app.get("/health")
        def health():
            return {"ok": True, "mode": "subapps"}

    else:
        # flat 모드: 이 app 자체는 의미 없음. 참고용.
        @app.get("/health")
        def health():
            return {"ok": True, "mode": "flat (no mounts)"}

    return app


app = create_app()
