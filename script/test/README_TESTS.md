
# V2 Test Pack (pytest)

## 설치
pip install -U pytest

## 실행 (repo root = V_2)
pytest -q

## 포함 테스트
- ensemble/match.py: IoU + 클러스터링
- ensemble/core.py : PASS_3 / FAIL_SINGLE / MISS 시나리오
- ensemble/export.py: YOLO TXT / COCO JSON export
- gateway/bundler.py: weight_dir 스코프 주입 정책
- judge/judge_entry.py: judge -> ensemble delegation + DTO 변환

## 의도
- GPU / 실제 모델 추론 없이도 "전체 파이프라인 핵심 로직"이 깨지지 않는지 빠르게 검증
- 실제 모델 컨테이너/학습은 별도의 e2e(통합) 테스트에서 확인 권장
