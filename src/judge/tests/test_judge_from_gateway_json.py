"""
Test judge_core using a real GatewayBundle JSON input.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.common.dto.dto import GatewayBundle
from src.judge.judge_core.judge_entry import judge_bundle


FIXTURE = Path("src/judge/tests/gateway_bundle_64_mixed.json")


def main():
    print(f"Loading gateway bundle from: {FIXTURE}")

    raw = json.loads(FIXTURE.read_text())

    bundle = GatewayBundle.model_validate(raw)

    verdicts, attributions = judge_bundle(bundle)

    hist = {}
    for v in verdicts:
        hist[v.verdict] = hist.get(v.verdict, 0) + 1

    print("\n=== Verdict histogram ===")
    for k, v in sorted(hist.items()):
        print(f"{k:7s}: {v}")

    print("\n=== Sample verdicts (first 10) ===")
    for v in verdicts[:10]:
        print(f"{v.image_id}: {v.verdict}")

    print(f"\nFAIL attributions: {len(attributions)}")
    if attributions:
        print("Sample FAIL reasons:")
        for r in attributions[:3]:
            print(f"- {r.image_id}: {[x.type for x in r.reasons]}")


if __name__ == "__main__":
    main()
