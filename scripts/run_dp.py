"""이 파일은 DP 최적표를 만들고 출판 참조표와 셀 단위로 사전 비교한다
입력: --pre-register 플래그, --out 사전등록 JSON 경로
출력: expected_mismatch.json 파일 1개와 표준출력 요약표
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from blackjack_rl.chartspec import CHART_ROWS, DEALER_COLS, ChartTable
from blackjack_rl.dp.exact import action_values, optimal_chart, solve_optimal
from blackjack_rl.reference import (
    REFERENCE_RULES,
    TIE_DELTA,
    ReferenceChart,
    load_reference_chart,
)
from blackjack_rl.rules import RULES_V1, RuleSet
from blackjack_rl.state import DOUBLE, HIT, SPLIT, STAND

DEFAULT_OUT = (Path(__file__).resolve().parents[1]
               / "src" / "blackjack_rl" / "data" / "expected_mismatch.json")

NOTATION_TO_ACTION = {"S": STAND, "H": HIT, "D": DOUBLE, "Ds": DOUBLE, "Y": SPLIT}


def value_of_notation(q: np.ndarray, notation: str) -> float:
    """표기 하나가 가리키는 행동의 DP 가치를 돌려준다."""
    if notation == "N":
        # 왜: 'N'은 단일 행동이 아니라 '스플릿 말고 나머지 중 최선'이라는 뜻이다.
        without_split = np.delete(q, SPLIT)
        return float(np.nanmax(without_split))
    return float(q[NOTATION_TO_ACTION[notation]])


def classify(dp_delta: float) -> str:
    """손해 크기만 보고 사유를 둘 중 하나로 정한다."""
    # 왜: 참조표 로딩이 덱 모형 말고 모든 규칙이 같음을 이미 강제했으므로
    #     rule_diff는 여기서 나올 수 없다. 남는 원인은 동점 아니면 덱 모형뿐이다.
    if abs(dp_delta) < TIE_DELTA:
        return "statistical_tie"
    return "deck_model"


def find_mismatches(dp_chart: ChartTable, ref_chart: ReferenceChart,
                    rules: RuleSet) -> list[dict]:
    """DP표와 출판표가 다른 칸을 모아 손해 크기와 사유를 붙인다."""
    rows: list[dict] = []
    for i, chart_row in enumerate(CHART_ROWS):
        for j, dealer_up in enumerate(DEALER_COLS):
            dp_note = str(dp_chart.notation[i, j])
            ref_note = str(ref_chart.notation[i, j])
            if dp_note == ref_note:
                continue
            key = chart_row.to_key(dealer_up, can_double=True)
            q = action_values(key, rules)
            dp_delta = value_of_notation(q, dp_note) - value_of_notation(q, ref_note)
            rows.append({
                "cell": chart_row.label,
                "dealer": dealer_up,
                "dp": dp_note,
                "ref": ref_note,
                "dp_delta": round(dp_delta, 6),
                "cause": classify(dp_delta),
            })
    return rows


def build_pre_registration(rules: RuleSet = RULES_V1,
                           reference_rules: RuleSet = REFERENCE_RULES) -> dict:
    """학습 시작 전에 '원래 다를 칸' 목록을 만든다."""
    dp_chart = optimal_chart(rules)
    ref_chart = load_reference_chart(reference_rules)
    entries = find_mismatches(dp_chart, ref_chart, rules)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dp_rules_fp": rules.fingerprint(),
        "reference_rules_fp": reference_rules.fingerprint(),
        "reference_source": ref_chart.source,
        "tie_delta": TIE_DELTA,
        "n_cells": len(CHART_ROWS) * len(DEALER_COLS),
        "n_mismatch": len(entries),
        "entries": entries,
    }


def write_pre_registration(payload: dict, path: Path) -> Path:
    """사전등록 JSON을 파일로 쓴다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return path


def render_summary(payload: dict) -> str:
    """사전등록 결과를 마크다운 표로 만든다."""
    lines = [
        f"DP 규칙 지문: {payload['dp_rules_fp']} (무한덱)",
        f"참조표 규칙 지문: {payload['reference_rules_fp']} (4덱 슈)",
        f"전체 {payload['n_cells']}칸 중 사전등록 불일치 {payload['n_mismatch']}칸",
        "",
        "| 셀 | 딜러 | DP | 출판표 | DP 손해 | 사유 |",
        "|---|---|---|---|---|---|",
    ]
    for entry in payload["entries"]:
        dealer = "A" if entry["dealer"] == 11 else str(entry["dealer"])
        lines.append(
            f"| {entry['cell']} | {dealer} | {entry['dp']} | {entry['ref']} | "
            f"{entry['dp_delta']:.5f} | {entry['cause']} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="DP 정답표 생성과 출판표 사전 비교")
    parser.add_argument("--pre-register", action="store_true",
                        help="출판표와 비교한 예상 불일치 칸을 JSON으로 사전등록한다")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    result = solve_optimal(RULES_V1)
    print(f"규칙 지문 {result.rules_fp} / 최적 플레이 정확 EV {result.ev_initial:+.5f}")

    if args.pre_register:
        payload = build_pre_registration()
        path = write_pre_registration(payload, Path(args.out))
        print()
        print(render_summary(payload))
        print(f"\n사전등록 저장: {path}")


if __name__ == "__main__":
    main()
