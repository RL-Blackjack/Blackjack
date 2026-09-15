"""이 파일은 상태 한 칸의 DP 정확 EV와 학습 Q를 한 화면에 나란히 출력한다
입력: --total/--dealer/--soft/--pair 플래그와 선택적 --run 스냅샷 npz 경로
출력: 표준출력 한 화면(DP EV 4개 / 학습 Q 4개 / 방문수 N / 기준표 정답 / 우리 표기)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 왜: 발표 중에는 PYTHONPATH를 손으로 넣을 시간이 없다. 맨손 실행에서도 돌아야 한다.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from blackjack_rl.chartspec import (CHART_ROWS, DEALER_COLS, ROW_INDEX,  # noqa: E402
                                    ChartRow, project)
from blackjack_rl.dp.exact import action_values, optimal_chart  # noqa: E402
from blackjack_rl.reference import TIE_DELTA, load_reference_chart  # noqa: E402
from blackjack_rl.rules import RULES_V1, RuleSet  # noqa: E402
from blackjack_rl.state import (ACTION_NAMES, ACTION_NAMES_KO, ACTIONS,  # noqa: E402
                                StateKey, legal_actions)

TRUE_WORDS = ("true", "t", "yes", "y", "1")
FALSE_WORDS = ("false", "f", "no", "n", "0")
LINE = "=" * 62


def parse_bool(text: str) -> int:
    """'true'/'false' 같은 말을 0/1로 바꾼다."""
    말 = str(text).strip().lower()
    if 말 in TRUE_WORDS:
        return 1
    if 말 in FALSE_WORDS:
        return 0
    raise ValueError(f"true 또는 false로 적어라: {text!r}")


def dealer_index(dealer: int) -> int:
    """딜러 업카드를 StateKey 규약 2~11로 바꾼다(에이스는 11)."""
    if dealer in (1, 11):
        return 11
    if 2 <= dealer <= 10:
        return int(dealer)
    raise ValueError(f"딜러 업카드는 2~10 또는 A(1 또는 11)여야 한다: {dealer}")


def find_row(total: int, is_soft: int, can_split: int) -> ChartRow:
    """이 상태가 36행 표의 어느 행인지 찾는다."""
    # 왜: (total, is_soft, can_split) 세 값이 36행 안에서 유일하다.
    #     하드 2r의 랭크가 total//2로 복원되므로 페어 랭크 축이 따로 필요 없다.
    for row in CHART_ROWS:
        if (row.total == total and row.is_soft == is_soft
                and row.can_split == can_split):
            return row
    raise ValueError(
        f"36행 표에 없는 조합이다: total={total}, soft={is_soft}, pair={can_split}")


def load_q_and_n(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """스냅샷 npz에서 q_final과 n_final만 꺼낸다."""
    # 왜: allow_pickle=False가 재현성 5단 장치의 5번이다. 그리고 두 키만 읽으면
    #     스냅샷 스키마가 늘어나도 이 스크립트는 안 고쳐도 된다.
    with np.load(path, allow_pickle=False) as z:
        q = np.asarray(z["q_final"], dtype=np.float64)
        n = np.asarray(z["n_final"], dtype=np.int64)
    return q, n


def _머리말(key: StateKey, row: ChartRow, rules: RuleSet) -> list[str]:
    딜러이름 = "A" if key.dealer_up == 11 else str(key.dealer_up)
    return [
        LINE,
        f" 상태: {row.label} vs 딜러 {딜러이름}   (표 {row.kind} 구역)",
        f" StateKey{tuple(key)}",
        f" 규칙 지문 {rules.fingerprint()}"
        f"  |  {'더블 가능' if key.can_double else '더블 불가'}"
        f"  |  {'스플릿 가능' if key.can_split else '스플릿 불가'}",
        LINE,
    ]


def _행동표(key: StateKey, dp_q: np.ndarray,
            학습_q: np.ndarray | None, 방문: np.ndarray | None) -> list[str]:
    mask = legal_actions(key)
    최선 = int(np.nanargmax(dp_q))
    줄들 = ["", " 행동          DP 정확 EV       학습 Q       방문수 N",
            " " + "-" * 56]
    for a in ACTIONS:
        이름 = f"{ACTION_NAMES[a]}({ACTION_NAMES_KO[a]})"
        if not bool(mask[a]):
            줄들.append(f" {이름:<12} {'불법':>12} {'-':>12} {'-':>12}")
            continue
        dp_글자 = f"{float(dp_q[a]):+.4f}"
        if 학습_q is None:
            q_글자, n_글자 = "-", "-"
        else:
            q_글자 = f"{float(학습_q[a]):+.4f}"
            n_글자 = f"{int(방문[a]):,}"
        꼬리 = "   <- DP 최선" if a == 최선 else ""
        줄들.append(f" {이름:<12} {dp_글자:>12} {q_글자:>12} {n_글자:>12}{꼬리}")
    return 줄들


def explain(key: StateKey, rules: RuleSet, run_path: Path | None = None) -> str:
    """상태 한 칸에 대한 설명 화면 전체를 문자열로 만든다."""
    row = find_row(key.total, key.is_soft, key.can_split)
    dp_q = action_values(key, rules)

    if run_path is None:
        학습_q = None
        방문 = None
        우리표 = optimal_chart(rules)
        표출처 = "DP 최적표"
    else:
        q_table, n_table = load_q_and_n(run_path)
        학습_q = q_table[key]
        방문 = n_table[key]
        우리표 = project(q_table, rules)
        표출처 = run_path.name

    줄들 = _머리말(key, row, rules)
    줄들 += _행동표(key, dp_q, 학습_q, 방문)

    순위 = np.sort(dp_q[~np.isnan(dp_q)])[::-1]
    갭 = float(순위[0] - 순위[1])
    꼬리 = "   -> 통계적 동점 구간" if 갭 < TIE_DELTA else ""
    줄들 += ["", f" DP 갭(최선 - 차선) {갭:.6f}{꼬리}"]

    if 학습_q is None:
        줄들.append(" 학습 스냅샷 없음 (--run 으로 npz를 주면 학습 Q와 방문수가 채워진다)")

    ref = load_reference_chart()
    i = ROW_INDEX[row.label]
    j = DEALER_COLS.index(key.dealer_up)
    기준글자 = str(ref.notation[i, j])
    우리글자 = str(우리표.notation[i, j]) or "(미결정)"

    줄들 += [
        "",
        f" 우리 표기 ({표출처}): {우리글자}",
        f" 기준표(출판 4덱 S17 DAS) 정답: {기준글자}",
    ]
    if 우리글자 != 기준글자:
        줄들.append(" ※ 기준표와 다르다. data/expected_mismatch.json의 사전등록 목록을 보라.")
    줄들.append(LINE)
    return "\n".join(줄들)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="상태 한 칸을 한 화면에 설명한다")
    p.add_argument("--total", type=int, required=True, help="플레이어 합계 4~21")
    p.add_argument("--dealer", type=int, required=True, help="딜러 업카드 2~10 또는 11(A)")
    p.add_argument("--soft", default="false", help="에이스를 11로 세는 중인가")
    p.add_argument("--pair", default="false", help="같은 값 두 장인가(스플릿 가능한가)")
    p.add_argument("--no-double", action="store_true",
                   help="첫 두 장이 아닌 상태로 본다(더블 불가)")
    p.add_argument("--run", default=None, help="학습 스냅샷 npz 경로(없으면 DP만)")
    args = p.parse_args(argv)

    key = StateKey(
        total=int(args.total),
        is_soft=parse_bool(args.soft),
        dealer_up=dealer_index(int(args.dealer)),
        can_double=0 if args.no_double else 1,
        can_split=parse_bool(args.pair),
        split_depth=0,
    )
    run_path = Path(args.run) if args.run else None
    print(explain(key, RULES_V1, run_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
