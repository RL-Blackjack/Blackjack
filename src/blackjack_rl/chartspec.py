"""이 파일은 기본전략표 36행 x 10열 격자의 정의를 한다
입력: 딜러 업카드 숫자와 더블 가능 여부
출력: ChartRow / ChartTable 과 격자 상수, 각 칸에 대응하는 StateKey
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blackjack_rl.rules import RuleSet
from blackjack_rl.state import ACTION_NAMES, DOUBLE, SPLIT, STAND, StateKey

# 딜러 업카드 열. 11은 에이스를 뜻한다.
DEALER_COLS: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)


@dataclass(frozen=True)
class ChartRow:
    """전략표의 한 행. 이 행이 어떤 상태를 가리키는지를 숫자로 들고 있다."""

    kind: str        # "hard" | "soft" | "pair"
    label: str       # "16", "A,7", "8,8"
    total: int       # 이 행의 합계 (A,7은 18, 8,8은 16)
    is_soft: int     # 0 또는 1
    can_split: int   # 0 또는 1 (페어 행만 1)

    def to_key(self, dealer_up: int, *, can_double: bool) -> StateKey:
        """이 행 x 딜러 업카드 칸에 해당하는 StateKey를 만든다."""
        # 왜: 표의 한 칸은 상태 하나가 아니라 '더블 가능/불가능' 두 상태를 겹쳐 보여준다.
        #     그래서 can_double을 키워드 인자로 강제해 호출부에서 실수로 섞이지 않게 한다.
        return StateKey(
            total=self.total,
            is_soft=self.is_soft,
            dealer_up=int(dealer_up),
            can_double=1 if can_double else 0,
            can_split=self.can_split,
            split_depth=0,
        )


def _build_rows() -> tuple[ChartRow, ...]:
    """하드 17행 + 소프트 9행 + 페어 10행 = 36행을 순서대로 만든다."""
    rows: list[ChartRow] = []

    # 하드 5~21 (17행).
    # 왜: 하드 4는 반드시 2,2 페어라서 하드 구역이 아니라 페어 구역에만 둔다.
    for total in range(5, 22):
        rows.append(ChartRow("hard", str(total), total, 0, 0))

    # 소프트 A,2 ~ A,10 (9행). A,2는 합계 13, A,10은 합계 21이다.
    for other in range(2, 11):
        rows.append(ChartRow("soft", f"A,{other}", 11 + other, 1, 0))

    # 페어 A,A + 2,2 ~ 10,10 (10행). A,A만 소프트 12다.
    rows.append(ChartRow("pair", "A,A", 12, 1, 1))
    for rank in range(2, 11):
        rows.append(ChartRow("pair", f"{rank},{rank}", rank * 2, 0, 1))

    return tuple(rows)


CHART_ROWS: tuple[ChartRow, ...] = _build_rows()
ROW_INDEX: dict[str, int] = {row.label: i for i, row in enumerate(CHART_ROWS)}


@dataclass
class ChartTable:
    """36x10 전략표 한 장. action은 행동 번호, notation은 화면에 찍을 글자다."""

    action: np.ndarray     # int8[36,10], -1은 미결정(방문 0회)
    notation: np.ndarray   # '<U2'[36,10], ""는 미결정
    rules_fp: str          # 이 표를 만든 RuleSet의 지문


def _argmax_q(Q: np.ndarray, key: StateKey) -> int | None:
    """Q 배열에서 이 키의 합법 행동 중 최선의 행동 번호를 돌려준다. 전부 nan이면 None."""
    q = Q[key.total, key.is_soft, key.dealer_up,
          key.can_double, key.can_split, key.split_depth]
    if np.all(np.isnan(q)):
        return None
    return int(np.nanargmax(q))


def project(Q: np.ndarray, rules: RuleSet, *, visits: np.ndarray | None = None) -> ChartTable:
    """Q 배열을 36x10 전략표로 사영하고 S/H/D/Ds/Y/N 표기를 만든다."""
    action = np.full((36, 10), -1, dtype=np.int8)
    notation = np.full((36, 10), "", dtype="<U2")

    for i, row in enumerate(CHART_ROWS):
        for j, dealer_up in enumerate(DEALER_COLS):
            if visits is not None and int(visits[i, j]) == 0:
                # 왜: 한 번도 안 가본 칸은 Q가 초기 노이즈뿐이라 표기하면 일치율을 오염시킨다
                continue

            if row.kind == "pair":
                best = _argmax_q(Q, row.to_key(dealer_up, can_double=True))
                if best is None:
                    continue
                action[i, j] = best
                notation[i, j] = "Y" if best == SPLIT else "N"
                continue

            # 하드·소프트 행: 더블이 가능한 상태에서 1순위를 본다
            first = _argmax_q(Q, row.to_key(dealer_up, can_double=True))
            if first is None:
                continue
            action[i, j] = first
            if first != DOUBLE:
                notation[i, j] = ACTION_NAMES[first]
                continue

            # 왜: 'D'와 'Ds'의 차이는 더블이 막혔을 때 무엇을 하느냐다.
            #     그래서 can_double=False로 한 번 더 조회해 2순위를 얻는다.
            second = _argmax_q(Q, row.to_key(dealer_up, can_double=False))
            notation[i, j] = "Ds" if second == STAND else "D"

    return ChartTable(action=action, notation=notation, rules_fp=rules.fingerprint())
