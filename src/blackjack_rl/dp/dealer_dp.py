"""이 파일은 무한덱에서 딜러 최종 결과의 확률분포를 해석적으로(시뮬 없이) 계산한다.
입력: 딜러 업카드(1=에이스 ~ 10), RuleSet, 피크 여부 peeked.
출력: 길이 6의 확률 배열 [17,18,19,20,21,bust](합 1.0)과 버스트 확률 하나.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from blackjack_rl.rules import RuleSet

# 왜: 무한덱에서는 10/J/Q/K가 전부 10으로 묶이므로 10만 4/13이고 나머지는 1/13이다.
#     인덱스 0은 쓰지 않는다(카드값 1~10을 그대로 인덱스로 쓰기 위한 자리).
CARD_PROB = (
    0.0,
    1.0 / 13.0, 1.0 / 13.0, 1.0 / 13.0, 1.0 / 13.0, 1.0 / 13.0,
    1.0 / 13.0, 1.0 / 13.0, 1.0 / 13.0, 1.0 / 13.0, 4.0 / 13.0,
)

OUTCOME_LABELS = ("17", "18", "19", "20", "21", "bust")
BUST_INDEX = 5


def _add_card(total: int, soft: bool, card: int) -> tuple[int, bool]:
    """현재 (합, 소프트여부)에 카드 한 장을 더한 뒤의 (합, 소프트여부)."""
    new_total = total + card
    new_soft = soft
    if card == 1 and new_total + 10 <= 21:
        # 왜: 새로 받은 에이스를 11로 올려도 21을 넘지 않으면 그게 항상 이득이다.
        new_total += 10
        new_soft = True
    if new_total > 21 and new_soft:
        # 왜: 버스트 위기가 오면 11로 쓰던 에이스를 1로 되돌린다(에이스는 최대 한 장만 11).
        new_total -= 10
        new_soft = False
    return new_total, new_soft


def _dealer_stands(total: int, soft: bool, rules: RuleSet) -> bool:
    """딜러가 여기서 멈추는가. (dealer.py와 독립으로 다시 쓴 규칙 판정)"""
    if total < 17:
        return False
    if total == 17 and soft and rules.dealer_hits_soft_17:
        return False
    return True


def _dist_from(total: int, soft: bool, rules: RuleSet, memo: dict) -> np.ndarray:
    """(합, 소프트여부)에서 출발했을 때 딜러 최종 결과의 확률분포."""
    key = (total, soft)
    if key in memo:
        return memo[key]

    dist = np.zeros(6, dtype=np.float64)
    if total > 21:
        dist[BUST_INDEX] = 1.0
    elif _dealer_stands(total, soft, rules):
        dist[total - 17] = 1.0
    else:
        for card in range(1, 11):
            next_total, next_soft = _add_card(total, soft, card)
            dist += CARD_PROB[card] * _dist_from(next_total, next_soft, rules, memo)

    memo[key] = dist
    return dist


def _is_natural(up: int, hole: int) -> bool:
    """업카드와 홀카드가 블랙잭(에이스+10)인가."""
    return (up == 1 and hole == 10) or (up == 10 and hole == 1)


@lru_cache(maxsize=None)
def dealer_outcome_dist(up: int, rules: RuleSet, *, peeked: bool) -> np.ndarray:
    """업카드 up에서 딜러 최종 결과의 확률 [17,18,19,20,21,bust].

    peeked=True면 '딜러 BJ 아님'으로 조건화한 홀카드 분포 위에서 재정규화한다.
    up은 카드값 1~10이다(1=에이스). StateKey의 2~11이 아니다.
    """
    if not 1 <= up <= 10:
        raise ValueError(f"업카드는 1(A)~10이어야 합니다: {up}")

    memo: dict = {}
    dist = np.zeros(6, dtype=np.float64)
    weight = 0.0
    for hole in range(1, 11):
        if peeked and _is_natural(up, hole):
            # 왜: 피크로 '딜러 BJ 아님'이 확인된 상황이므로 BJ가 되는 홀카드를 제외한다.
            continue
        prob = CARD_PROB[hole]
        weight += prob
        total, soft = _add_card(0, False, up)
        total, soft = _add_card(total, soft, hole)
        dist = dist + prob * _dist_from(total, soft, rules, memo)

    # ★ 재정규화. 이 한 줄을 빼면 up=A는 합이 9/13, up=10은 12/13이 되어
    #   기본전략표에서 가장 논쟁적인 두 열(딜러 A, 딜러 10)이 조용히 전부 어긋난다.
    dist = dist / weight

    # 왜: lru_cache가 같은 배열 객체를 계속 돌려주므로, 호출자가 실수로 덮어쓰면
    #     이후 모든 계산이 오염된다. 아예 쓰기 금지로 잠가 둔다.
    dist.flags.writeable = False
    return dist


def dealer_bust_prob(up: int, rules: RuleSet) -> float:
    """업카드 up에서 딜러가 버스트할 확률(피크 이전, 홀카드 정보 없음 기준)."""
    return float(dealer_outcome_dist(up, rules, peeked=False)[BUST_INDEX])
