"""이 파일은 한 라운드를 어느 상태에서 시작할지 고른다.
입력: 난수 생성기
출력: StartState(지정 시작) 또는 None(자연 딜), 그리고 탐험적 시작 샘플러
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np

from blackjack_rl.hand import Hand
from blackjack_rl.rules import RULES_V1, RuleSet
from blackjack_rl.state import REACHABLE_KEYS, StateKey, legal_actions


@dataclass
class StartState:
    """라운드를 특정 상황에서 시작시키기 위한 지정값.

    player_cards: 플레이어에게 강제로 줄 카드(값 1~10, A=1)
    dealer_up: 딜러 업카드(값 1~10). 홀카드는 환경이 무작위로 뽑는다.
    forced_first_action: 첫 결정을 이 행동으로 강제한다(탐험적 시작·셀 검증용).
    """

    player_cards: list[int]
    dealer_up: int
    forced_first_action: int | None = None


class StartSampler(Protocol):
    """시작 상태 샘플러의 공통 모양."""

    def sample(self, rng: np.random.Generator) -> StartState | None: ...


class NaturalDeal:
    """자연 딜 — 실제 게임과 같은 분포로 시작한다.

    # 왜 None을 돌려주는가: 환경이 이미 올바른 딜 절차를 갖고 있으므로,
    #   '지정하지 않는다'를 None 하나로 표현하면 환경에 분기가 하나만 생긴다.
    """

    def sample(self, rng: np.random.Generator) -> StartState | None:
        return None


class StartPair(NamedTuple):
    """탐험적 시작 하나. '이 카드로 시작해서 이 행동을 강제하면 이 키에 도달한다'는 뜻이다."""

    key: StateKey
    cards: tuple[int, ...]     # 플레이어에게 강제로 줄 카드(값 1~10)
    dealer_up_card: int        # 카드값 규약(1=A). StateKey의 11이 아니다.
    action: int


def cards_for_key(key: StateKey) -> list[int] | None:
    """상태 키 하나를 만들어 낼 플레이어 카드를 복원한다. 만들 수 없으면 None."""
    total = key.total
    is_soft = key.is_soft

    if key.can_split == 1:
        # 왜: 같은 랭크 두 장의 합은 하드 2r이고 유일한 소프트 페어는 (A,A)=소프트 12이므로
        #     (total, is_soft)에서 페어의 랭크가 유일하게 복원된다. state.py의 같은 논리다.
        rank = 1 if is_soft == 1 else total // 2
        cards = [rank, rank]
    elif key.can_double == 1:
        # can_double=1은 '첫 두 장'이라는 뜻이므로 반드시 카드가 두 장이어야 한다.
        if is_soft == 1:
            cards = [1, total - 11]
        elif total <= 12:
            cards = [2, total - 2]
        else:
            cards = [10, total - 10]
    elif is_soft == 1:
        # can_double=0은 '이미 한 장 더 받았다'는 뜻이므로 세 장으로 만든다.
        cards = [1, 1, total - 12]
    elif total <= 14:
        cards = [2, 2, total - 4]
    else:
        cards = [2, 10, total - 12]

    for card in cards:
        if card < 1 or card > 10:
            return None
    return cards


def _makes_this_key(cards: list[int], key: StateKey, rules: RuleSet) -> bool:
    """이 카드로 시작하면 정말 이 키에서 첫 결정을 하게 되는가."""
    hand = Hand(cards=list(cards))
    if hand.is_blackjack:
        # 왜: 내추럴 블랙잭이면 환경이 결정을 한 번도 묻지 않고 라운드를 끝낸다.
        #     그런 시작을 쌍 목록에 넣으면 그 칸은 영원히 표본 0개로 남는다.
        return False
    # 왜: 직접 조립한 키를 믿지 않고 Hand.state_key에게 되묻는다. 환경이 쓰는 그 함수다.
    #     복원 규칙이 틀리면 여기서 즉시 걸러지므로 학습이 엉뚱한 칸을 갱신할 수 없다.
    return hand.state_key(key.dealer_up, rules, 1) == key


def build_start_pairs(
    rules: RuleSet,
) -> tuple[tuple[StartPair, ...], tuple[StateKey, ...]]:
    """도달 가능한 모든 (상태, 합법 행동) 쌍과, 시작으로는 만들 수 없는 키를 돌려준다."""
    pairs: list[StartPair] = []
    uncovered: list[StateKey] = []

    for key in REACHABLE_KEYS:
        cards = cards_for_key(key)
        if cards is None or not _makes_this_key(cards, key, rules):
            # 왜: 하드 4(2,2) / 하드 20(10,10) / 소프트 21(A,10) 세 모양은 두 장으로 만들면
            #     각각 페어가 되거나 블랙잭이 된다. 실제 게임에서는 스플릿 깊이 한도에
            #     걸린 페어나 스플릿 자식으로만 생기므로 '시작 상태'로는 만들 수 없다.
            #     숨기지 않고 목록으로 내보내서 평가 쪽이 미방문 칸으로 처리하게 한다.
            uncovered.append(key)
            continue

        # 왜: StateKey는 에이스를 11로 쓰지만 환경과 딜러는 카드값(1=A)을 받는다.
        #     11을 그대로 넘기면 딜러 손 합계가 11로 계산돼 블랙잭 판정이 틀어진다.
        up_card = 1 if key.dealer_up == 11 else key.dealer_up
        mask = legal_actions(key)
        for action in range(4):
            if mask[action]:
                pairs.append(StartPair(key, tuple(cards), up_card, action))

    return tuple(pairs), tuple(uncovered)


class ExploringStarts:
    """탐험적 시작 — 도달 가능한 (상태, 행동) 쌍을 균등하게 뽑아 첫 행동까지 강제한다.

    # 왜 필요한가: ε-greedy만으로는 '자연 딜에서 거의 안 나오는 칸'의 표본이 수십 배
    #   적어서, 표 구석이 영원히 노이즈로 남는다. 모든 (s,a)를 같은 횟수로 강제 방문시키면
    #   모든 칸이 같은 속도로 수렴한다. 대신 시작 분포가 실제 게임과 다르므로
    #   '학습용'이고, 평가는 반드시 자연 딜로 따로 한다.
    """

    def __init__(self, rules: RuleSet = RULES_V1) -> None:
        self.rules = rules
        self.pairs, self.uncovered_keys = build_start_pairs(rules)
        # 왜: 정렬해서 고정한다. 순서가 실행마다 달라지면 보고서 표가 매번 달라진다.
        self.covered_keys = tuple(sorted({pair.key for pair in self.pairs}))

    @property
    def n_pairs(self) -> int:
        """균등하게 뽑는 (상태, 행동) 쌍의 개수."""
        return len(self.pairs)

    def sample(self, rng: np.random.Generator) -> StartState:
        """쌍 하나를 균등하게 뽑아 환경이 먹을 StartState로 바꾼다."""
        i = int(rng.integers(0, len(self.pairs)))
        pair = self.pairs[i]
        return StartState(
            player_cards=list(pair.cards),
            dealer_up=pair.dealer_up_card,
            forced_first_action=pair.action,
        )
