"""이 파일은 블랙잭의 한 손(Hand)을 표현하고 합계·소프트 여부·상태 키를 계산한다.
입력: 카드값 리스트(1=에이스, 10=10/J/Q/K), RuleSet, 딜러 업카드, 라운드 내 손 개수.
출력: Hand 데이터클래스와 그 속성(total, is_soft, is_pair, is_blackjack, state_key).
"""
from __future__ import annotations

from dataclasses import dataclass

from blackjack_rl.rules import RuleSet
from blackjack_rl.state import StateKey

ACE = 1


def dealer_up_index(dealer_up: int) -> int:
    """딜러 업카드를 StateKey용 2~11 인덱스로 바꾼다(에이스는 11)."""
    # 왜: 카드 표현은 1=A인데 StateKey의 dealer_up은 2~11이라 접점이 한 곳은 필요하다.
    #     그 한 곳을 여기로 고정해 두면 env/DP/차트가 전부 같은 규약을 쓴다.
    if dealer_up == 1 or dealer_up == 11:
        return 11
    if 2 <= dealer_up <= 10:
        return dealer_up
    raise ValueError(f"딜러 업카드가 범위 밖입니다: {dealer_up}")


@dataclass
class Hand:
    """플레이어 또는 딜러가 들고 있는 카드 한 벌."""

    cards: list[int]
    bet: float = 1.0
    split_depth: int = 0
    from_split: bool = False
    from_split_ace: bool = False
    parent: int | None = None
    doubled: bool = False
    done: bool = False

    @property
    def total(self) -> int:
        """소프트 에이스를 반영한 최선의 합."""
        base = sum(self.cards)
        # 왜: 에이스 두 장을 동시에 11로 쓰면 최소 22라 언제나 버스트다.
        #     그래서 '에이스 한 장만 11로 올려 본다'로 항상 충분하다.
        if ACE in self.cards and base + 10 <= 21:
            return base + 10
        return base

    @property
    def is_soft(self) -> bool:
        """에이스를 11로 쓰고 있으면 소프트다."""
        return self.total > sum(self.cards)

    @property
    def is_bust(self) -> bool:
        return self.total > 21

    @property
    def is_pair(self) -> bool:
        # 왜: 카드값이 이미 1~10으로 들어오므로 10/J/Q/K는 전부 10이다.
        #     따라서 값 비교 한 번으로 '10과 K도 페어' 규칙이 자동으로 지켜진다.
        return len(self.cards) == 2 and self.cards[0] == self.cards[1]

    @property
    def is_blackjack(self) -> bool:
        # 왜: 스플릿으로 만든 21은 블랙잭이 아니다(3:2가 아니라 1:1).
        return len(self.cards) == 2 and self.total == 21 and not self.from_split

    def can_double(self, rules: RuleSet) -> bool:
        """이 손에서 더블다운이 가능한가."""
        if len(self.cards) != 2 or self.doubled:
            return False
        if self.from_split and not rules.double_after_split:
            return False
        if self.from_split_ace and rules.split_aces_one_card:
            # 왜: 에이스를 스플릿한 손은 카드를 한 장만 받고 즉시 끝나므로
            #     더블다운 자체가 존재할 수 없다.
            return False
        if not rules.double_any_two:
            # 왜: DA2가 꺼진 규칙(유럽식)에서는 하드 9/10/11에서만 더블할 수 있다.
            return (not self.is_soft) and self.total in (9, 10, 11)
        return True

    def can_split(self, rules: RuleSet, hands_in_round: int) -> bool:
        """이 손에서 스플릿이 가능한가."""
        if not self.is_pair:
            return False
        if self.split_depth >= rules.max_split_depth:
            return False
        if hands_in_round >= rules.max_hands_per_round:
            return False
        if self.from_split_ace and not rules.resplit_aces:
            return False
        return True

    def state_key(self, dealer_up: int, rules: RuleSet, hands_in_round: int) -> StateKey:
        """이 손의 결정 상태 키를 만든다. dealer_up은 카드값(1=A) 또는 11을 받는다."""
        total = self.total
        if total > 21:
            raise ValueError(f"버스트한 손에는 상태가 없습니다: {self.cards}")
        # 왜: 기본 모드는 split_depth를 상태에서 뺀다(어블레이션 플래그로만 켠다).
        if rules.include_split_depth_in_state:
            depth = self.split_depth
        else:
            depth = 0
        return StateKey(
            total=total,
            is_soft=1 if self.is_soft else 0,
            dealer_up=dealer_up_index(dealer_up),
            can_double=1 if self.can_double(rules) else 0,
            can_split=1 if self.can_split(rules, hands_in_round) else 0,
            split_depth=depth,
        )
