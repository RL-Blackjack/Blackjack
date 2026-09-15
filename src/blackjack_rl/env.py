"""이 파일은 블랙잭 한 라운드의 기록 자료구조를 정의한다.
입력: 없음(타입 정의 전용 모듈).
출력: Ctx, ActionFn, HandRecord, RoundResult."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from blackjack_rl.cards import Shoe
from blackjack_rl.dealer import dealer_has_blackjack, play_dealer
from blackjack_rl.hand import Hand
from blackjack_rl.rng import Streams
from blackjack_rl.rules import RuleSet
from blackjack_rl.starts import StartState
from blackjack_rl.state import DOUBLE, HIT, SPLIT, STAND, StateKey, legal_actions


class Ctx(NamedTuple):
    """상태가 아니라 '컨텍스트'다. 기본 플레이 에이전트는 전부 무시하고,
    카운팅·베팅 에이전트만 true_count와 decks_left를 읽는다."""

    true_count: float
    decks_left: float
    hand_index: int
    n_hands: int
    bet: float


# 왜: Ctx를 1주차부터 시그니처에 박아두면 9주차에 카운팅이 들어와도 행동 함수가 안 바뀌고,
#     에이전트가 슈를 클로저로 붙잡지 않으므로 평가할 때 ReplayShoe(CRN)로 갈아끼울 수 있다.
ActionFn = Callable[[StateKey, np.ndarray, Ctx], int]


@dataclass
class HandRecord:
    """손 하나의 기록. 라운드를 트리로 들고 다니지 않고 평탄한 리스트로 두되,
    parent 인덱스로 스플릿 관계만 남긴다."""

    # 왜: 리스트 기본값은 default_factory로 줘야 손마다 각자의 리스트를 갖는다.
    trajectory: list[tuple[StateKey, int]] = field(default_factory=list)
    parent: int | None = None
    parent_step: int | None = None
    bet: float = 1.0
    result: float = 0.0
    subtree_result: float = 0.0


@dataclass
class RoundResult:
    """라운드 하나의 전체 기록. 환경은 판단하지 않고 이것만 만든다."""

    hands: list[HandRecord] = field(default_factory=list)
    dealer_up: int = 0
    dealer_cards: list[int] = field(default_factory=list)
    net: float = 0.0
    n_decisions: int = 0
    tc_at_deal: float = 0.0


def _dealer_up_key(card: int) -> int:
    """카드값 규약(1=A)을 상태 키 규약(11=A)으로 바꾼다."""
    # 왜: 설계서 §3.1에서 StateKey.dealer_up을 2..11로 못박았기 때문이다.
    if card == 1:
        return 11
    return card


def _is_split_node(rec: HandRecord) -> bool:
    """마지막 결정이 SPLIT이면 이 손은 두 자식으로 갈라진 분기 노드다(자기 결과 없음)."""
    if len(rec.trajectory) == 0:
        return False
    return rec.trajectory[-1][1] == SPLIT


def _accumulate_subtree(records: list[HandRecord]) -> float:
    """뒤에서부터 subtree_result를 누적하고 라운드 순손익을 돌려준다."""
    for rec in records:
        rec.subtree_result = 0.0

    net = 0.0
    # 왜: 자식은 언제나 부모보다 뒤 인덱스에 붙으므로, 뒤에서부터 한 번만 훑으면
    #     자식의 합이 부모에 모두 모인 뒤에 부모를 처리하게 된다(재귀 불필요).
    for i in range(len(records) - 1, -1, -1):
        rec = records[i]
        rec.subtree_result = rec.result + rec.subtree_result
        net += rec.result
        if rec.parent is not None:
            records[rec.parent].subtree_result += rec.subtree_result
    return net


class BlackjackEnv:
    """라운드를 끝까지 주도하고 기록만 한다. 판단은 전부 act가 한다."""

    def __init__(self, rules: RuleSet, shoe: Shoe, streams: Streams) -> None:
        self.rules = rules
        self.shoe = shoe
        self.streams = streams

    def play_round(
        self,
        act: ActionFn,
        start: StartState | None = None,
        bet: float = 1.0,
    ) -> RoundResult:
        rules = self.rules
        shoe = self.shoe
        shoe.maybe_shuffle()

        # 1) 딜
        if start is None:
            player_cards = [shoe.draw(), shoe.draw()]
            up_card = shoe.draw()
            forced_first_action = None
        else:
            player_cards = list(start.player_cards)
            up_card = start.dealer_up
            forced_first_action = start.forced_first_action
        hole_card = shoe.draw()
        up_key = _dealer_up_key(up_card)

        hands = [Hand(cards=player_cards, bet=bet)]
        records = [HandRecord(bet=bet)]
        result = RoundResult(
            hands=records,
            dealer_up=up_key,
            dealer_cards=[],
            net=0.0,
            n_decisions=0,
            tc_at_deal=float(shoe.true_count),
        )

        dealer_bj = dealer_has_blackjack(up_card, hole_card)
        player_bj = hands[0].is_blackjack

        # 2) 피크와 내추럴 — 결정이 하나도 없이 끝나는 경우
        # 왜: dealer_peeks=False(유럽식)면 딜러 BJ를 아직 모르므로 플레이를 계속한다.
        #     그 경우 딜러 BJ는 정산에서 드러나고 더블·스플릿한 금액까지 전부 잃는다.
        if (rules.dealer_peeks and dealer_bj) or player_bj:
            hands[0].done = True
            dealer_cards = [up_card, hole_card]
        else:
            self._play_hands(act, hands, records, result, up_key, forced_first_action)
            any_live_hand = False
            for i, h in enumerate(hands):
                if _is_split_node(records[i]):
                    continue
                if h.total <= 21:
                    any_live_hand = True
            dealer_cards = play_dealer(shoe, up_card, hole_card, rules, any_live_hand)

        # 3) 정산
        result.dealer_cards = list(dealer_cards)
        dealer_total = Hand(cards=list(dealer_cards)).total
        self._settle(hands, records, dealer_total, dealer_bj)
        result.net = _accumulate_subtree(records)
        return result

    def _play_hands(
        self,
        act: ActionFn,
        hands: list[Hand],
        records: list[HandRecord],
        result: RoundResult,
        up_key: int,
        forced_first_action: int | None,
    ) -> None:
        rules = self.rules
        shoe = self.shoe
        i = 0
        n_live = 1  # 실제로 딜러와 승부할 손의 수(분기 노드는 세지 않는다)

        while i < len(hands):
            h = hands[i]
            rec = records[i]

            if h.done:
                i += 1
                continue

            # 스플릿으로 생긴 손은 카드가 한 장이므로 여기서 두 번째 장을 받는다
            if len(h.cards) < 2:
                h.cards.append(shoe.draw())
                if h.from_split_ace and rules.split_aces_one_card:
                    h.done = True  # 에이스 스플릿은 한 장만 받고 즉시 종단
                    continue

            if h.total > 21:
                h.done = True
                continue

            key = h.state_key(up_key, rules, n_live)
            mask = legal_actions(key)
            ctx = Ctx(
                true_count=float(shoe.true_count),
                decks_left=float(shoe.decks_remaining),
                hand_index=i,
                n_hands=n_live,
                bet=h.bet,
            )

            if forced_first_action is not None and result.n_decisions == 0:
                action = int(forced_first_action)
            else:
                action = int(act(key, mask, ctx))

            if action < 0 or action > 3 or not mask[action]:
                raise ValueError(f"불법 행동이다: key={key}, action={action}")

            rec.trajectory.append((key, action))
            result.n_decisions += 1

            if action == STAND:
                h.done = True
            elif action == HIT:
                h.cards.append(shoe.draw())
                if h.total > 21:
                    h.done = True
            elif action == DOUBLE:
                h.bet = h.bet * 2.0
                h.doubled = True
                h.cards.append(shoe.draw())
                h.done = True
            else:  # SPLIT
                first_card = h.cards[0]
                second_card = h.cards[1]
                is_ace_split = first_card == 1
                h.done = True
                step = len(rec.trajectory) - 1
                # 왜: 자식을 리스트 중간에 끼워 넣으면 이미 기록된 parent 인덱스가 어긋난다.
                #     항상 끝에 덧붙이고, 진행 순서는 인덱스 순서를 그대로 따른다.
                for card in (first_card, second_card):
                    hands.append(
                        Hand(
                            cards=[card],
                            bet=h.bet,
                            split_depth=h.split_depth + 1,
                            from_split=True,
                            from_split_ace=is_ace_split,
                            parent=i,
                        )
                    )
                    records.append(HandRecord(parent=i, parent_step=step, bet=h.bet))
                n_live += 1  # 손 하나가 둘로 갈라지므로 살아있는 손은 1개 늘어난다

    def _settle(
        self,
        hands: list[Hand],
        records: list[HandRecord],
        dealer_total: int,
        dealer_bj: bool,
    ) -> None:
        rules = self.rules
        for i, h in enumerate(hands):
            rec = records[i]
            rec.bet = h.bet

            if _is_split_node(rec):
                rec.result = 0.0        # 분기 노드는 자기 결과가 없다
            elif dealer_bj:
                rec.result = 0.0 if h.is_blackjack else -h.bet
            elif h.is_blackjack:
                rec.result = rules.blackjack_payout * h.bet
            elif h.total > 21:
                rec.result = -h.bet
            elif dealer_total > 21 or h.total > dealer_total:
                rec.result = h.bet
            elif h.total < dealer_total:
                rec.result = -h.bet
            else:
                rec.result = 0.0
