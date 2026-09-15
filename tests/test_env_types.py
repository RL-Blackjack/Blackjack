"""이 파일은 env.py의 자료구조가 설계서 §2.7대로 생겼는지 검사한다.
입력: blackjack_rl.env의 Ctx, ActionFn, HandRecord, RoundResult.
출력: pytest 통과/실패."""

import numpy as np

from blackjack_rl.env import ActionFn, Ctx, HandRecord, RoundResult
from blackjack_rl.state import HIT, STAND, StateKey


def test_Ctx는_필드_다섯_개짜리_네임드튜플이다():
    ctx = Ctx(true_count=1.5, decks_left=2.0, hand_index=1, n_hands=2, bet=1.0)
    assert ctx.true_count == 1.5
    assert ctx.decks_left == 2.0
    assert ctx.hand_index == 1
    assert ctx.n_hands == 2
    assert ctx.bet == 1.0
    # 네임드튜플이므로 길이 5이고 순서대로 언팩된다
    assert len(ctx) == 5
    tc, decks, idx, n, bet = ctx
    assert (tc, decks, idx, n, bet) == (1.5, 2.0, 1, 2, 1.0)


def test_ActionFn_모양의_함수를_그대로_호출할_수_있다():
    key = StateKey(16, 0, 10, 1, 0, 0)
    mask = np.array([True, True, False, False])
    ctx = Ctx(0.0, 6.0, 0, 1, 1.0)

    def 언제나_스탠드(key, mask, ctx):
        return STAND

    act: ActionFn = 언제나_스탠드
    assert act(key, mask, ctx) == STAND


def test_HandRecord_기본값():
    rec = HandRecord()
    assert rec.trajectory == []
    assert rec.parent is None
    assert rec.parent_step is None
    assert rec.bet == 1.0
    assert rec.result == 0.0
    assert rec.subtree_result == 0.0


def test_HandRecord는_손마다_따로_리스트를_갖는다():
    # 왜: 기본값을 mutable로 공유하면 모든 손이 같은 궤적을 쓰게 되는 고전 버그가 난다
    a = HandRecord()
    b = HandRecord()
    a.trajectory.append((StateKey(12, 0, 6, 1, 0, 0), HIT))
    assert len(a.trajectory) == 1
    assert len(b.trajectory) == 0


def test_HandRecord_필드를_직접_채울_수_있다():
    key = StateKey(13, 0, 6, 1, 0, 0)
    rec = HandRecord(
        trajectory=[(key, STAND)],
        parent=0,
        parent_step=0,
        bet=2.0,
        result=-2.0,
        subtree_result=-2.0,
    )
    assert rec.trajectory[0] == (key, STAND)
    assert rec.parent == 0
    assert rec.parent_step == 0
    assert rec.bet == 2.0
    assert rec.result == -2.0
    assert rec.subtree_result == -2.0


def test_RoundResult_기본값과_직접_채우기():
    empty = RoundResult()
    assert empty.hands == []
    assert empty.dealer_up == 0
    assert empty.dealer_cards == []
    assert empty.net == 0.0
    assert empty.n_decisions == 0
    assert empty.tc_at_deal == 0.0

    r = RoundResult(
        hands=[HandRecord(result=1.5)],
        dealer_up=10,
        dealer_cards=[10, 8],
        net=1.5,
        n_decisions=0,
        tc_at_deal=-0.5,
    )
    assert len(r.hands) == 1
    assert r.hands[0].result == 1.5
    assert r.dealer_up == 10
    assert r.dealer_cards == [10, 8]
    assert r.net == 1.5
    assert r.tc_at_deal == -0.5
