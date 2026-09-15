"""이 파일은 배당 규칙(설계서 §3.3)이 정확한지 검사한다.
입력: 미리 정한 카드를 순서대로 내주는 대본 슈와 BlackjackEnv.
출력: pytest 통과/실패."""

import pytest

from blackjack_rl.env import BlackjackEnv
from blackjack_rl.rng import make_streams
from blackjack_rl.rules import RULES_V1
from blackjack_rl.state import DOUBLE, HIT, SPLIT, STAND


class ScriptedShoe:
    """테스트 전용 슈. 미리 정한 카드를 순서대로 내준다.
    카드를 뽑는 순서: 플레이어1, 플레이어2, 딜러 업카드, 딜러 홀카드, 그 뒤는 플레이 순서."""

    def __init__(self, cards):
        self.cards = list(cards)
        self.i = 0
        self.running_count = 0
        self.decks_remaining = 4.0
        self.true_count = 0.0

    def draw(self):
        card = self.cards[self.i]
        self.i += 1
        return card

    def maybe_shuffle(self):
        return False


def make_env(cards):
    return BlackjackEnv(RULES_V1, ScriptedShoe(cards), make_streams(0))


def never_called(key, mask, ctx):
    raise AssertionError("결정이 필요 없는 라운드인데 행동 함수가 불렸다")


def always(action):
    def act(key, mask, ctx):
        return action

    return act


def test_내추럴_블랙잭은_1_5배를_받는다():
    # 플레이어 A,10 = 21(2장) / 딜러 7,5 = 12
    env = make_env([1, 10, 7, 5])
    r = env.play_round(never_called)
    assert r.net == 1.5
    assert r.n_decisions == 0
    assert r.hands[0].result == 1.5


def test_양쪽_모두_블랙잭이면_푸시():
    # 플레이어 A,10 / 딜러 A,10
    env = make_env([1, 10, 1, 10])
    r = env.play_round(never_called)
    assert r.net == 0.0
    assert r.n_decisions == 0


def test_딜러만_블랙잭이면_피크로_즉시_1단위_패():
    # 플레이어 10,6 = 16 / 딜러 A,10 = 블랙잭
    env = make_env([10, 6, 1, 10])
    r = env.play_round(never_called)
    assert r.net == -1.0
    assert r.n_decisions == 0


def test_더블은_이기면_플러스2():
    # 플레이어 5,6 = 11 → 더블로 10을 받아 21 / 딜러 6,10 = 16 → 10을 받아 26 버스트
    env = make_env([5, 6, 6, 10, 10, 10])
    r = env.play_round(always(DOUBLE))
    assert r.net == 2.0
    assert r.n_decisions == 1
    assert r.hands[0].bet == 2.0


def test_더블은_지면_마이너스2():
    # 플레이어 5,4 = 9 → 더블로 2를 받아 11 / 딜러 10,10 = 20 스탠드
    env = make_env([5, 4, 10, 10, 2])
    r = env.play_round(always(DOUBLE))
    assert r.net == -2.0


def test_같은_수면_푸시_0():
    # 플레이어 10,9 = 19 스탠드 / 딜러 9,10 = 19
    env = make_env([10, 9, 9, 10])
    r = env.play_round(always(STAND))
    assert r.net == 0.0


def test_버스트는_딜러가_카드를_더_뽑지_않아도_마이너스1():
    # 플레이어 10,6 = 16 → 히트로 10을 받아 26 버스트 / 딜러 7,5 = 12 (살아있는 손이 없어 미추첨)
    env = make_env([10, 6, 7, 5, 10])
    r = env.play_round(always(HIT))
    assert r.net == -1.0
    assert r.n_decisions == 1


def test_스플릿으로_만든_21은_블랙잭이_아니라_1배다():
    # 플레이어 A,A 스플릿 → 두 손 모두 10을 받아 21 / 딜러 6,10 = 16 → 10을 받아 26 버스트
    env = make_env([1, 1, 6, 10, 10, 10, 10])

    def act(key, mask, ctx):
        if mask[SPLIT]:
            return SPLIT
        return STAND

    r = env.play_round(act)
    assert len(r.hands) == 3          # 분기 노드 1 + 자식 손 2
    assert r.hands[0].result == 0.0   # 분기 노드는 자기 결과가 없다
    assert r.hands[1].result == 1.0   # 1.5가 아니다
    assert r.hands[2].result == 1.0
    assert r.net == 2.0


def test_손_결과의_합은_언제나_라운드_순손익과_같다():
    env = make_env([1, 1, 6, 10, 10, 10, 10])

    def act(key, mask, ctx):
        if mask[SPLIT]:
            return SPLIT
        return STAND

    r = env.play_round(act)
    합 = 0.0
    for rec in r.hands:
        합 += rec.result
    assert 합 == pytest.approx(r.net)
    assert r.hands[0].subtree_result == pytest.approx(r.net)
