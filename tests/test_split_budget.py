"""이 파일은 스플릿 상한과 에이스 스플릿 규칙이 실제로 걸리는지 검사한다.
입력: 미리 정한 카드를 순서대로 내주는 대본 슈와 BlackjackEnv.
출력: pytest 통과/실패."""

from blackjack_rl.env import BlackjackEnv
from blackjack_rl.rng import make_streams
from blackjack_rl.rules import RULES_V1
from blackjack_rl.state import SPLIT, STAND


class ScriptedShoe:
    """테스트 전용 슈. 미리 정한 카드를 순서대로 내준다."""

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


def 쪼갤_수_있으면_쪼개기(key, mask, ctx):
    if mask[SPLIT]:
        return SPLIT
    return STAND


def leaf_indices(r):
    """자식이 없는 손(실제로 딜러와 승부하는 손)의 인덱스."""
    parents = set()
    for rec in r.hands:
        if rec.parent is not None:
            parents.add(rec.parent)
    return [i for i in range(len(r.hands)) if i not in parents]


def test_라운드_손_수_상한_4가_실제로_걸린다():
    # 8이 계속 나와 매번 페어가 되지만, 살아있는 손이 4개가 되는 순간 SPLIT이 불법이 된다.
    env = make_env([8, 8, 7, 10] + [8] * 6)
    r = env.play_round(쪼갤_수_있으면_쪼개기)
    assert len(leaf_indices(r)) == 4
    assert len(r.hands) == 7            # 분기 노드 3 + 잎 4
    assert r.n_decisions == 7           # 스플릿 3번 + 스탠드 4번
    # 딜러 7,10 = 17 스탠드. 잎 네 손은 전부 16이라 전부 패배.
    assert r.net == -4.0


def test_에이스_스플릿은_한_장만_받고_즉시_끝난다():
    # 플레이어 A,A 스플릿 → 두 손이 5를 한 장씩 받고 결정 없이 종단 / 딜러 7,10 = 17
    env = make_env([1, 1, 7, 10, 5, 5])
    호출된_행동 = []

    def act(key, mask, ctx):
        if mask[SPLIT]:
            호출된_행동.append(SPLIT)
            return SPLIT
        호출된_행동.append(STAND)
        return STAND

    r = env.play_round(act)
    assert 호출된_행동 == [SPLIT]       # 자식 손에서는 행동 함수가 아예 안 불린다
    assert r.n_decisions == 1
    assert r.hands[1].trajectory == []
    assert r.hands[2].trajectory == []
    assert r.net == -2.0               # 소프트 16 두 손이 딜러 17에 패배


def test_스플릿_직후_자식_손은_더블을_다시_할_수_있다():
    # DAS=True이므로 8,8을 쪼갠 뒤 8,3 = 11에서 can_double이 1이어야 한다.
    env = make_env([8, 8, 7, 10, 3, 3])
    본_키 = []

    def act(key, mask, ctx):
        본_키.append(key)
        if mask[SPLIT]:
            return SPLIT
        return STAND

    r = env.play_round(act)
    assert 본_키[0].can_split == 1
    assert 본_키[1].total == 11
    assert 본_키[1].can_double == 1     # 스플릿 자식도 더블 가능(DAS)
    assert 본_키[1].can_split == 0      # 8,3은 페어가 아니다
    assert r.net == -2.0
