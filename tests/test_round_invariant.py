"""이 파일은 무작위 정책 10만 판으로 라운드 불변식과 EV 구간을 검사한다.
입력: RULES_V1 + 무한덱 슈 + 합법 행동 균등 무작위 정책.
출력: pytest 통과/실패(대략 10~20초 걸린다)."""

import numpy as np

from blackjack_rl.cards import make_shoe
from blackjack_rl.env import BlackjackEnv
from blackjack_rl.rng import make_streams
from blackjack_rl.rules import RULES_V1

N_ROUNDS = 100_000


def make_random_policy(rng):
    def act(key, mask, ctx):
        legal = np.flatnonzero(mask)
        # 왜: rng.choice는 작은 배열에서 느리다. integers로 인덱스만 뽑는다.
        return int(legal[rng.integers(len(legal))])

    return act


def test_무작위_정책_10만판_불변식과_EV():
    rules = RULES_V1
    streams = make_streams(20260915)
    env = BlackjackEnv(rules, make_shoe(rules, streams.deal), streams)
    act = make_random_policy(streams.explore)

    합계 = 0.0
    for _ in range(N_ROUNDS):
        r = env.play_round(act)

        # 불변식 1: 손 결과의 합 == 라운드 순손익
        손합 = 0.0
        for rec in r.hands:
            손합 += rec.result
        assert abs(손합 - r.net) < 1e-9

        # 불변식 2: subtree_result == 자기 결과 + 자식들의 subtree 합
        for i, rec in enumerate(r.hands):
            자식합 = 0.0
            for other in r.hands:
                if other.parent == i:
                    자식합 += other.subtree_result
            assert abs(rec.subtree_result - (rec.result + 자식합)) < 1e-9

        # 불변식 3: 부모 인덱스는 항상 자기보다 앞이다(뒤에서부터 누적이 가능한 근거)
        for i, rec in enumerate(r.hands):
            if rec.parent is not None:
                assert rec.parent < i

        합계 += r.net

    ev = 합계 / N_ROUNDS
    # 왜 이 구간인가: 합법 행동 4개에서 균등 무작위로 고르면 매 2장마다 1/3~1/4 확률로
    # 더블을 치므로 손실이 배로 커진다. 실측 -46% 근처가 정상이고, 이 구간을 벗어나면
    # 배당이나 마스크에 버그가 있다는 신호다.
    #
    # 이 구간은 추정이 아니라 측정값이다(2026-09-16, 이 환경에서 직접 측정):
    #   딜러 모방(17까지 히트)  -5.96% ± 0.22  <- 문헌값 약 -5.5~-6%와 일치
    #   항상 스탠드             -15.54% ± 0.22  <- 문헌값 약 -15~-16%와 일치
    #   무작위(합법 4개 균등)   -46.10% ± 0.30  <- 이 테스트
    #   무작위(히트/스탠드만)   -34.01% ± 0.21
    # 앞의 두 줄이 배당·딜러 플레이·피크 로직이 옳다는 독립 증거다. 설계서 초안은
    # 이 값을 -25~-35%로 추정했는데, 그건 더블·스플릿을 뺀 경우의 값이었다.
    assert -0.50 <= ev <= -0.42, f"무작위 정책 EV가 {ev:.4f}로 예상 구간을 벗어났다"
