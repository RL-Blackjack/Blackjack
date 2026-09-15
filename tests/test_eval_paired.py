"""이 파일은 CRN 쌍대비교와 칸별 자연 발생 빈도를 검사한다.
입력: conftest의 rules·dp 픽스처
출력: 쌍대비교의 분산 절감, 자연 빈도의 해석값 대조
"""

import math

import numpy as np

from blackjack_rl.chartspec import CHART_ROWS, DEALER_COLS
from blackjack_rl.dp.exact import greedy_policy_full
from blackjack_rl.eval.simulate import (EVAL_SEED, make_card_stream,
                                        natural_cell_freq, paired_compare,
                                        round_nets)
from blackjack_rl.state import HIT, REACHABLE_KEYS, STAND

N_PAIRED = 100_000
N_FREQ = 200_000
ROW_OF_LABEL = {행.label: i for i, 행 in enumerate(CHART_ROWS)}


def 하드16_vs_10을_스탠드로_바꾼_정책(policy):
    """하드 16 vs 딜러 10 칸만 히트에서 스탠드로 바꾼 정책을 만든다."""
    바뀐것 = np.asarray(policy, dtype=np.int8).copy()
    for i, key in enumerate(REACHABLE_KEYS):
        if key.total == 16 and key.is_soft == 0 and key.dealer_up == 10:
            if 바뀐것[i] == HIT:
                바뀐것[i] = STAND
    return 바뀐것


def test_같은_정책끼리_쌍대비교하면_차이가_정확히_0이다(rules, dp):
    policy = greedy_policy_full(dp)
    stream = make_card_stream(EVAL_SEED, 20_000, rules)
    결과 = paired_compare(policy, policy, rules, stream)
    # 왜 정확히 0인가: 같은 카드 + 같은 정책이면 라운드 결과가 글자 그대로 같다.
    #   여기가 0이 아니면 ReplayShoe가 라운드마다 초기화되지 않는다는 뜻이다.
    assert 결과.diff == 0.0
    assert 결과.se == 0.0
    assert 결과.divergence_rate == 0.0


def test_CRN_쌍대비교가_독립_스트림보다_표준오차가_훨씬_작다(rules, dp):
    policy = greedy_policy_full(dp)
    다른정책 = 하드16_vs_10을_스탠드로_바꾼_정책(policy)
    stream = make_card_stream(EVAL_SEED, N_PAIRED, rules)
    쌍대 = paired_compare(policy, 다른정책, rules, stream)

    다른스트림 = make_card_stream(EVAL_SEED + 777, N_PAIRED, rules)
    a = round_nets(policy, rules, stream)
    b = round_nets(다른정책, rules, 다른스트림)
    독립_표준오차 = math.sqrt(a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / math.sqrt(N_PAIRED)

    print(f"\nCRN  se = {쌍대.se:.6f} (갈라진 라운드 {쌍대.divergence_rate:.3%})")
    print(f"독립 se = {독립_표준오차:.6f}")
    print(f"분산 절감 = {(독립_표준오차 / 쌍대.se) ** 2:.1f}배")
    # 왜 3배인가: 실측 분산 절감은 49.5배(표준오차 7배)였다. 시드가 바뀌어도
    #   흔들리지 않도록 아주 넉넉하게 3배만 요구한다.
    assert 쌍대.se * 3 < 독립_표준오차
    # 두 정책이 갈리는 라운드는 하드 16 vs 10이 나오는 라운드뿐이라 드물다(실측 1.516%).
    assert 0.005 < 쌍대.divergence_rate < 0.05


def test_칸별_자연빈도가_해석값과_맞는다(rules, dp):
    freq = natural_cell_freq(greedy_policy_full(dp), rules,
                             make_card_stream(EVAL_SEED, N_FREQ, rules))
    assert freq.shape == (36, 10)
    assert float(freq.min()) >= 0.0

    i = ROW_OF_LABEL["10,10"]
    j = DEALER_COLS.index(7)
    해석값 = (4.0 / 13.0) ** 2 * (1.0 / 13.0)
    print(f"\n10,10 vs 7 : 실측 {freq[i, j]:.6f} / 해석 {해석값:.6f}")
    # 왜 이 칸을 고르는가: 10,10은 첫 두 장으로만 만들어지고(DP는 10을 쪼개지 않는다),
    #   딜러 업카드 7은 피크로 라운드가 끝나는 경우도 없어서 해석값이 정확하다.
    #   n=20만 실측 0.007155 vs 해석 0.0072827 (0.67시그마).
    assert abs(float(freq[i, j]) - 해석값) < 0.0006

    # 라운드당 표 안에서 내리는 결정 수는 1보다 조금 작다(BJ로 결정 없이 끝나는 판 때문).
    print(f"칸 결정 합계 = {freq.sum():.5f}")   # 실측 0.95511
    assert 0.93 < float(freq.sum()) < 0.98


def test_결정이_생길_수_없는_행의_빈도는_0이다(rules, dp):
    freq = natural_cell_freq(greedy_policy_full(dp), rules,
                             make_card_stream(EVAL_SEED, 50_000, rules))
    # 왜 0인가: 하드 21은 첫 두 장으로 만들 수 없고(A+10은 소프트 21),
    #   A,10은 내추럴 블랙잭이라 결정 기회가 아예 없으며,
    #   하드 20(can_split=0)은 10,10을 세 번 쪼갠 뒤에만 생기는데 DP는 10을 쪼개지 않는다.
    #   이 30칸이 Task 7의 n_undecided 기대값 30의 뿌리다.
    for 라벨 in ("21", "A,10", "20"):
        assert float(freq[ROW_OF_LABEL[라벨]].sum()) == 0.0, 라벨
