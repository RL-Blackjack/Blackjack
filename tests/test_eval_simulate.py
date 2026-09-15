"""이 파일은 CRN 카드 스트림과 절대 평가가 DP 정확값과 맞는지 본다.
입력: conftest의 rules·dp 픽스처
출력: 스트림 모양/분포, 표본 크기 공식, 버그 경보, EV 3시그마 검증
"""

import numpy as np
import pytest

from blackjack_rl.dp.exact import evaluate_policy, greedy_policy_full
from blackjack_rl.eval.simulate import (CARDS_PER_HAND, EVAL_SEED, SanityAlarm,
                                        baseline_evs, evaluate, make_card_stream,
                                        required_n, required_visits, sanity_alarm)
from blackjack_rl.rules import RULES_SHOE
from blackjack_rl.state import REACHABLE_KEYS

N_EVAL = 200_000


def test_카드스트림은_라운드별_칸으로_잘려있고_10이_4_13이다(rules):
    stream = make_card_stream(EVAL_SEED, 5_000, rules)
    assert stream.shape == (5_000, CARDS_PER_HAND)
    assert stream.dtype == np.int8
    assert int(stream.min()) >= 1 and int(stream.max()) <= 10
    비율 = float(np.mean(stream == 10))
    # 왜 이 구간인가: 참값 4/13 = 0.3077, 표본 16만 장이라 표준오차가 0.0012다.
    #   n=20만 스트림의 실측 비율은 0.30762였다.
    assert 0.303 < 비율 < 0.313


def test_같은_시드는_같은_스트림_다른_시드는_다른_스트림(rules):
    a = make_card_stream(EVAL_SEED, 1_000, rules)
    b = make_card_stream(EVAL_SEED, 1_000, rules)
    c = make_card_stream(EVAL_SEED + 1, 1_000, rules)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_슈모드_스트림은_만들지_않고_막는다():
    with pytest.raises(ValueError):
        make_card_stream(EVAL_SEED, 100, RULES_SHOE)


def test_required_n이_설계서의_508만과_맞는다():
    # 왜 1.15인가: 라운드당 순손익의 표준편차 실측값이 1.1534다.
    assert required_n(0.001, 1.15) == 5_080_330
    assert required_n(0.002, 1.15) == 1_270_083


def test_required_visits가_19나누기_델타제곱과_맞는다():
    # 왜 19인가: delta=1.0을 넣으면 계수 자체가 나온다. 설계서의 "약 19/델타^2"과 같다.
    assert required_visits(1.0) == 19
    assert required_visits(0.001) == 18_994_286
    assert required_visits(0.01) == 189_943


def test_sanity_alarm은_EV가_바닥보다_좋으면_예외를_던진다():
    sanity_alarm(-0.005108)          # DP 정확 EV는 통과해야 한다
    with pytest.raises(SanityAlarm) as err:
        sanity_alarm(-0.0005)
    본문 = str(err.value)
    assert "스플릿 보상" in 본문 and "3:2" in 본문


def test_DP최적정책의_CRN_EV가_DP정확값과_3시그마_안에서_일치한다(rules, dp):
    stream = make_card_stream(EVAL_SEED, N_EVAL, rules)
    결과 = evaluate(greedy_policy_full(dp), rules, stream)
    print(f"\nCRN EV = {결과.ev:+.5f} ± {결과.se:.5f} (n={결과.n:,})")
    print(f"DP  EV = {dp.ev_initial:+.5f}")
    assert 결과.n == N_EVAL
    assert abs(결과.ev - dp.ev_initial) < 3 * 결과.se
    assert 결과.ci95[0] < dp.ev_initial < 결과.ci95[1]


def test_항상스탠드_정책의_CRN_EV도_DP정확값과_3시그마_안에서_일치한다(rules):
    policy = np.zeros(len(REACHABLE_KEYS), dtype=np.int8)   # 0 = STAND
    정확값 = evaluate_policy(policy, rules)
    # 왜 문헌값 -0.1554가 아닌가: 문헌값은 규칙 전제(덱 모형·S17/H17)가 우리와 다르다.
    #   같은 규칙으로 직접 풀면 -0.160377이고, 테스트는 그 정확값과 대조한다.
    assert 정확값 == pytest.approx(-0.160377, abs=2e-6)
    stream = make_card_stream(EVAL_SEED, N_EVAL, rules)
    결과 = evaluate(policy, rules, stream)
    print(f"\n항상 스탠드: 시뮬 {결과.ev:+.5f} ± {결과.se:.5f} / DP {정확값:+.5f}")
    # 왜 이 정책도 보는가: 최적 정책 하나만 맞으면 '우연히 맞았다'를 배제할 수 없다.
    assert abs(결과.ev - 정확값) < 3 * 결과.se


def test_기준_정책_네_개의_EV가_DP_실측값과_맞는다(rules):
    """학습곡선 그림(Task 15)과 최종 평가표(Task 17)가 둘 다 이 함수를 쓴다.
    숫자를 두 곳에 손으로 적으면 한 저장소 안에서 두 그림이 다른 말을 한다."""
    표 = dict(baseline_evs(rules))
    # 왜 문헌값과 다른가: 문헌값의 규칙 전제가 우리와 다르다. 문헌의
    #   '항상 스탠드 -15.54%', '딜러 모방 -5.96%'는 그 배당을 빼고 계산한 값이다.
    assert 표["DP 최적(상한)"] == pytest.approx(-0.005108, abs=2e-6)
    assert 표["딜러 모방"] == pytest.approx(-0.056746, abs=2e-6)
    assert 표["항상 스탠드"] == pytest.approx(-0.160377, abs=2e-6)
    assert 표["무작위(합법 균등)"] == pytest.approx(-0.460235, abs=2e-6)
    # 순서도 고정한다 — evaluate.py의 리포트가 첫 항목을 'DP 상한'으로 읽는다.
    assert [이름 for 이름, _ in baseline_evs(rules)][0] == "DP 최적(상한)"
