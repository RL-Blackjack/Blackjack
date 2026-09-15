"""이 파일은 3계층 일치율·4분류·EV 손실 계산을 검사한다.
입력: conftest의 rules·dp 픽스처
출력: 자명/비자명 칸 수, DP표 자기대조 100%, 뒤집힌 칸의 EV 손실
"""

import numpy as np
import pytest

from blackjack_rl.chartspec import CHART_ROWS, DEALER_COLS, ChartTable, project
from blackjack_rl.dp.exact import greedy_policy_full, optimal_chart
from blackjack_rl.eval.agreement import (TRIVIAL_DELTA, chart_gap, chart_margin,
                                         chart_visits, compare,
                                         load_pre_registered,
                                         make_agreement_fn, trivial_mask)
from blackjack_rl.eval.simulate import EVAL_SEED, make_card_stream, natural_cell_freq
from blackjack_rl.reference import load_reference_chart
from blackjack_rl.state import Q_SHAPE

ROW_OF_LABEL = {행.label: i for i, 행 in enumerate(CHART_ROWS)}
N_FREQ = 200_000


@pytest.fixture(scope="module")
def freq(rules, dp):
    """칸별 자연 발생 빈도. 모듈 안에서 한 번만 센다."""
    stream = make_card_stream(EVAL_SEED, N_FREQ, rules)
    return natural_cell_freq(greedy_policy_full(dp), rules, stream)


@pytest.fixture(scope="module")
def ref():
    return load_reference_chart()


@pytest.fixture(scope="module")
def pre():
    return load_pre_registered()


def 다_방문했다고_치는_방문표():
    """방문 수 때문에 미결정이 생기지 않게 하는 36x10 표."""
    return np.full((36, 10), 1_000_000, dtype=np.int32)


def test_자명한_칸은_186개_비자명한_칸은_174개다(dp):
    mask = trivial_mask(dp)
    자명 = int(mask.sum())
    print(f"\n자명(갭 >= {TRIVIAL_DELTA}) {자명}칸 / 비자명 {360 - 자명}칸")
    # 왜 하드코딩인가: 이 숫자는 규칙이 고정되면 바뀌지 않는 DP의 결정값이다.
    #   바뀌었다면 DP나 표 격자 어딘가가 달라진 것이므로 반드시 알아야 한다.
    assert 자명 == 186
    assert int((~mask).sum()) == 174


def test_판정용_갭과_표시용_margin은_같은_숫자가_아니다(dp):
    """페어 행에서 갭은 'SPLIT vs 나머지 최선'이고 margin은 '1위 vs 2위'다.
    임계값 0.20을 똑같이 써도 186칸과 179칸으로 갈린다. 일치율 분모는 갭 쪽이다."""
    assert int((chart_gap(dp) >= TRIVIAL_DELTA).sum()) == 186
    assert int((chart_margin(dp.Q) >= TRIVIAL_DELTA).sum()) == 179


def test_chart_gap은_동점칸과_뻔한칸을_숫자로_가른다(dp):
    gap = chart_gap(dp)
    assert gap.shape == (36, 10)
    assert float(gap.min()) >= 0.0
    # 하드 16 vs 10은 60년 논쟁이 붙은 칸이라 갭이 0.005보다 작다(실측 0.000604).
    assert float(gap[ROW_OF_LABEL["16"], DEALER_COLS.index(10)]) < 0.005
    # 8,8 vs 6은 스플릿이 압도적으로 좋아 갭이 0.5를 넘는다(실측 0.564303).
    assert float(gap[ROW_OF_LABEL["8,8"], DEALER_COLS.index(6)]) > 0.5


def test_chart_visits는_7차원_방문표를_36x10_int32로_사영한다():
    방문표 = np.zeros(Q_SHAPE, dtype=np.int32)
    키 = CHART_ROWS[ROW_OF_LABEL["8,8"]].to_key(6, can_double=True)
    방문표[키.total, 키.is_soft, 키.dealer_up, 키.can_double, 키.can_split, 키.split_depth, 3] = 7
    사영 = chart_visits(방문표)
    assert 사영.shape == (36, 10)
    # 왜 int32인가: npz 스키마의 chart_visits가 int32다. 여기서 int64를 돌려주면
    #   학습 쪽과 일치율 쪽의 같은 표가 두 dtype으로 갈라진다.
    assert 사영.dtype == np.int32
    assert int(사영[ROW_OF_LABEL["8,8"], DEALER_COLS.index(6)]) == 7
    assert int(사영.sum()) == 7


def test_chart_margin은_float32이고_유명한_칸의_실측값과_같다(dp):
    m = chart_margin(dp.Q)
    assert m.shape == (36, 10)
    assert m.dtype == np.float32
    assert float(m[ROW_OF_LABEL["16"], DEALER_COLS.index(10)]) == pytest.approx(
        0.000604, abs=1e-5)
    assert float(m[ROW_OF_LABEL["21"], DEALER_COLS.index(2)]) == pytest.approx(
        1.882007, abs=1e-4)
    assert float(m[ROW_OF_LABEL["A,A"], DEALER_COLS.index(6)]) == pytest.approx(
        0.481426, abs=1e-4)


def test_DP최적표를_AI표로_넣으면_일치율이_100퍼센트다(rules, dp, ref, pre, freq):
    ai = optimal_chart(rules)
    보고서 = compare(ai, ref, dp, 다_방문했다고_치는_방문표(), freq, pre)
    print(f"\ntier_a={보고서.tier_a:.6f} tier_b={보고서.tier_b:.6f} "
          f"미결정={보고서.n_undecided} EV손실={보고서.ev_loss_pp:.6f}%p")
    for m in 보고서.mismatches:
        print(f"  {m.cell} vs {m.dealer}: AI={m.ai} 참조={m.ref} 사유={m.cause}")
    # 왜 정확히 1.0인가: AI표가 DP표 그 자체이므로 사전등록 2칸을 뺀 358칸이 전부 맞는다.
    assert 보고서.tier_a == 1.0
    assert 보고서.tier_b == 1.0
    assert 보고서.n_undecided == 0
    assert 보고서.ev_loss_pp == 0.0
    assert len(보고서.mismatches) == 2
    assert {(m.cell, m.dealer) for m in 보고서.mismatches} == {("A,2", 5), ("A,4", 4)}
    assert {m.cause for m in 보고서.mismatches} == {"deck_model", "statistical_tie"}


def test_비자명한_칸_하나를_뒤집으면_두_계층이_모두_내려간다(rules, dp, ref, pre, freq):
    ai = optimal_chart(rules)
    망친표 = ChartTable(action=ai.action.copy(), notation=ai.notation.copy(),
                      rules_fp=ai.rules_fp)
    i, j = ROW_OF_LABEL["13"], DEALER_COLS.index(10)
    assert str(망친표.notation[i, j]) == "H"
    망친표.notation[i, j] = "S"      # 하드 13 vs 10 은 갭 0.115176의 비자명 칸이다

    보고서 = compare(망친표, ref, dp, 다_방문했다고_치는_방문표(), freq, pre)
    print(f"\ntier_a={보고서.tier_a:.6f} tier_b={보고서.tier_b:.6f} "
          f"EV손실={보고서.ev_loss_pp:.4f}%p")
    assert 보고서.tier_a == pytest.approx(357 / 358)
    assert 보고서.tier_b == pytest.approx(171 / 172)
    # 왜 이 구간인가: 실측 빈도 0.023715 x DP 갭 0.115176 x 100 = 0.27314%p 다.
    assert 0.25 < 보고서.ev_loss_pp < 0.30
    틀린칸 = [m for m in 보고서.mismatches if m.cell == "13"]
    assert len(틀린칸) == 1
    assert 틀린칸[0].cause == "undertrained"
    assert 틀린칸[0].dp_delta == pytest.approx(0.115176, abs=1e-5)


def test_자명한_칸을_뒤집으면_tier_a만_내려가고_tier_b는_그대로다(rules, dp, ref, pre, freq):
    ai = optimal_chart(rules)
    망친표 = ChartTable(action=ai.action.copy(), notation=ai.notation.copy(),
                      rules_fp=ai.rules_fp)
    i, j = ROW_OF_LABEL["8,8"], DEALER_COLS.index(6)
    assert str(망친표.notation[i, j]) == "Y"
    망친표.notation[i, j] = "N"      # 8,8 vs 6 은 갭 0.564303의 자명한 칸이다

    보고서 = compare(망친표, ref, dp, 다_방문했다고_치는_방문표(), freq, pre)
    print(f"\ntier_a={보고서.tier_a:.6f} tier_b={보고서.tier_b:.6f}")
    # 왜 이렇게 갈리는가: 헤드라인(tier_a)은 떨어져도 실질 지표(tier_b)는 안 떨어진다.
    #   "자명한 칸으로 일치율을 부풀렸냐"는 질문에 두 숫자를 나란히 보여주는 이유다.
    assert 보고서.tier_a == pytest.approx(357 / 358)
    assert 보고서.tier_b == 1.0


def test_방문이_0인_칸은_미결정으로_세고_분모에서_빠진다(rules, dp, ref, pre, freq):
    방문표 = (freq > 0).astype(np.int32) * 1_000
    ai = project(dp.Q, rules, visits=방문표)
    보고서 = compare(ai, ref, dp, 방문표, freq, pre)
    print(f"\n미결정={보고서.n_undecided} tier_a={보고서.tier_a:.6f}")
    # 왜 30인가: 하드 20 / 하드 21 / A,10 세 행(각 10칸)은 자연 딜로 결정이 생기지 않는다.
    #   하드 21 행은 REACHABLE_KEYS에 아예 없고, 나머지 두 행은 ExploringStarts도
    #   만들 수 없는 키다. 학습 산출물의 n_undecided는 30에서 시작한다.
    assert 보고서.n_undecided == 30
    assert 보고서.tier_a == 1.0
    assert len(보고서.mismatches) == 2


def test_일치율_훅은_표와_방문표를_받아_두_숫자를_돌려준다(rules, dp):
    """train/runner의 AgreementFn 자리에 그대로 꽂히는 함수다.
    이 훅이 없으면 모든 산출물의 agree_a/agree_b가 영구히 nan으로 남는다."""
    # 왜 5,000핸드인가: 훅 배선을 확인하는 테스트라 빈도 정밀도가 필요 없다.
    #   실제 실행은 scripts/train.py의 기본값 200,000핸드를 쓴다.
    훅 = make_agreement_fn(rules, dp, n_freq=5_000)
    표 = optimal_chart(rules)
    a, b = 훅(표, 다_방문했다고_치는_방문표())
    assert a == pytest.approx(1.0)
    assert b == pytest.approx(1.0)
