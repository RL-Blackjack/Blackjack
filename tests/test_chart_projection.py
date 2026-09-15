"""이 파일은 Q 배열을 36x10 전략표로 사영하는 project()를 검증한다
입력: RULES_V1로 푼 DP 결과 Q
출력: pytest 통과/실패
"""

import numpy as np
import pytest

from blackjack_rl.chartspec import CHART_ROWS, DEALER_COLS, ROW_INDEX, project
from blackjack_rl.dp.exact import optimal_chart, solve_optimal
from blackjack_rl.rules import RULES_H17, RULES_V1
from blackjack_rl.state import DOUBLE, HIT, SPLIT, STAND


@pytest.fixture(scope="module")
def dp():
    return solve_optimal(RULES_V1)


@pytest.fixture(scope="module")
def chart(dp):
    return project(dp.Q, RULES_V1)


def col(dealer_up):
    return DEALER_COLS.index(dealer_up)


# ── 표의 기본 형태 ──

def test_표는_36x10이고_지문을_달고_있다(chart):
    assert chart.action.shape == (36, 10)
    assert chart.notation.shape == (36, 10)
    assert chart.action.dtype == np.int8
    assert chart.rules_fp == RULES_V1.fingerprint()


def test_DP표는_미결정_칸이_없다(chart):
    assert int(chart.action.min()) >= 0
    assert "" not in set(chart.notation.ravel().tolist())


def test_쓰이는_표기는_여섯_가지뿐이다(chart):
    assert set(chart.notation.ravel().tolist()) <= {"S", "H", "D", "Ds", "Y", "N"}


def test_페어행은_Y나_N만_쓰고_나머지는_안_쓴다(chart):
    for row_i, row in enumerate(CHART_ROWS):
        for j in range(10):
            mark = chart.notation[row_i, j]
            if row.kind == "pair":
                assert mark in ("Y", "N")
            else:
                assert mark in ("S", "H", "D", "Ds")


def test_방문이_0인_칸은_미결정으로_남는다(dp):
    visits = np.ones((36, 10), dtype=np.int32)
    visits[0, 0] = 0
    table = project(dp.Q, RULES_V1, visits=visits)
    assert table.action[0, 0] == -1
    assert table.notation[0, 0] == ""
    assert table.action[0, 1] >= 0


# ── 알려진 칸 ──

def test_알려진_칸들이_기본전략과_같다(chart):
    # 왜 A,8 vs 6이 'S'인가: 정본 RULES_V1은 S17(딜러가 소프트 17에서 스탠드)이다.
    # 소프트 19 vs 딜러 6은 S17에서 스탠드가 더블보다 1.64%p 유리하다
    # (DP: STAND 0.495977 / DOUBLE 0.479599, Wizard of Odds 공개값과 소수 셋째 자리까지 일치).
    # 이 칸이 'Ds'가 되는 것은 H17 차트뿐이며, 그건 아래 전용 테스트가 검증한다.
    assert chart.notation[ROW_INDEX["A,8"], col(6)] == "S"
    assert chart.notation[ROW_INDEX["A,7"], col(3)] == "Ds"
    assert chart.notation[ROW_INDEX["11"], col(10)] == "D"
    assert chart.notation[ROW_INDEX["8,8"], col(10)] == "Y"
    assert chart.notation[ROW_INDEX["10,10"], col(6)] == "N"
    assert chart.notation[ROW_INDEX["12"], col(4)] == "S"
    assert chart.notation[ROW_INDEX["16"], col(10)] == "H"


def test_Ds와_D의_action은_모두_DOUBLE이다(chart):
    assert chart.action[ROW_INDEX["A,7"], col(3)] == DOUBLE
    assert chart.action[ROW_INDEX["11"], col(10)] == DOUBLE


def test_규칙을_S17에서_H17로_바꾸면_A8_vs_6만_뒤집힌다():
    """규칙 한 칸을 바꾸면 표가 '딱 그만큼만' 달라지는지 본다.

    이게 이 프로젝트에서 가장 강한 검증이다. 표를 외운 게 아니라 규칙에서
    유도했다면, S17→H17 변경은 알려진 몇 칸만 바꿔야 한다. 소프트 19 vs 6은
    그중 가장 유명한 칸이다(S17에서 스탠드, H17에서 더블).
    """
    s17 = optimal_chart(RULES_V1)
    h17 = optimal_chart(RULES_H17)

    자리 = (ROW_INDEX["A,8"], col(6))
    assert s17.notation[자리] == "S"
    assert h17.notation[자리] == "Ds"
    assert h17.action[자리] == DOUBLE

    # 규칙 변경의 영향이 표 전체로 번지지 않는지: 360칸 중 달라지는 칸은 소수다.
    다른칸 = int((s17.notation != h17.notation).sum())
    assert 1 <= 다른칸 <= 12, f"S17과 H17의 차이가 {다른칸}칸이다(1~12 예상)"


def test_YN의_action은_SPLIT_여부와_맞는다(chart):
    assert chart.action[ROW_INDEX["8,8"], col(10)] == SPLIT
    assert chart.action[ROW_INDEX["10,10"], col(6)] != SPLIT


# ── D / Ds 표기가 1순위·2순위에서 제대로 나오는가 ──

def test_D는_더블_금지시_HIT_Ds는_더블_금지시_STAND다(dp, chart):
    for row_i, row in enumerate(CHART_ROWS):
        if row.kind == "pair":
            continue
        for j, dealer_up in enumerate(DEALER_COLS):
            mark = chart.notation[row_i, j]
            if mark not in ("D", "Ds"):
                continue
            key = row.to_key(dealer_up, can_double=False)
            q = dp.Q[key.total, key.is_soft, key.dealer_up,
                     key.can_double, key.can_split, key.split_depth]
            second = int(np.nanargmax(q))
            if mark == "D":
                assert second == HIT
            else:
                assert second == STAND


# ── 일관성: 페어 칸에서 SPLIT을 빼면 같은 합계의 일반 상태와 같아야 한다 ──

def test_페어칸에서_SPLIT을_빼면_같은_합계의_일반상태와_같은_행동이다(dp):
    for row in CHART_ROWS:
        if row.kind != "pair":
            continue
        for dealer_up in DEALER_COLS:
            key = row.to_key(dealer_up, can_double=True)
            q_pair = dp.Q[key.total, key.is_soft, key.dealer_up,
                          key.can_double, 1, 0].copy()
            q_pair[SPLIT] = np.nan
            q_plain = dp.Q[key.total, key.is_soft, key.dealer_up,
                           key.can_double, 0, 0]
            assert int(np.nanargmax(q_pair)) == int(np.nanargmax(q_plain))


def test_페어행의_비스플릿_선택이_하드_표_칸과_일치한다(dp, chart):
    for row in CHART_ROWS:
        if row.kind != "pair" or row.is_soft == 1:
            continue                       # A,A는 소프트 12라 하드 표에 대응 행이 없다
        hard_label = str(row.total)
        if hard_label not in ROW_INDEX:
            continue                       # 2,2(합계 4)는 하드 표에 4행이 없다
        for j, dealer_up in enumerate(DEALER_COLS):
            key = row.to_key(dealer_up, can_double=True)
            q_pair = dp.Q[key.total, key.is_soft, key.dealer_up, 1, 1, 0].copy()
            q_pair[SPLIT] = np.nan
            assert int(np.nanargmax(q_pair)) == int(chart.action[ROW_INDEX[hard_label], j])


# ── optimal_chart는 project와 같은 표를 준다 ──

def test_optimal_chart는_project_결과와_같다(chart):
    table = optimal_chart(RULES_V1)
    assert np.array_equal(table.action, chart.action)
    assert np.array_equal(table.notation, chart.notation)
    assert table.rules_fp == chart.rules_fp
