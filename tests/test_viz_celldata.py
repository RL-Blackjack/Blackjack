"""이 파일은 전략표 칸 색·마스크 계산 함수들을 검증한다
입력: 정확 DP 해와 출판 참조표
출력: pytest 통과/실패
"""

import numpy as np
import pytest

from blackjack_rl.chartspec import DEALER_COLS, ROW_INDEX, project
from blackjack_rl.eval.agreement import TRIVIAL_DELTA, chart_gap, chart_margin
from blackjack_rl.reference import load_reference_chart
from blackjack_rl.viz.celldata import (GRID_SHAPE, MARGIN_VMAX, mismatch_grid,
                                       shade_grid, tie_grid, undecided_grid)


def cell(label, dealer_up):
    """'16' 행 x 딜러 10 열 같은 사람 말을 (행 번호, 열 번호)로 바꾼다."""
    return ROW_INDEX[label], DEALER_COLS.index(dealer_up)


def test_설치된_matplotlib과_pillow_버전이_핀과_같다():
    # 왜: Task 13~15가 이 두 개 위에서 돈다. 버전이 밀리면 그림이 미묘하게
    #     달라지고 GIF 크기도 달라지므로, 설치 여부를 테스트로 못박는다.
    import matplotlib
    import PIL
    assert matplotlib.__version__ == "3.11.2"
    assert PIL.__version__ == "12.3.0"


def test_격자_모양은_36x10이다():
    assert GRID_SHAPE == (36, 10)


def test_색_최대값은_자명_임계값과_같은_상수다():
    # 왜: 두 곳에 0.20을 따로 적으면 한쪽만 고쳐도 아무도 모른다.
    #     색의 최대값과 '자명한 칸' 임계값은 같은 숫자여야 그림과 표가 같은 말을 한다.
    assert MARGIN_VMAX == TRIVIAL_DELTA == 0.20


def test_chart_margin은_360칸_전부_유한하다(dp):
    m = chart_margin(dp.Q)
    assert m.shape == (36, 10)
    assert np.isfinite(m).all()
    assert (m >= 0).all()


def test_유명한_칸들의_margin_실측값(dp):
    m = chart_margin(dp.Q)
    # 하드 16 vs 10은 교과서가 "가장 가까운 결정"이라 부르는 칸이다
    assert m[cell("16", 10)] == pytest.approx(0.000604, abs=1e-5)
    # 하드 21 vs 2는 스탠드가 압도적이라 margin이 1을 훌쩍 넘는다
    assert m[cell("21", 2)] == pytest.approx(1.882007, abs=1e-4)
    assert m[cell("A,A", 6)] == pytest.approx(0.481426, abs=1e-4)
    assert m[cell("8,8", 10)] == pytest.approx(0.059275, abs=1e-4)


def test_동점칸은_정확히_5칸이다(dp):
    # 왜: TIE_DELTA=0.005 기준으로 DP가 동점이라 부르는 칸 목록을 못박는다.
    #     이 목록이 흔들리면 일치율 보고의 4분류가 통째로 흔들린다.
    tie = tie_grid(chart_margin(dp.Q))
    found = {(r, c) for r, c in zip(*np.nonzero(tie))}
    expected = {cell("12", 4), cell("16", 10), cell("A,4", 4),
                cell("A,7", 2), cell("3,3", 2)}
    assert found == expected


def test_색이_가장_진한_칸과_자명한_칸은_같은_뜻이_아니다(dp):
    """페어 행에서 margin은 '1위 vs 2위', 갭은 'SPLIT vs 나머지'다.
    같은 임계값 0.20을 써도 179칸과 186칸으로 갈린다. 색은 표시, 갭은 판정이다."""
    assert int((chart_margin(dp.Q) >= MARGIN_VMAX).sum()) == 179
    assert int((chart_gap(dp) >= TRIVIAL_DELTA).sum()) == 186


def test_미결정_마스크는_방문0인_칸만_잡는다():
    visits = np.ones((36, 10), dtype=np.int32)
    visits[30, 4] = 0
    visits[35, 9] = 0
    und = undecided_grid(visits)
    assert int(und.sum()) == 2
    assert bool(und[30, 4])


def test_미결정_마스크는_모양이_다르면_거부한다():
    with pytest.raises(ValueError):
        undecided_grid(np.zeros((10, 10), dtype=np.int32))


def test_shade는_0과_MARGIN_VMAX_사이로_잘린다(dp):
    m = chart_margin(dp.Q)
    und = np.zeros((36, 10), dtype=bool)
    und[0, 0] = True
    s = shade_grid(m, und)
    assert np.isnan(s[0, 0])
    rest = s[~und]
    assert rest.min() >= 0.0
    assert rest.max() <= MARGIN_VMAX


def test_DP최적표와_출판표의_불일치는_사전등록한_2칸뿐이다(dp, rules):
    ai = project(dp.Q, rules)
    ref = load_reference_chart()
    bad = mismatch_grid(ai.notation, ref.notation)
    found = {(r, c) for r, c in zip(*np.nonzero(bad))}
    assert found == {cell("A,2", 5), cell("A,4", 4)}


def test_미결정칸은_불일치로_세지_않는다(dp, rules):
    ai = project(dp.Q, rules)
    ref = load_reference_chart()
    und = np.zeros((36, 10), dtype=bool)
    und[cell("A,2", 5)] = True
    bad = mismatch_grid(ai.notation, ref.notation, und)
    assert int(bad.sum()) == 1
