"""이 파일은 전략표 히트맵 렌더러가 올바른 그림 객체를 만드는지 검증한다
입력: 정확 DP 해와 출판 참조표
출력: pytest 통과/실패
"""

import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from blackjack_rl.chartspec import ChartTable, project
from blackjack_rl.eval.agreement import chart_margin
from blackjack_rl.reference import load_reference_chart
from blackjack_rl.viz.celldata import tie_grid, undecided_grid
from blackjack_rl.viz.heatmap import MISMATCH_COLOR, policy_heatmap, use_korean_font


@pytest.fixture
def chart(dp, rules):
    return project(dp.Q, rules)


@pytest.fixture
def margin(dp):
    return chart_margin(dp.Q)


@pytest.fixture
def overlay():
    ref = load_reference_chart()
    # 왜: 오버레이는 notation만 읽는다. 참조표 CSV에는 행동 번호가 없으므로
    #     action은 -1(미정)으로 채워 ChartTable 모양만 맞춘다.
    return ChartTable(action=np.full((36, 10), -1, dtype=np.int8),
                      notation=ref.notation, rules_fp=ref.rules_fp)


def test_Figure를_돌려준다(chart, margin):
    assert isinstance(policy_heatmap(chart, margin), Figure)


def test_pyplot_전역목록을_더럽히지_않는다(chart, margin):
    # 왜: Task 14에서 GIF용으로 수십 장을 만든다. pyplot에 쌓이면
    #     "20장 넘게 열렸다" 경고가 뜨고 메모리도 안 돌아온다.
    import matplotlib.pyplot as plt
    before = list(plt.get_fignums())
    policy_heatmap(chart, margin)
    assert list(plt.get_fignums()) == before


def test_글자가_360칸_전부_찍힌다(chart, margin):
    fig = policy_heatmap(chart, margin)
    assert len(fig.axes[0].texts) == 360


def test_미결정칸에는_글자를_찍지_않는다(chart, margin):
    visits = np.ones((36, 10), dtype=np.int32)
    visits[25:, :] = 0                      # 페어 구역 11행을 통째로 미방문 처리
    fig = policy_heatmap(chart, margin, undecided_mask=undecided_grid(visits))
    assert len(fig.axes[0].texts) == 360 - 11 * 10


def test_색배열은_margin을_자른_값이다(chart, margin):
    fig = policy_heatmap(chart, margin)
    image = fig.axes[0].images[0]
    assert image.get_array().shape == (36, 10)
    assert image.get_clim() == (0.0, 0.20)


def test_오버레이를_주면_불일치_2칸에_굵은_테두리가_생긴다(chart, margin, overlay):
    fig = policy_heatmap(chart, margin, overlay=overlay)
    want = to_rgba(MISMATCH_COLOR)
    boxes = [p for p in fig.axes[0].patches
             if isinstance(p, Rectangle) and p.get_edgecolor() == want]
    assert len(boxes) == 2


def test_오버레이가_없으면_불일치_테두리도_없다(chart, margin):
    assert len(policy_heatmap(chart, margin).axes[0].patches) == 0


def test_동점마스크를_주면_5칸에_점선_테두리가_생긴다(chart, margin):
    fig = policy_heatmap(chart, margin, tie_mask=tie_grid(margin))
    assert len(fig.axes[0].patches) == 5


def test_제목이_그대로_붙는다(chart, margin):
    fig = policy_heatmap(chart, margin, title="테스트 제목")
    assert fig.axes[0].get_title() == "테스트 제목"


def test_한글폰트를_고른다():
    name = use_korean_font()
    assert isinstance(name, str) and name != ""
