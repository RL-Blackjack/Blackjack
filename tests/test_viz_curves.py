"""이 파일은 학습곡선·편향곡선 렌더러가 올바른 축과 선을 만드는지 검증한다
입력: 손으로 만든 가짜 EV 곡선 배열
출력: pytest 통과/실패
"""

import numpy as np
import pytest
from matplotlib.figure import Figure

from blackjack_rl.eval.simulate import baseline_evs
from blackjack_rl.rules import RULES_V1
from blackjack_rl.viz.curves import (BASELINE_EVS, BOTTOM_YLIM, DP_OPTIMAL_EV,
                                     TOP_YLIM, bias_curves, learning_curves)


@pytest.fixture
def episodes():
    # 왜: snapshot.log_schedule과 같은 로그 간격 200점을 쓴다.
    return np.geomspace(1e3, 1e7, 200)


def fake_curve(episodes, final, start=-0.46, speed=2.0):
    """무작위 -46%에서 final로 수렴하는 가짜 학습곡선."""
    decades = np.log10(episodes / episodes[0])
    weight = 1.0 - np.exp(-speed * decades)
    return start + (final - start) * weight


def test_학습곡선은_2단_패널이다(episodes):
    fig = learning_curves(episodes, {"MC": fake_curve(episodes, -0.0062)},
                          dp_ev=DP_OPTIMAL_EV)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2


def test_두_패널의_y구간이_설계서대로다(episodes):
    fig = learning_curves(episodes, {"MC": fake_curve(episodes, -0.0062)},
                          dp_ev=DP_OPTIMAL_EV)
    top, bottom = fig.axes
    assert top.get_ylim() == TOP_YLIM
    assert bottom.get_ylim() == BOTTOM_YLIM
    assert top.get_xscale() == "log"
    assert bottom.get_xscale() == "log"


def test_DP최적_EV가_두_패널_모두에_점선으로_깔린다(episodes):
    fig = learning_curves(episodes, {"MC": fake_curve(episodes, -0.0062)},
                          dp_ev=DP_OPTIMAL_EV)
    for ax in fig.axes:
        dotted = [ln for ln in ax.lines if ln.get_linestyle() == ":"]
        assert len(dotted) == 1
        assert dotted[0].get_ydata()[0] == pytest.approx(DP_OPTIMAL_EV)


def test_다섯_곡선을_한_패널에_그린다(episodes):
    이름들 = ("MC", "Q러닝", "Double Q", "SARSA", "Expected SARSA")
    greedy = {이름: fake_curve(episodes, -0.006 - 0.0002 * i)
              for i, 이름 in enumerate(이름들)}
    behavior = {이름: fake_curve(episodes, -0.012) for 이름 in 이름들}
    fig = learning_curves(episodes, greedy, behavior=behavior, dp_ev=DP_OPTIMAL_EV)
    top = fig.axes[0]
    # 그리디 5 + 행동 5 + DP 점선 1 + 기준선 3
    assert len(top.lines) == 5 + 5 + 1 + len(BASELINE_EVS)


def test_기준선_세_개가_이_저장소의_DP_실측값이다():
    # 왜: 문헌값(-0.1554 / -0.0596)을 쓰면 학습곡선 그림과 최종 평가표가
    #     같은 정책에 대해 서로 다른 숫자를 말하게 된다. 우리 DP가 플레이어
    #     규칙 전제가 달라서 문헌과 다른 것이 정상이다.
    assert BASELINE_EVS["무작위(합법 균등)"] == pytest.approx(-0.460235, abs=2e-6)
    assert BASELINE_EVS["항상 스탠드"] == pytest.approx(-0.160377, abs=2e-6)
    assert BASELINE_EVS["딜러 모방"] == pytest.approx(-0.056746, abs=2e-6)
    assert DP_OPTIMAL_EV == pytest.approx(-0.005108, abs=2e-6)


def test_기준선_상수가_실제_DP_계산과_같다():
    """상수를 손으로 적어 두면 언젠가 갈라진다. 같은 규칙으로 직접 풀어 대조한다."""
    실측 = dict(baseline_evs(RULES_V1))
    for 이름, 값 in BASELINE_EVS.items():
        assert 실측[이름] == pytest.approx(값, abs=2e-6), 이름
    assert 실측["DP 최적(상한)"] == pytest.approx(DP_OPTIMAL_EV, abs=2e-6)


def test_곡선_길이가_다르면_거부한다(episodes):
    with pytest.raises(ValueError):
        learning_curves(episodes, {"MC": np.zeros(10)}, dp_ev=DP_OPTIMAL_EV)


def test_편향곡선은_0선을_깔고_곡선을_그린다(episodes):
    decades = np.log10(episodes / episodes[0])
    series = {"Q러닝": 0.35 * np.exp(-0.4 * decades),
              "Double Q": 0.05 * np.exp(-0.6 * decades)}
    fig = bias_curves(episodes, series, title="Double Q 편향")
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert len(ax.lines) == 2 + 1
    assert ax.get_title() == "Double Q 편향"


def test_편향곡선도_길이를_확인한다(episodes):
    with pytest.raises(ValueError):
        bias_curves(episodes, {"Q러닝": np.zeros(7)})
