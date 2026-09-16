"""이 파일은 학습곡선이 기준선과 축을 올바르게 그리는지 확인한다.
입력: 가짜 곡선 배열.
출력: pytest 통과/실패.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from tabs import tab2_curves  # noqa: E402


def 가짜곡선():
    ep = np.geomspace(1000, 3_000_000, 50)
    그리디 = np.linspace(-0.46, -0.0071, 50)
    행동 = 그리디 - 0.01
    return ep, 그리디, 행동


def test_곡선이_두_개_이상이고_기준선이_붙는다():
    ep, g, b = 가짜곡선()
    기준 = {"항상 스탠드": -0.160377, "딜러 모방": -0.056746}
    fig = tab2_curves.build_curve(ep, g, b, -0.005108, 기준)
    # 그리디 + 행동 + DP + 기준선 2개
    assert len(fig.data) >= 3 + len(기준)


def test_x축이_로그다():
    # 왜: 선형 축이면 초반 급락이 후반의 미세한 개선을 전부 뭉갠다.
    ep, g, b = 가짜곡선()
    fig = tab2_curves.build_curve(ep, g, b, -0.005108, {})
    assert fig.layout.xaxis.type == "log"


def test_DP_상한선이_점선이다():
    ep, g, b = 가짜곡선()
    fig = tab2_curves.build_curve(ep, g, b, -0.005108, {})
    점선 = [t for t in fig.data if getattr(t, "name", "") and "DP" in t.name]
    assert len(점선) == 1
    assert 점선[0].line.dash == "dash"


def test_기준선_이름이_그대로_들어간다():
    # 왜 이 이름인가: eval.simulate.baseline_evs가 실제로 쓰는 이름이다
    #   ("DP 최적(상한)" / "딜러 모방" / "항상 스탠드" / "무작위(합법 균등)").
    ep, g, b = 가짜곡선()
    기준 = {"항상 스탠드": -0.160377}
    fig = tab2_curves.build_curve(ep, g, b, -0.005108, 기준)
    이름들 = {getattr(t, "name", "") for t in fig.data}
    assert "항상 스탠드" in 이름들
