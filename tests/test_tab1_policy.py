"""이 파일은 전략표 히트맵이 올바른 모양과 표기를 만드는지 확인한다.
입력: 가짜 36x10 배열들.
출력: pytest 통과/실패.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from tabs import tab1_policy  # noqa: E402


def 가짜표():
    action = np.zeros((36, 10), dtype=np.int8)
    notation = np.full((36, 10), "S", dtype="<U2")
    margin = np.linspace(0.0, 1.0, 360).reshape(36, 10).astype(np.float32)
    undecided = np.zeros((36, 10), dtype=bool)
    undecided[0, 0] = True
    return action, notation, margin, undecided


def test_히트맵이_36x10이다():
    a, n, m, u = 가짜표()
    fig = tab1_policy.build_heatmap(a, n, m, u)
    z = np.array(fig.data[0].z)
    assert z.shape == (36, 10)


def test_미결정_칸은_글자를_찍지_않는다():
    a, n, m, u = 가짜표()
    fig = tab1_policy.build_heatmap(a, n, m, u)
    글자 = np.array(fig.data[0].text)
    assert 글자[0, 0] == ""
    assert 글자[0, 1] == "S"


def test_미결정_칸의_색값은_결측이다():
    # 왜: 0으로 두면 '확신이 0'처럼 보인다. 아예 비워야 회백색으로 그려진다.
    a, n, m, u = 가짜표()
    fig = tab1_policy.build_heatmap(a, n, m, u)
    z = np.array(fig.data[0].z, dtype=float)
    assert np.isnan(z[0, 0])


def test_행동마다_색이_정의되어_있다():
    for 표기 in ("S", "H", "D", "Ds", "Y", "N"):
        assert 표기 in tab1_policy.ACTION_COLORS


def test_오버레이는_다른_칸에만_테두리를_친다():
    a, n, m, u = 가짜표()
    fig = tab1_policy.build_heatmap(a, n, m, u)
    참조 = np.full((36, 10), "S", dtype="<U2")
    참조[5, 5] = "H"
    참조[7, 2] = "H"
    tab1_policy.overlay_reference(fig, 참조, n)
    assert len(fig.layout.shapes) == 2


def test_오버레이는_미결정_칸을_세지_않는다():
    # 왜: 학습으로 채울 수 없는 칸을 '틀렸다'고 표시하면 오해를 만든다.
    a, n, m, u = 가짜표()
    n2 = n.copy()
    n2[0, 0] = ""
    fig = tab1_policy.build_heatmap(a, n2, m, u)
    참조 = np.full((36, 10), "H", dtype="<U2")
    tab1_policy.overlay_reference(fig, 참조, n2)
    # 0,0은 미결정이므로 제외 -> 359칸
    assert len(fig.layout.shapes) == 359
