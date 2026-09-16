"""이 파일은 판단 근거 막대그래프가 학습값과 참값을 함께 그리는지 확인한다.
입력: 가짜 Q 행과 DP 행.
출력: pytest 통과/실패.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from tabs import tab3_explain  # noqa: E402


def test_막대가_학습값과_참값_두_계열이다():
    q = np.array([-0.54, -0.53, -1.08, np.nan])
    dp_row = np.array([-0.5404, -0.5398, -1.0797, np.nan])
    mask = np.array([True, True, True, False])
    fig = tab3_explain.build_bars(q, dp_row, mask, chosen=1)
    assert len(fig.data) == 2
    이름들 = {t.name for t in fig.data}
    assert "학습된 Q" in 이름들
    assert "DP 참값" in 이름들


def test_불법_행동은_막대에서_빠진다():
    q = np.array([-0.54, -0.53, -1.08, np.nan])
    dp_row = np.array([-0.5404, -0.5398, -1.0797, np.nan])
    mask = np.array([True, True, True, False])
    fig = tab3_explain.build_bars(q, dp_row, mask, chosen=1)
    # 합법 3개만 남는다
    assert len(fig.data[0].x) == 3


def test_고른_행동이_표시된다():
    q = np.array([-0.54, -0.53, -1.08, np.nan])
    dp_row = np.array([-0.5404, -0.5398, -1.0797, np.nan])
    mask = np.array([True, True, True, False])
    fig = tab3_explain.build_bars(q, dp_row, mask, chosen=1)
    assert "히트" in str(fig.layout.title.text)
