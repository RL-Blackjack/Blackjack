"""이 파일은 PNG 저장과 GIF 내보내기가 실제 파일을 만드는지 검증한다
입력: 정확 DP 해로 그린 히트맵 그림들과 pytest tmp_path
출력: pytest 통과/실패
"""

import numpy as np
import pytest
from PIL import Image

from blackjack_rl.chartspec import project
from blackjack_rl.eval.agreement import chart_margin
from blackjack_rl.viz.celldata import undecided_grid
from blackjack_rl.viz.export import export_gif, save_png
from blackjack_rl.viz.heatmap import policy_heatmap


def make_frames(dp, rules, n):
    """학습이 진행되는 것처럼 보이는 가짜 프레임 n장을 만든다."""
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.25, size=dp.Q.shape)
    frames = []
    for weight in np.linspace(0.0, 1.0, n):
        # 왜: 노이즈에서 DP 정답으로 서서히 섞는다. 실제 스냅샷이 없어도
        #     "색이 굳어가는" 장면을 그대로 재현할 수 있다.
        q = dp.Q * weight + noise * (1.0 - weight)
        visits = np.full((36, 10), 1000, dtype=np.int32)
        visits[26 + int(10 * (1.0 - weight)):, :] = 0
        frames.append(policy_heatmap(project(q, rules), chart_margin(q),
                                     undecided_mask=undecided_grid(visits),
                                     title=f"가중치 {weight:.2f}"))
    return frames


def test_PNG를_저장하면_파일이_생긴다(dp, rules, tmp_path):
    fig = policy_heatmap(project(dp.Q, rules), chart_margin(dp.Q), title="DP 최적표")
    out = save_png(fig, tmp_path / "chart.png")
    path = tmp_path / "chart.png"
    assert str(path) == out
    assert path.exists()
    # 왜: 실측 93,916 ~ 100,647 바이트다. 아래 구간은 "빈 그림이 아니다"만
    #     보증하도록 넓게 잡았다.
    assert 20_000 < path.stat().st_size < 400_000
    with Image.open(path) as im:
        assert im.format == "PNG"


def test_PNG는_없는_폴더도_만든다(dp, rules, tmp_path):
    fig = policy_heatmap(project(dp.Q, rules), chart_margin(dp.Q))
    out = save_png(fig, tmp_path / "media" / "sub" / "chart.png")
    assert (tmp_path / "media" / "sub" / "chart.png").exists()
    assert out.endswith("chart.png")


def test_GIF_프레임_수가_넣은_그림_수와_같다(dp, rules, tmp_path):
    frames = make_frames(dp, rules, 4)
    out = export_gif(frames, tmp_path / "anim.gif", fps=6)
    path = tmp_path / "anim.gif"
    assert path.exists() and str(path) == out
    with Image.open(path) as im:
        assert im.format == "GIF"
        assert im.n_frames == 4


def test_GIF_파일_크기가_프레임당_수십KB다(dp, rules, tmp_path):
    frames = make_frames(dp, rules, 4)
    export_gif(frames, tmp_path / "anim.gif", fps=6)
    size = (tmp_path / "anim.gif").stat().st_size
    # 왜: 실측 4프레임 205,282 바이트(프레임당 약 46KB, dpi=90). 40프레임이면
    #     1,843,535 바이트라 GitHub에 그냥 커밋할 수 있다. 이 구간을 벗어나면
    #     dpi나 FIG_SIZE가 바뀐 것이고, 200프레임 GIF는 약 9MB가 되므로 만들지 않는다.
    assert 60_000 < size < 600_000


def test_프레임이_비면_거부한다(tmp_path):
    with pytest.raises(ValueError):
        export_gif([], tmp_path / "anim.gif")


def test_프레임_크기가_섞이면_거부한다(dp, rules, tmp_path):
    frames = make_frames(dp, rules, 2)
    frames[1].set_size_inches(4.0, 4.0)
    with pytest.raises(ValueError):
        export_gif(frames, tmp_path / "anim.gif")
