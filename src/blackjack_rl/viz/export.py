"""이 파일은 matplotlib 그림을 PNG 파일과 GIF 애니메이션으로 저장하는 일을 한다
입력: Figure 하나 또는 Figure 목록과 저장 경로
출력: 저장한 파일의 경로 문자열 (파일은 디스크에 남는다)
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from matplotlib.animation import PillowWriter
from matplotlib.figure import Figure

PNG_DPI = 110
GIF_DPI = 90


def save_png(fig: Figure, path: str | Path, dpi: int = PNG_DPI) -> str:
    """그림 하나를 PNG로 저장하고 경로 문자열을 돌려준다."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    return str(out)


def export_gif(frames: Sequence[Figure], path: str | Path, fps: int = 12,
               dpi: int = GIF_DPI) -> str:
    """Figure 목록을 GIF 한 장으로 묶고 경로 문자열을 돌려준다."""
    if len(frames) == 0:
        raise ValueError("frames가 비어 있다. 그림이 최소 한 장은 필요하다.")

    base = tuple(frames[0].get_size_inches())
    for i, fig in enumerate(frames):
        if tuple(fig.get_size_inches()) != base:
            # 왜: PillowWriter는 self.fig의 크기로 픽셀 버퍼 크기를 계산한다.
            #     프레임 크기가 섞이면 PIL이 "not enough image data"로 죽는다.
            raise ValueError(f"{i}번 프레임 크기 {tuple(fig.get_size_inches())}가 "
                             f"첫 프레임 {base}와 다르다")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    with writer.saving(frames[0], str(out), dpi=dpi):
        for fig in frames:
            # 왜: PillowWriter.grab_frame은 self.fig 한 장만 캡처한다.
            #     프레임마다 다른 Figure를 쓰려면 이 자리를 갈아 끼우는 수밖에 없다.
            writer.fig = fig
            writer.grab_frame()
    return str(out)
