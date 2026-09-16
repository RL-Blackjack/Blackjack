"""이 파일은 610개 상태를 훈련용과 시험용으로 두 가지 방식으로 나눈다.
입력: 상태 목록, 훈련 비율, 시드.
출력: 훈련·시험 인덱스를 담은 Split 객체.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from blackjack_rl.chartspec import DEALER_COLS
from blackjack_rl.state import StateKey

RATIOS: tuple[float, float, float] = (0.2, 0.6, 1.0)
SPLIT_KINDS: tuple[str, str] = ("random", "structural")


@dataclass(frozen=True)
class Split:
    """훈련/시험을 어떻게 갈랐는지 담는다."""

    kind: str
    ratio: float
    seed: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    held_out_dealers: tuple[int, ...]

    def name(self) -> str:
        """파일명과 표에 쓸 짧은 이름."""
        return f"{self.kind}_{int(self.ratio * 100)}_seed{self.seed}"

    def describe_ko(self) -> str:
        """사람이 읽을 한 줄 설명."""
        if self.ratio >= 1.0:
            return (f"{self.kind}: 610칸을 모두 훈련에 쓰고 같은 칸으로 평가한다"
                    f"(상한선 측정, 시험 칸 없음)")
        if self.kind == "random":
            return (f"무작위 분할: 610칸 중 {len(self.train_idx)}칸을 무작위로 훈련, "
                    f"나머지 {len(self.test_idx)}칸으로 시험한다")
        빼둔 = ", ".join("A" if d == 11 else str(d) for d in self.held_out_dealers)
        return (f"구조적 분할: 딜러 {빼둔} 열을 통째로 빼고 "
                f"{len(self.train_idx)}칸으로 훈련해, 본 적 없는 그 열 "
                f"{len(self.test_idx)}칸으로 시험한다")


def _검사(ratio: float) -> None:
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio는 0보다 크고 1 이하여야 한다: {ratio}")


def _전체(keys: Sequence[StateKey], kind: str, seed: int) -> Split:
    """비율 1.0일 때 훈련=시험=전체인 Split을 만든다."""
    전부 = np.arange(len(keys))
    return Split(kind=kind, ratio=1.0, seed=seed,
                 train_idx=전부, test_idx=전부.copy(), held_out_dealers=())


def random_split(keys: Sequence[StateKey], ratio: float, seed: int) -> Split:
    """610칸에서 무작위로 훈련 칸을 고른다."""
    _검사(ratio)
    if ratio >= 1.0:
        return _전체(keys, "random", seed)

    rng = np.random.default_rng(seed)
    섞인 = rng.permutation(len(keys))
    n = int(len(keys) * ratio)
    return Split(kind="random", ratio=ratio, seed=seed,
                 train_idx=np.sort(섞인[:n]), test_idx=np.sort(섞인[n:]),
                 held_out_dealers=())


def structural_split(keys: Sequence[StateKey], ratio: float, seed: int) -> Split:
    """딜러 업카드 열을 통째로 갈라 훈련/시험을 나눈다."""
    _검사(ratio)
    if ratio >= 1.0:
        return _전체(keys, "structural", seed)

    rng = np.random.default_rng(seed)
    섞인열 = rng.permutation(np.array(DEALER_COLS))
    n열 = max(1, round(len(DEALER_COLS) * ratio))
    훈련열 = set(int(x) for x in 섞인열[:n열])
    빼둔열 = tuple(sorted(int(x) for x in 섞인열[n열:]))

    # 왜 열 단위인가: 플레이어 합계로 나누면 15와 16이 거의 같아 여전히 쉽다.
    #   딜러 업카드는 전략을 질적으로 바꾸는 축이라, 한 열을 통째로 빼면
    #   보간이 아니라 외삽이 된다.
    훈련 = np.array([i for i, k in enumerate(keys) if k.dealer_up in 훈련열])
    시험 = np.array([i for i, k in enumerate(keys) if k.dealer_up not in 훈련열])
    return Split(kind="structural", ratio=ratio, seed=seed,
                 train_idx=훈련, test_idx=시험, held_out_dealers=빼둔열)


def make_split(kind: str, keys: Sequence[StateKey], ratio: float, seed: int) -> Split:
    """방식 이름으로 분할을 만든다."""
    if kind == "random":
        return random_split(keys, ratio, seed)
    if kind == "structural":
        return structural_split(keys, ratio, seed)
    raise ValueError(f"kind는 {SPLIT_KINDS} 중 하나여야 한다: {kind!r}")


def all_splits(keys: Sequence[StateKey], seed: int) -> list[Split]:
    """2방식 × 3비율 = 6개 분할을 만든다."""
    return [make_split(kind, keys, ratio, seed)
            for kind in SPLIT_KINDS for ratio in RATIOS]
