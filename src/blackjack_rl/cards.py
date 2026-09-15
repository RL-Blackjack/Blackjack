"""이 파일은 카드 한 장의 값과 카드를 뽑아 주는 슈 세 종류를 정의한다.
입력: RuleSet과 난수 Generator, 또는 미리 만들어 둔 카드 배열.
출력: 카드 값 1~10 정수, Hi-Lo 태그, InfiniteShoe/FiniteShoe/ReplayShoe.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from blackjack_rl.rules import RuleSet

# 한 덱 13랭크의 값 목록. 그대로 뽑기 가중치가 된다(10이 4/13).
# 왜: 카드를 랭크 코드(1~13)로 들고 다니면 어디선가 한 번은 값으로 바꿔야 하고
#     그 지점이 버그의 단골이다. 처음부터 값 1~10만 쓴다. A=1, J/Q/K=10.
CARD_VALUES: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10)
CARDS_PER_DECK: int = 52

# 왜: 남은 덱이 0에 가까우면 트루카운트가 무한대로 튄다. 실제 카지노 관행대로
#     0.5덱을 바닥으로 둔다.
_MIN_DECKS_FOR_TRUE_COUNT: float = 0.5


def card_value(c: int) -> int:
    """카드 한 장의 셈 값을 돌려준다. A는 1이다(11로 세는 건 hand.py 몫)."""
    if c == 11:
        # 왜: dealer_up은 에이스를 11로 표기한다. 여기서 카드 값 1로 되돌려
        #     표기와 값을 섞어 쓰는 사고를 한곳에서 막는다.
        return 1
    if not 1 <= c <= 10:
        raise ValueError(f"카드 값은 1~10(에이스 표기 11 포함)이어야 한다: {c}")
    return int(c)


def hilo_tag(c: int) -> int:
    """Hi-Lo 카운트 태그. 2~6은 +1, 7~9는 0, 10과 A는 -1."""
    값 = card_value(c)
    if 2 <= 값 <= 6:
        return 1
    if 7 <= 값 <= 9:
        return 0
    return -1


@runtime_checkable
class Shoe(Protocol):
    """카드를 한 장씩 내주는 물건. 환경은 이 네 가지만 보고 쓴다."""

    running_count: int

    def draw(self) -> int: ...

    def maybe_shuffle(self) -> bool: ...

    @property
    def decks_remaining(self) -> float: ...

    @property
    def true_count(self) -> float: ...


class InfiniteShoe:
    """복원추출 슈. 카드를 빼도 분포가 그대로라 카운팅이 무의미하다."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        # 왜: 무한덱에서는 언제나 0이지만, Shoe 프로토콜을 맞추려고 속성은 둔다.
        self.running_count: int = 0

    def draw(self) -> int:
        # 왜: rng.choice()보다 integers()가 몇 배 빠르다. 한 학기에 수억 번 부른다.
        자리 = int(self._rng.integers(0, len(CARD_VALUES)))
        return CARD_VALUES[자리]

    def maybe_shuffle(self) -> bool:
        return False

    @property
    def decks_remaining(self) -> float:
        return float("inf")

    @property
    def true_count(self) -> float:
        return 0.0


class FiniteShoe:
    """실제 카지노처럼 여러 덱을 섞어 놓고 컷카드까지만 쓰는 슈."""

    def __init__(self, decks: int, penetration: float, rng: np.random.Generator) -> None:
        if not 1 <= decks <= 8:
            raise ValueError(f"decks는 1~8이어야 한다: {decks}")
        if not 0.1 <= penetration <= 1.0:
            raise ValueError(f"penetration은 0.1~1.0이어야 한다: {penetration}")

        self._decks = int(decks)
        self._rng = rng
        self._total = CARDS_PER_DECK * self._decks
        self._cut_index = int(self._total * penetration)
        self._cards = np.zeros(self._total, dtype=np.int8)
        self._pos = 0
        self.running_count: int = 0
        self.shuffle()

    def shuffle(self) -> None:
        """슈를 새로 만들어 섞고 카운트를 0으로 되돌린다."""
        한덱 = np.repeat(np.array(CARD_VALUES, dtype=np.int8), 4)
        self._cards = np.tile(한덱, self._decks)
        self._rng.shuffle(self._cards)
        self._pos = 0
        self.running_count = 0

    def draw(self) -> int:
        if self._pos >= self._total:
            raise RuntimeError(
                "슈가 소진됐다. 라운드와 라운드 사이에 maybe_shuffle()을 불러야 한다."
            )
        카드 = int(self._cards[self._pos])
        self._pos += 1
        self.running_count += hilo_tag(카드)
        return 카드

    def maybe_shuffle(self) -> bool:
        """컷카드를 지났으면 섞고 True, 아니면 아무것도 안 하고 False."""
        if self._pos >= self._cut_index:
            self.shuffle()
            return True
        return False

    @property
    def cards_remaining(self) -> int:
        return self._total - self._pos

    @property
    def decks_remaining(self) -> float:
        return self.cards_remaining / CARDS_PER_DECK

    @property
    def true_count(self) -> float:
        return self.running_count / max(self.decks_remaining, _MIN_DECKS_FOR_TRUE_COUNT)


class ReplayShoe:
    """미리 만들어 둔 카드 배열을 순서대로 내주는 슈. CRN 평가 전용이다."""

    def __init__(self, cards: Sequence[int] | np.ndarray) -> None:
        배열 = np.asarray(cards, dtype=np.int8)
        if 배열.ndim != 1:
            raise ValueError("cards는 1차원이어야 한다.")
        if 배열.size == 0:
            raise ValueError("cards가 비어 있다.")
        if int(배열.min()) < 1 or int(배열.max()) > 10:
            raise ValueError("cards의 모든 값이 1~10이어야 한다.")

        self._cards = 배열
        self._pos = 0
        self.running_count: int = 0

    def draw(self) -> int:
        if self._pos >= self._cards.size:
            raise RuntimeError("재생할 카드가 다 떨어졌다. 카드 스트림을 더 길게 만들어라.")
        카드 = int(self._cards[self._pos])
        self._pos += 1
        self.running_count += hilo_tag(카드)
        return 카드

    def maybe_shuffle(self) -> bool:
        # 왜: 같은 카드로 두 정책을 비교하는 게 존재 이유다. 여기서 섞으면 CRN이 깨진다.
        return False

    @property
    def cards_remaining(self) -> int:
        return int(self._cards.size) - self._pos

    @property
    def decks_remaining(self) -> float:
        return self.cards_remaining / CARDS_PER_DECK

    @property
    def true_count(self) -> float:
        return self.running_count / max(self.decks_remaining, _MIN_DECKS_FOR_TRUE_COUNT)


def make_shoe(rules: RuleSet, rng: np.random.Generator) -> Shoe:
    """규칙이 정한 덱 방식에 맞는 슈를 만들어 준다."""
    if rules.deck_mode == "infinite":
        return InfiniteShoe(rng)
    return FiniteShoe(decks=rules.decks, penetration=rules.penetration, rng=rng)
