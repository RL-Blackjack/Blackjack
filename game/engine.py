"""이 파일은 시드와 행동 목록만으로 블랙잭 한 라운드를 재현한다.
입력: 시드 정수 하나와 지금까지 고른 행동 목록.
출력: 다음 결정을 기다리는 Pending 또는 정산이 끝난 Finished.
"""

from __future__ import annotations

import secrets
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from blackjack_rl.cards import InfiniteShoe
from blackjack_rl.dp.exact import add_card
from blackjack_rl.env import BlackjackEnv, RoundResult
from blackjack_rl.hand import Hand
from blackjack_rl.rng import make_streams
from blackjack_rl.rules import RULES_V1, RuleSet
from blackjack_rl.state import DOUBLE, HIT, SPLIT, StateKey

SEED_BITS: int = 62
MAX_CARD: int = 10


class IllegalAction(Exception):
    """클라이언트가 지금 고를 수 없는 행동을 보냈다."""


class _중단(Exception):
    """더 재생할 행동이 없다. 내부 신호이므로 밖으로 나가지 않는다."""

    def __init__(self, key: StateKey, legal: np.ndarray, ctx, cards: list[int]) -> None:
        super().__init__("다음 행동이 필요하다")
        self.key = key
        self.legal = legal
        self.ctx = ctx
        self.cards = cards


@dataclass(frozen=True)
class Pending:
    """다음 결정을 기다리는 상태."""

    key: StateKey
    legal: tuple[int, ...]
    hand_index: int
    n_hands: int
    bet: float
    dealer_up: int
    player_cards: tuple[int, ...]


@dataclass(frozen=True)
class HandView:
    """끝난 손 하나의 모습."""

    cards: tuple[int, ...]
    total: int
    bet: float
    result: float


@dataclass(frozen=True)
class Finished:
    """정산까지 끝난 라운드."""

    dealer_up: int
    dealer_cards: tuple[int, ...]
    net: float
    n_decisions: int
    hands: tuple[HandView, ...]


RoundView = Pending | Finished


def new_seed() -> int:
    """라운드 하나에 쓸 새 시드. 추측할 수 없어야 한다."""
    # 왜 secrets인가: 시드를 맞히면 다음 카드를 알 수 있다. 리더보드가 걸려
    #   있으므로 예측 가능한 난수를 쓰면 안 된다.
    return secrets.randbits(SEED_BITS)


def infer_card(before: tuple[int, int], after: tuple[int, int]) -> int:
    """(합계, 소프트) 전이를 만든 카드를 되찾는다. 도달 가능한 범위에서 유일하다."""
    for c in range(1, MAX_CARD + 1):
        if add_card(before[0], before[1], c) == (after[0], after[1]):
            return c
    raise ValueError(f"이 전이를 설명하는 카드가 없다: {before} -> {after}")


def _단일(rank: int) -> tuple[int, int]:
    """카드 한 장만 든 손의 (합계, 소프트). 에이스는 11로 센다."""
    return (11, 1) if rank == 1 else (rank, 0)


class _기록슈:
    """슈를 감싸 뽑은 카드를 순서대로 남긴다. 코어는 이것을 보통 슈로 본다."""

    def __init__(self, inner) -> None:  # noqa: ANN001
        self.inner = inner
        self.log: list[int] = []

    def draw(self) -> int:
        카드 = self.inner.draw()
        self.log.append(카드)
        return 카드

    def maybe_shuffle(self) -> bool:
        return self.inner.maybe_shuffle()

    @property
    def decks_remaining(self) -> float:
        return self.inner.decks_remaining

    @property
    def true_count(self) -> float:
        return self.inner.true_count


def _손합계(cards: Sequence[int]) -> int:
    """카드 목록의 블랙잭 합계."""
    return Hand(cards=list(cards)).total


def rebuild_hand_cards(log: Sequence[int], result: RoundResult) -> dict[int, list[int]]:
    """끝난 라운드의 뽑기 로그를 손 인덱스별 카드로 나눈다. 어긋나면 RuntimeError."""
    # 왜 로그로 다시 만드는가: 마지막 히트·더블 카드, 에이스 스플릿 자식의 둘째 장,
    #   내추럴 라운드의 두 장은 act가 다시 불리지 않아 전이로 되찾을 수 없다.
    #   코어는 손 0의 두 장, 딜러 업·홀, 그 뒤 손 인덱스 순서로 뽑고 딜러가 마지막에 뽑는다.
    카드: dict[int, list[int]] = {}
    자리 = 4
    try:
        카드[0] = [log[0], log[1]]
        for j, rec in enumerate(result.hands):
            if j >= 1:
                # 왜 부모의 첫 장인가: 스플릿은 페어에서만 되므로 두 장의 값이 같다.
                카드[j] = [카드[rec.parent][0], log[자리]]
                자리 += 1
            # 왜 분기 노드도 여기서 도는가: 화면에서는 빠지지만 뽑는 순서에는 끼어 있다.
            for _키, 행동 in rec.trajectory:
                if 행동 in (HIT, DOUBLE):
                    카드[j].append(log[자리])
                    자리 += 1
    except IndexError as e:
        raise RuntimeError("뽑기 로그가 플레이어 몫보다 짧다") from e
    if list(log[2:4]) + list(log[자리:]) != [int(c) for c in result.dealer_cards]:
        raise RuntimeError("플레이어 몫을 뺀 로그가 딜러 카드와 다르다")
    return 카드


def play(seed: int, actions: Sequence[int], *,
         rules: RuleSet = RULES_V1) -> RoundView:
    """시드로 라운드를 처음부터 재현하고 행동 목록만큼 진행한다."""
    슈 = _기록슈(InfiniteShoe(np.random.default_rng(np.random.SeedSequence(int(seed)))))
    환경 = BlackjackEnv(rules, 슈, make_streams(0))

    남은 = list(actions)
    카드: dict[int, list[int]] = {}
    앞선: dict[int, tuple[int, int]] = {}
    다음자리 = [1]

    def act(key: StateKey, legal: np.ndarray, ctx) -> int:  # noqa: ANN001
        자리 = ctx.hand_index
        if 자리 == 0 and 0 not in 카드:
            # 딜 순서상 로그[0], 로그[1]이 플레이어의 첫 두 장이다.
            카드[0] = [슈.log[0], 슈.log[1]]

        앞 = 앞선.get(자리)
        if 앞 is not None and 앞 != (key.total, key.is_soft):
            # 지난번 이 손에게 물어본 뒤로 (합계, 소프트)가 바뀌었다면 그 사이에
            # 카드를 한 장 받은 것이다. 전이가 카드를 유일하게 결정한다.
            카드[자리].append(infer_card(앞, (key.total, key.is_soft)))
        앞선[자리] = (key.total, key.is_soft)

        if not 남은:
            raise _중단(key, legal, ctx, list(카드[자리]))

        고른것 = 남은.pop(0)
        if not isinstance(고른것, (int, np.integer)) or not 0 <= int(고른것) <= 3:
            raise IllegalAction(f"행동 번호가 범위를 벗어났다: {고른것!r}")
        고른것 = int(고른것)
        if not legal[고른것]:
            raise IllegalAction(f"지금 고를 수 없는 행동이다: {고른것} (상태 {key})")

        if 고른것 == SPLIT:
            # 왜 이렇게 자리를 예측할 수 있는가: env._play_hands가 자식 둘을
            #   언제나 목록 끝에 덧붙이기 때문이다(중간에 끼우면 parent 인덱스가
            #   어긋나므로 코어가 그렇게 하지 않는다).
            랭크 = 1 if (key.total == 12 and key.is_soft) else key.total // 2
            for 자식 in (다음자리[0], 다음자리[0] + 1):
                카드[자식] = [랭크]
                앞선[자식] = _단일(랭크)
            다음자리[0] += 2

        return 고른것

    try:
        결과 = 환경.play_round(act)
    except _중단 as e:
        return Pending(
            key=e.key,
            legal=tuple(int(a) for a in np.flatnonzero(e.legal)),
            hand_index=int(e.ctx.hand_index),
            n_hands=int(e.ctx.n_hands),
            bet=float(e.ctx.bet),
            dealer_up=int(e.key.dealer_up),
            player_cards=tuple(e.cards),
        )

    if 남은:
        # 왜: 라운드가 이미 끝났는데 행동이 더 남았다면 저장된 기록이 깨진 것이다.
        #   조용히 무시하면 틀린 상태를 보여 주게 된다.
        raise IllegalAction(f"라운드가 끝났는데 행동이 {len(남은)}개 더 있다")

    복원 = rebuild_hand_cards(슈.log, 결과)
    손들 = []
    for i, rec in enumerate(결과.hands):
        if rec.trajectory and rec.trajectory[-1][1] == SPLIT:
            continue      # 갈라진 분기 노드는 자기 결과가 없다
        몫 = 복원[i]
        손들.append(HandView(cards=tuple(몫), total=_손합계(몫),
                            bet=float(rec.bet), result=float(rec.result)))

    return Finished(
        dealer_up=int(결과.dealer_up),
        dealer_cards=tuple(int(c) for c in 결과.dealer_cards),
        net=float(결과.net),
        n_decisions=int(결과.n_decisions),
        hands=tuple(손들),
    )
