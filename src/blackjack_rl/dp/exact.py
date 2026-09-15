"""이 파일은 무한덱 블랙잭의 정확 동적계획법 풀이를 한다
입력: RuleSet(규칙)과 StateKey(상태), 평가할 정책 배열
출력: DPResult(Q, V, gap, ev_initial, rules_fp)와 정책의 정확 기대수익(float)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from blackjack_rl.chartspec import ChartTable, project
from blackjack_rl.dp.dealer_dp import dealer_outcome_dist
from blackjack_rl.rules import RuleSet
from blackjack_rl.state import (
    DOUBLE,
    HIT,
    KEY_INDEX,
    Q_SHAPE,
    REACHABLE_KEYS,
    SPLIT,
    STAND,
    StateKey,
)

# 무한덱 카드 확률. 1은 에이스, 10은 10/J/Q/K 네 랭크가 겹쳐서 4/13이다.
CARD_PROBS: dict[int, float] = {
    c: (4.0 / 13.0 if c == 10 else 1.0 / 13.0) for c in range(1, 11)
}
DEALER_TOTALS = (17, 18, 19, 20, 21)
V_SHAPE = Q_SHAPE[:-1]
PAIR_HARD_TOTALS = (4, 6, 8, 10, 12, 14, 16, 18, 20)

# ─────────────────────────────────────────────────────────────────────
# 정직성 각주 (발표 슬라이드에도 그대로 들어간다)
#
# 이 DP는 '한 손이 몇 번 쪼개졌는가'(split_depth)만 추적하고,
# '이 라운드에 손이 이미 몇 개인가'(max_hands_per_round=4)는 추적하지 않는다.
# 손별 깊이 3만 보면 이론상 최대 8손까지 허용되므로 실제 카지노 규칙보다
# 아주 조금 후하게 계산된다. 즉 이것은 '분산 0의 완전 정확 DP'가 아니라
# '라운드 손 수 상한을 근사한, 그 외에는 정확한 DP'다.
# 근사의 크기는 max_hands_per_round on/off 시뮬레이션 두 벌로 %p 단위로 측정한다.
# ─────────────────────────────────────────────────────────────────────


@dataclass
class DPResult:
    """DP가 푼 결과 한 묶음."""

    Q: np.ndarray            # Q_SHAPE, 불법 행동은 nan
    V: np.ndarray            # V_SHAPE, 상태가치
    gap: np.ndarray          # V_SHAPE, 최선 - 차선
    ev_initial: float        # 최적 플레이의 정확 하우스엣지
    rules_fp: str


def add_card(total: int, is_soft: int, card: int) -> tuple[int, int]:
    """(합계, 소프트여부)에 카드 한 장을 더한 새 (합계, 소프트여부)를 돌려준다."""
    # 왜: Hand 클래스는 카드 리스트를 매번 다시 더해서 수십만 번 도는 DP 재귀에 느리다.
    #     DP는 숫자 두 개만으로 완전히 같은 계산을 한다.
    if card == 1:
        if is_soft:
            # 이미 에이스 하나를 11로 쓰는 중이므로 새 에이스는 1로 센다
            new_total, new_soft = total + 1, 1
        else:
            new_total, new_soft = total + 11, 1
    else:
        new_total, new_soft = total + card, int(is_soft)

    if new_total > 21 and new_soft:
        new_total -= 10      # 11로 세던 에이스를 1로 내린다
        new_soft = 0
    return new_total, new_soft


@lru_cache(maxsize=None)
def stand_ev(total: int, dealer_up: int, rules: RuleSet) -> float:
    """total로 스탠드했을 때의 정확 기대수익(베팅 1단위 기준)."""
    # 왜: dealer_outcome_dist는 카드값 1~10(1=에이스)을 받는데 StateKey의
    #     dealer_up은 2~11(11=에이스)이다. 여기서 한 번만 변환해 둔다.
    up_card = 1 if dealer_up == 11 else dealer_up
    dist = dealer_outcome_dist(up_card, rules, peeked=True)
    ev = float(dist[5])                       # 딜러 버스트 = 무조건 승
    for i, dealer_total in enumerate(DEALER_TOTALS):
        if total > dealer_total:
            ev += float(dist[i])
        elif total < dealer_total:
            ev -= float(dist[i])
    return ev


def _is_pair_total(total: int, is_soft: int) -> bool:
    """이 (합계, 소프트) 조합이 페어일 수 있는가."""
    if is_soft:
        return total == 12       # 유일한 소프트 페어는 A,A
    return total in PAIR_HARD_TOTALS


class _Solver:
    """상태값을 재귀로 푸는 메모이제이션 계산기. mode가 행동 선택 방식을 정한다."""

    def __init__(self, rules, mode="optimal", policy_full=None, pi=None):
        if not rules.dealer_peeks:
            # 왜: 노홀카드(유럽식)는 딜러 BJ에 더블·스플릿 추가 베팅까지 몰수되어
            #     EV 분해식 자체가 달라진다. 조용히 틀린 값을 주는 대신 막는다.
            raise ValueError("dp.exact는 dealer_peeks=True 규칙만 지원한다")
        self.rules = rules
        self.mode = mode                 # "optimal" | "deterministic" | "stochastic"
        self.policy_full = policy_full
        self.pi = pi
        self._q_memo: dict[tuple, np.ndarray] = {}
        self._v_memo: dict[tuple, float] = {}

    # 내부 상태는 (total, is_soft, can_double, can_split, depth, dealer_up) 튜플이다.

    def q_values(self, st: tuple) -> np.ndarray:
        cached = self._q_memo.get(st)
        if cached is not None:
            return cached
        total, is_soft, can_double, can_split, depth, up = st
        q = np.full(4, np.nan, dtype=np.float64)
        q[STAND] = stand_ev(total, up, self.rules)
        q[HIT] = self._hit_value(st)
        if can_double:
            q[DOUBLE] = self._double_value(st)
        if can_split:
            q[SPLIT] = self._split_value(st)
        self._q_memo[st] = q
        return q

    def state_value(self, st: tuple) -> float:
        cached = self._v_memo.get(st)
        if cached is not None:
            return cached
        q = self.q_values(st)
        if self.mode == "optimal":
            v = float(np.nanmax(q))
        elif self.mode == "deterministic":
            v = float(q[self._policy_action(st, q)])
        else:
            v = self._stochastic_value(st, q)
        self._v_memo[st] = v
        return v

    def _hit_value(self, st: tuple) -> float:
        total, is_soft, _cd, _cs, depth, up = st
        ev = 0.0
        for card, p in CARD_PROBS.items():
            new_total, new_soft = add_card(total, is_soft, card)
            if new_total > 21:
                ev += p * -1.0
            else:
                # 왜: 카드를 한 장 더 받으면 첫 두 장이 아니게 되므로 더블·스플릿이 사라진다
                ev += p * self.state_value((new_total, new_soft, 0, 0, depth, up))
        return ev

    def _double_value(self, st: tuple) -> float:
        total, is_soft, _cd, _cs, _depth, up = st
        ev = 0.0
        for card, p in CARD_PROBS.items():
            new_total, _new_soft = add_card(total, is_soft, card)
            if new_total > 21:
                ev += p * -1.0
            else:
                ev += p * stand_ev(new_total, up, self.rules)
        return 2.0 * ev      # 더블은 베팅이 2배이고 카드는 딱 한 장만 받는다

    def _split_value(self, st: tuple) -> float:
        total, is_soft, _cd, _cs, depth, up = st
        rank = 1 if is_soft else total // 2
        one_hand = 0.0
        for card, p in CARD_PROBS.items():
            one_hand += p * self._child_value(rank, card, depth + 1, up)
        # 왜: 무한덱에서 스플릿한 두 자식 손은 카드 분포가 같고 서로 독립이다.
        #     딜러 결과를 공유하지만 기댓값은 선형이므로 EV가 손별로 분해된다.
        #     따라서 Q(s, SPLIT | depth=d) = 2 * E_c[V_{d+1}(rank, c)] 가 정확하고,
        #     깊이가 max_split_depth 이하로 유한하므로 이 재귀는 반드시 끝난다.
        #     즉 "DP는 리스플릿을 못 푼다"는 통념은 무한덱에서 거짓이고,
        #     max_split_depth를 1로 내릴 필요가 없다.
        return 2.0 * one_hand

    def _child_value(self, rank: int, card: int, depth: int, up: int) -> float:
        if rank == 1:
            total, is_soft = add_card(11, 1, card)
        else:
            total, is_soft = add_card(rank, 0, card)

        if rank == 1 and self.rules.split_aces_one_card:
            # 왜: 스플릿한 에이스는 카드 한 장만 받고 즉시 끝난다.
            #     21이 나와도 블랙잭이 아니라 1:1이므로 stand_ev로 그냥 정산한다.
            return stand_ev(total, up, self.rules)

        can_double = 0
        if self.rules.double_after_split and self._double_allowed(total, is_soft):
            can_double = 1

        can_split = 0
        if card == rank and depth < self.rules.max_split_depth:
            if rank == 1 and not self.rules.resplit_aces:
                can_split = 0     # 기본 규칙은 에이스 리스플릿을 금지한다
            else:
                can_split = 1

        return self.state_value((total, is_soft, can_double, can_split, depth, up))

    def _double_allowed(self, total: int, is_soft: int) -> bool:
        if self.rules.double_any_two:
            return True
        # 왜: DA2가 꺼진 규칙에서는 하드 9/10/11에서만 더블을 허용하는 것이 카지노 표준이다
        return (not is_soft) and total in (9, 10, 11)

    def _key_of(self, st: tuple) -> StateKey:
        total, is_soft, can_double, can_split, depth, up = st
        stored_depth = depth if self.rules.include_split_depth_in_state else 0
        return StateKey(total, is_soft, up, can_double, can_split, stored_depth)

    def _policy_action(self, st: tuple, q: np.ndarray) -> int:
        idx = KEY_INDEX.get(self._key_of(st))
        if idx is None:
            # 왜: state.py의 BFS 열거에 없는 키에는 정책이 정의돼 있지 않다.
            #     항상 합법인 STAND로 고정해 결정론을 지킨다. 이 분기가 실제로
            #     타면 '최적 정책 평가 == ev_initial' 테스트가 깨지므로 숨지 않는다.
            return STAND
        action = int(self.policy_full[idx])
        if action < 0 or action > 3 or np.isnan(q[action]):
            return STAND
        return action

    def _stochastic_value(self, st: tuple, q: np.ndarray) -> float:
        idx = KEY_INDEX.get(self._key_of(st))
        if idx is None:
            return float(q[STAND])
        probs = self.pi[idx]
        value = 0.0
        weight = 0.0
        for action in range(4):
            p = float(probs[action])
            if p <= 0.0 or np.isnan(q[action]):
                continue
            value += p * float(q[action])
            weight += p
        if weight <= 0.0:
            return float(q[STAND])
        # 왜: 불법 행동에 확률이 남아 있어도 합법 행동들 안에서 다시 정규화한다
        return value / weight


def _all_states(rules: RuleSet) -> list[tuple]:
    """Q 배열을 채울 모든 (합계, 소프트, 더블가능, 스플릿가능, 깊이, 업카드)를 만든다."""
    states: list[tuple] = []
    for up in range(2, 12):
        for is_soft in (0, 1):
            lowest = 12 if is_soft else 4
            for total in range(lowest, 22):
                for can_double in (0, 1):
                    for depth in range(rules.max_split_depth + 1):
                        states.append((total, is_soft, can_double, 0, depth, up))
                        if not _is_pair_total(total, is_soft):
                            continue
                        if depth >= rules.max_split_depth:
                            continue   # 더 쪼갤 여유가 없으면 can_split=1 상태가 없다
                        states.append((total, is_soft, can_double, 1, depth, up))
    return states


def _initial_ev(solver: _Solver, rules: RuleSet) -> float:
    """첫 두 장과 딜러 업카드를 전부 열거해 라운드 시작 시점의 정확 EV를 구한다."""
    ev = 0.0
    for up_card, p_up in CARD_PROBS.items():
        dealer_up = 11 if up_card == 1 else up_card
        if up_card == 1:
            p_dealer_bj = CARD_PROBS[10]     # 업카드 A면 홀카드가 10일 확률
        elif up_card == 10:
            p_dealer_bj = CARD_PROBS[1]      # 업카드 10이면 홀카드가 A일 확률
        else:
            p_dealer_bj = 0.0

        for c1, p1 in CARD_PROBS.items():
            for c2, p2 in CARD_PROBS.items():
                weight = p_up * p1 * p2
                if c1 == 1:
                    total, is_soft = add_card(11, 1, c2)
                else:
                    total, is_soft = add_card(c1, 0, c2)

                # 두 장으로 소프트 21이 되는 경우는 A+10뿐이다 = 내추럴 블랙잭
                if total == 21 and is_soft == 1:
                    ev += weight * (1.0 - p_dealer_bj) * rules.blackjack_payout
                    continue

                can_double = 1 if solver._double_allowed(total, is_soft) else 0
                can_split = 1 if (c1 == c2 and rules.max_split_depth > 0) else 0
                v = solver.state_value((total, is_soft, can_double, can_split, 0, dealer_up))
                ev += weight * (p_dealer_bj * -1.0 + (1.0 - p_dealer_bj) * v)
    return ev


def solve_optimal(rules: RuleSet) -> DPResult:
    """규칙 하나에 대한 최적 Q/V/gap과 정확 하우스엣지를 푼다."""
    solver = _Solver(rules, mode="optimal")
    Q = np.full(Q_SHAPE, np.nan, dtype=np.float64)
    V = np.full(V_SHAPE, np.nan, dtype=np.float64)
    gap = np.full(V_SHAPE, np.nan, dtype=np.float64)

    for st in _all_states(rules):
        total, is_soft, can_double, can_split, depth, up = st
        q = solver.q_values(st)
        Q[total, is_soft, up, can_double, can_split, depth, :] = q
        # STAND와 HIT은 언제나 합법이므로 합법 행동은 항상 2개 이상이다
        ranked = np.sort(q[~np.isnan(q)])[::-1]
        V[total, is_soft, up, can_double, can_split, depth] = float(ranked[0])
        gap[total, is_soft, up, can_double, can_split, depth] = float(ranked[0] - ranked[1])

    return DPResult(
        Q=Q,
        V=V,
        gap=gap,
        ev_initial=_initial_ev(solver, rules),
        rules_fp=rules.fingerprint(),
    )


def evaluate_policy(policy_full: np.ndarray, rules: RuleSet) -> float:
    """결정적 정책의 정확 EV. 시뮬레이션 0회, 분산 0."""
    arr = np.asarray(policy_full, dtype=np.int8)
    if arr.shape != (len(REACHABLE_KEYS),):
        raise ValueError(f"policy_full 모양은 ({len(REACHABLE_KEYS)},)여야 한다: {arr.shape}")
    solver = _Solver(rules, mode="deterministic", policy_full=arr)
    return _initial_ev(solver, rules)


def evaluate_stochastic(pi: np.ndarray, rules: RuleSet) -> float:
    """확률적 정책(예: 엡실론 그리디)의 정확 EV."""
    arr = np.asarray(pi, dtype=np.float64)
    if arr.shape != (len(REACHABLE_KEYS), 4):
        raise ValueError(f"pi 모양은 ({len(REACHABLE_KEYS)}, 4)여야 한다: {arr.shape}")
    solver = _Solver(rules, mode="stochastic", pi=arr)
    return _initial_ev(solver, rules)


def action_values(key: StateKey, rules: RuleSet) -> np.ndarray:
    """상태 하나의 행동가치 4개. 불법 행동은 nan이다."""
    total, is_soft = int(key.total), int(key.is_soft)
    can_split, depth = int(key.can_split), int(key.split_depth)
    if can_split and not _is_pair_total(total, is_soft):
        raise ValueError(f"합계 {total}(soft={is_soft})는 페어가 될 수 없어 can_split=1이 불가능하다")
    if can_split and depth >= rules.max_split_depth:
        raise ValueError(f"split_depth={depth}에서는 더 쪼갤 수 없어 can_split=1이 불가능하다")
    solver = _Solver(rules, mode="optimal")
    st = (total, is_soft, int(key.can_double), can_split, depth, int(key.dealer_up))
    return solver.q_values(st).copy()


def greedy_policy_full(dp: DPResult) -> np.ndarray:
    """DPResult에서 REACHABLE_KEYS 순서의 결정적 최적 정책 배열을 뽑는다."""
    policy = np.zeros(len(REACHABLE_KEYS), dtype=np.int8)
    for i, key in enumerate(REACHABLE_KEYS):
        q = dp.Q[key.total, key.is_soft, key.dealer_up,
                 key.can_double, key.can_split, key.split_depth]
        if np.all(np.isnan(q)):
            policy[i] = STAND     # DP가 열거하지 않은 칸은 항상 합법인 STAND로 둔다
        else:
            policy[i] = int(np.nanargmax(q))
    return policy


def optimal_chart(rules: RuleSet) -> ChartTable:
    """규칙 하나를 풀어 DP 최적 36x10 전략표를 만든다."""
    # 왜: 표 격자 정의는 chartspec에만 있어야 하므로 DP는 Q만 주고 사영은 위임한다
    return project(solve_optimal(rules).Q, rules)
