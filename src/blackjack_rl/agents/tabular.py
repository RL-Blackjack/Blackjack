"""이 파일은 표(테이블) 기반 강화학습 에이전트 하나를 정의한다.
입력: RuleSet, Streams, 그리고 returns.compute_transitions가 만든 list[Transition].
출력: Q 테이블과 그리디 정책 배열, ε-greedy 행동 선택.
"""

from __future__ import annotations

import numpy as np

from blackjack_rl.env import Ctx
from blackjack_rl.returns import Transition
from blackjack_rl.rng import Streams
from blackjack_rl.rules import RuleSet
from blackjack_rl.state import (LEGAL, REACHABLE_KEYS, StateKey, legal_actions,
                                new_q_table, new_visit_table)

# 왜 다섯 종을 처음부터 다 받는가: "MC를 만들고 나중에 TD를 덧붙인다"로 쪼개면
#     __init__과 _target이 같은 모듈에 두 번 쓰이고, 나중 정의가 이겨서 MC가 조용히
#     깨진다. 알고리즘 간 차이는 아래 _target() 한 함수뿐이므로 처음부터 다섯이다.
TD_ALGOS: tuple[str, ...] = ("q", "doubleq", "sarsa", "esarsa")
ALGOS: tuple[str, ...] = ("mc", *TD_ALGOS)
STEPS: tuple[str, ...] = ("sample_average", "constant")
# 왜 s1 하나뿐인가: s2(트루카운트 구간을 상태에 넣는 인코더)는 계획서 ④ W9의 몫이다.
#     지금 목록에 넣어 두면 파싱은 되는데 동작이 없는 플래그가 생긴다.
ENCODERS: tuple[str, ...] = ("s1",)

_FIELDS = ("total", "is_soft", "dealer_up", "can_double", "can_split", "split_depth")

# 왜: REACHABLE_KEYS 610개를 for문으로 한 줄씩 훑으면 greedy_full() 한 번에
#     수 밀리초가 든다. 학습 중 200프레임을 찍으므로 그게 쌓인다.
#     축별 정수 배열을 미리 만들어 두면 Q[KEY_AXES]가 (610,4)를 한 번에 떠 온다.
KEY_AXES: tuple[np.ndarray, ...] = tuple(
    np.array([getattr(key, name) for key in REACHABLE_KEYS], dtype=np.intp)
    for name in _FIELDS
)
LEGAL_ROWS: np.ndarray = LEGAL[KEY_AXES]          # bool[610, 4]
N_LEGAL: np.ndarray = LEGAL_ROWS.sum(axis=1)      # int[610], 항상 2 이상


def eps_greedy_probs(q_row: np.ndarray, mask: np.ndarray, eps: float) -> np.ndarray:
    """상태 하나에서 ε-greedy가 각 행동을 고를 확률(길이 4, 합 1)."""
    n_legal = int(mask.sum())
    probs = np.zeros(4, dtype=np.float64)
    probs[mask] = eps / n_legal
    # 왜: 불법 행동의 Q는 -inf라서 그냥 argmax해도 안 뽑히지만,
    #     where로 한 번 더 막아 두면 나중에 Q 초기화를 바꿔도 안전하다.
    best = int(np.argmax(np.where(mask, q_row, -np.inf)))
    probs[best] += 1.0 - eps
    # 왜: eps/n_legal을 더하는 과정에서 부동소수 오차가 남아 합이 1.0에서
    #     1e-17만큼 어긋날 수 있다. rng.choice는 그걸 거절한다.
    probs /= probs.sum()
    return probs


def successor_value(
    algo: str,
    next_key: StateKey,
    Q: np.ndarray,
    QB: np.ndarray | None,
    eps: float,
    rng: np.random.Generator | None,
) -> float:
    """후계자 상태 하나의 부트스트랩 값. 알고리즘 이름이 여기서만 갈린다."""
    q_row = Q[next_key]
    mask = LEGAL[next_key]

    if algo == "q":
        # Q러닝: 다음 상태에서 최선을 두었다고 가정한다(오프폴리시).
        return float(np.max(np.where(mask, q_row, -np.inf)))

    if algo == "doubleq":
        # Double Q: 어느 행동이 최선인지는 Q가 고르고, 그 값은 QB에서 읽는다.
        # 왜: 고르는 표와 읽는 표를 분리하면 "잡음이 큰 쪽을 골라 그 큰 잡음을
        #     그대로 값으로 쓰는" 최대화 편향의 연결고리가 끊어진다.
        best = int(np.argmax(np.where(mask, q_row, -np.inf)))
        return float(QB[next_key][best])

    if algo == "esarsa":
        # Expected SARSA: 실제로 뽑지 않고 ε-greedy 분포의 기댓값을 그대로 쓴다.
        probs = eps_greedy_probs(q_row, mask, eps)
        # 왜: 불법 칸의 Q는 -inf이므로 0.0 * -inf = nan이 된다. 합법 칸만 곱한다.
        return float(np.sum(probs[mask] * q_row[mask]))

    if algo == "sarsa":
        # SARSA: 같은 분포에서 실제로 한 개를 뽑아 그 값을 쓴다.
        # 왜 여기서 다시 뽑는가: Transition에는 '다음에 실제로 취한 행동'이
        #     들어 있지 않다(returns.py 참조). 전이 목록을 되짚어 찾을 수도 있지만,
        #     스플릿한 형제 손이 같은 키를 가지는 경우(8,8 → 둘 다 13)에
        #     엉뚱한 손의 행동을 집어올 수 있어 오히려 부정확하다.
        #     SARSA의 타깃이 요구하는 것은 a' ~ π(·|s') 표본 하나뿐이고
        #     우리 행동 정책이 바로 그 π이므로, 같은 분포에서 다시 뽑아도
        #     고정점은 똑같이 Q^π다. Expected SARSA가 이 기댓값 버전이라는 점도
        #     발표에서 두 줄로 설명된다.
        if rng is None:
            raise ValueError("sarsa는 다음 행동을 뽑아야 하므로 rng가 필요하다")
        probs = eps_greedy_probs(q_row, mask, eps)
        action = int(rng.choice(4, p=probs))
        return float(q_row[action])

    raise ValueError(f"모르는 TD 알고리즘이다: {algo!r} (가능: {TD_ALGOS})")


def _target(
    algo: str,
    tr: Transition,
    Q: np.ndarray,
    QB: np.ndarray | None,
    eps: float,
    rng: np.random.Generator | None = None,
) -> float:
    """전이 하나의 학습 타깃. 이 모듈에서 단 한 번만 정의된다.

    split이면 next_keys 2개의 값을 '더한다'(설계서 §3.4).
    """
    if algo == "mc":
        # 왜 맨 앞인가: 몬테카를로는 부트스트랩을 하지 않는다. compute_transitions가
        #     이미 '손 궤적 단위 first-visit'으로 걸러 mc_return에 크레딧을 넣어 두었다
        #     (스플릿이면 자손 손 결과의 합). 아래 TD 경로로 내려가면 종단 전이가
        #     reward를 쓰고 스플릿 전이가 예외를 던져 MC가 조용히 깨진다.
        return float(tr.mc_return)

    # 왜 terminal이 아니라 len(next_keys)로 분기하는가:
    #     A,A를 스플릿하면 자식이 결정 없이 끝나므로 returns.py가
    #     terminal=False인데 next_keys=()인 전이를 만든다(실측 전체 전이의 0.43%).
    #     terminal로 분기하면 이 0.43%에서 빈 튜플을 순회해 조용히 0을 더한다.
    if len(tr.next_keys) == 0:
        return float(tr.reward)

    total = float(tr.reward)
    for next_key in tr.next_keys:
        # 왜 더하는가: 스플릿은 손 하나를 손 둘로 만든다. 두 손의 순손익이 모두
        #     이 결정의 책임이므로 리턴은 두 후계자 가치의 '합'이다. 평균을 쓰면
        #     베팅이 2배가 된 사실이 사라져 "쪼개면 손해"라는 틀린 표가 나온다.
        #     next_keys 길이는 0(종단)/1(히트)/2(스플릿)뿐이라 이 루프가 셋을 다 덮는다.
        total += successor_value(algo, next_key, Q, QB, eps, rng)
    return total


class TabularAgent:
    """Q 테이블을 ε-greedy로 굴리고 Transition으로 갱신하는 학습자.

    다섯 알고리즘(mc/q/doubleq/sarsa/esarsa)이 이 클래스 하나를 쓴다.
    갈리는 곳은 observe() 안의 _target() 호출 한 줄뿐이다.
    """

    def __init__(self, rules: RuleSet, streams: Streams, *,
                 algo: str = "mc",
                 step: str = "sample_average",
                 alpha: float = 0.02,
                 eps0: float = 0.25,
                 eps_final: float = 0.02,
                 eps_decay_at: int = 5_000_000,
                 encoder: str = "s1") -> None:
        if algo not in ALGOS:
            raise ValueError(f"algo는 {ALGOS} 중 하나여야 한다: {algo!r}")
        if step not in STEPS:
            raise ValueError(f"step은 {STEPS} 중 하나여야 한다: {step!r}")
        if encoder not in ENCODERS:
            raise ValueError(
                f"encoder는 {ENCODERS} 중 하나여야 한다: {encoder!r} "
                "(s2는 계획서 ④ W9에서 추가한다)")
        if eps_decay_at < 1:
            raise ValueError(f"eps_decay_at은 1 이상이어야 한다: {eps_decay_at}")
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha는 0 초과 1 이하여야 한다: {alpha}")

        self.rules = rules
        self.algo = algo
        self.step = step
        self.alpha = float(alpha)
        self.eps0 = float(eps0)
        self.eps_final = float(eps_final)
        self.eps_decay_at = int(eps_decay_at)
        self.encoder = encoder

        # 왜 Streams를 통째로 보관하는가: 탐험·Double Q 동전 던지기·SARSA의 a' 추출은
        #     전부 정책 쪽 난수라 explore 스트림에서만 나와야 한다. 여기 한 줄로
        #     묶어 두면 "알고리즘이 난수를 몇 번 쓰든 deal 스트림의 카드 순서는
        #     그대로"라는 rng.py의 약속이 코드로 지켜진다.
        self.streams = streams
        self.Q: np.ndarray = new_q_table(rules, streams.init)
        self.N: np.ndarray = new_visit_table(rules)
        self.QB: np.ndarray | None = None
        self.NB: np.ndarray | None = None
        if algo == "doubleq":
            self.QB = new_q_table(rules, streams.init)
            self.NB = new_visit_table(rules)

        # 왜 t 하나만 두는가: 'episodes'와 't' 두 이름을 두면 한쪽만 올리는 버그가 난다.
        self.t = 0
        self.eps = float(eps0)

    # ── ε 스케줄 ──

    def current_eps(self) -> float:
        """지금 에피소드 수에 맞는 ε. eps0에서 eps_final까지 직선으로 내려온다."""
        if self.t >= self.eps_decay_at:
            return self.eps_final
        # 왜: 지수 감쇠 대신 직선이다. '몇 번째 에피소드에 ε가 얼마였나'를
        #     슬라이드에서 암산으로 읽을 수 있어야 하기 때문이다.
        #     train.config.ExperimentConfig.eps_at()이 같은 식을 쓴다.
        진행 = self.t / self.eps_decay_at
        return self.eps0 + (self.eps_final - self.eps0) * 진행

    # ── 값 읽기: doubleq만 표가 둘이므로 읽는 지점을 여기 하나로 모은다 ──

    def q_eval(self, key: StateKey) -> np.ndarray:
        """행동 선택과 보고에 쓸 상태 하나의 행동가치 4개."""
        if self.QB is None:
            return self.Q[key]
        # 왜: Double Q의 두 표는 각각 데이터의 절반만 본다. 둘의 평균을 쓰면
        #     잡음이 줄고, 표 하나만 보고 판단하는 것보다 정책이 안정된다.
        return 0.5 * (self.Q[key] + self.QB[key])

    def q_eval_table(self) -> np.ndarray:
        """Q_SHAPE 전체를 한 번에. 프레임 기록·스냅샷 저장이 이걸 쓴다."""
        if self.QB is None:
            return self.Q
        # 왜 -inf가 살아남는가: 불법 칸은 두 표 모두 -inf라 0.5*(-inf + -inf) = -inf다.
        #     저장한 npz의 '불법 행동은 -inf' 점검이 doubleq에서도 그대로 통과한다.
        return 0.5 * (self.Q + self.QB)

    def visit_table(self) -> np.ndarray:
        """방문 횟수 전체. doubleq는 갱신이 두 표에 나뉘므로 둘의 합이 실제 방문수다."""
        if self.NB is None:
            return self.N
        return self.N + self.NB

    def max_q(self, key: StateKey) -> float:
        """이 상태에서 합법 행동들의 Q 최댓값."""
        return float(np.max(np.where(legal_actions(key), self.q_eval(key), -np.inf)))

    # ── 행동 선택: 다섯 알고리즘이 글자 그대로 같은 함수를 쓴다 ──

    def act(self, key: StateKey, mask: np.ndarray, ctx: Ctx) -> int:
        """ε 확률로 합법 행동 중 하나를 무작위로, 아니면 Q가 가장 큰 행동을 고른다."""
        if self.streams.explore.random() < self.eps:
            합법번호 = np.flatnonzero(mask)
            뽑기 = int(self.streams.explore.integers(0, 합법번호.size))
            return int(합법번호[뽑기])
        # 왜: 불법 칸은 Q가 이미 -inf라 argmax가 고를 수 없지만, 마스크를 한 번 더
        #     씌워 둔다. 나중에 Q를 파일에서 읽어 올 때 -inf가 섞여 들어오지 않아도
        #     이 한 줄 덕분에 불법 행동이 새어 나가지 않는다.
        return int(np.argmax(np.where(mask, self.q_eval(key), -np.inf)))

    def greedy_full(self) -> np.ndarray:
        """REACHABLE_KEYS 순서의 결정적 정책 배열. dp.evaluate_policy가 먹는 형식이다."""
        q = np.where(LEGAL_ROWS, self.q_eval_table()[KEY_AXES], -np.inf)
        return np.argmax(q, axis=1).astype(np.int8)

    def eps_greedy_dist(self, eps: float) -> np.ndarray:
        """ε-greedy 행동정책을 확률표로 편다. dp.evaluate_stochastic가 먹는 형식이다."""
        eps = float(eps)
        pi = np.zeros((len(REACHABLE_KEYS), 4), dtype=np.float64)
        pi[LEGAL_ROWS] = 1.0
        pi = pi * (eps / N_LEGAL)[:, None]
        최선 = self.greedy_full().astype(np.intp)
        pi[np.arange(len(REACHABLE_KEYS)), 최선] += 1.0 - eps
        return pi

    # ── 학습: 알고리즘이 갈리는 곳은 _target() 호출 한 줄뿐이다 ──

    def _step_size(self, visits: np.ndarray, cell: tuple) -> float:
        """이 (s,a) 칸에 쓸 학습률. 다섯 알고리즘이 똑같이 이 함수를 쓴다."""
        if self.step == "sample_average":
            # 왜: 1/N은 MC에서 표본평균과 정확히 같아진다. TD에서는 타깃이
            #     비정상(non-stationary)이라 초기 나쁜 타깃이 오래 남는 단점이 있고,
            #     그래서 step="constant"가 대조 실험으로 남아 있다.
            return 1.0 / float(visits[cell])
        return float(self.alpha)

    def observe(self, trs: list[Transition]) -> None:
        """라운드 하나가 낳은 전이들로 Q를 갱신하고 에피소드를 한 칸 넘긴다."""
        for tr in trs:
            cell = tuple(tr.key) + (int(tr.action),)

            if self.algo == "doubleq" and self.streams.explore.random() < 0.5:
                # 왜: 매 갱신마다 동전을 던져 '갱신할 표'와 '값을 읽을 표'를 맞바꾼다.
                #     두 표가 같은 데이터로 학습되면 독립성이 깨져 편향이 되살아난다.
                q_table, n_table, other = self.QB, self.NB, self.Q
            else:
                q_table, n_table, other = self.Q, self.N, self.QB

            타깃 = _target(self.algo, tr, q_table, other, self.eps,
                          self.streams.explore)
            n_table[cell] += 1
            q_table[cell] += self._step_size(n_table, cell) * (타깃 - q_table[cell])

        # 왜: 전이 개수가 아니라 라운드 수로 센다. 딜러 블랙잭이면 전이가 0개인데
        #     그것도 한 에피소드이고, 그래야 ε 스케줄이 계획한 속도로 내려간다.
        self.t += 1
        self.eps = self.current_eps()
