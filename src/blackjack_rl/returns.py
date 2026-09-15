"""이 파일은 라운드 기록을 학습용 전이(Transition) 목록으로 바꾼다.
입력: env.play_round가 만든 RoundResult 하나.
출력: list[Transition] — MC·Q러닝·Double Q·SARSA·E-SARSA·DQN이 공통으로 먹는 유일한 중간 표현."""

from typing import NamedTuple

from blackjack_rl.env import RoundResult
from blackjack_rl.state import SPLIT, StateKey


class Transition(NamedTuple):
    """결정 하나. 여섯 알고리즘이 문자 그대로 이 타입만 먹는다."""

    key: StateKey
    action: int
    reward: float
    next_keys: tuple[StateKey, ...]   # 0개=종단 / 1개=hit / 2개=split
    mc_return: float
    terminal: bool


def _children_of(r: RoundResult, hand_index: int, step: int) -> list[int]:
    """hand_index번 손이 step번째 결정에서 SPLIT해 만든 자식 손들의 인덱스."""
    out = []
    for j, rec in enumerate(r.hands):
        if rec.parent == hand_index and rec.parent_step == step:
            out.append(j)
    return out


def compute_transitions(r: RoundResult) -> list[Transition]:
    """설계서 §3.4의 크레딧 할당 규칙을 그대로 옮긴 함수.

    - hit/stand/double 결정: mc_return = 그 손 자신의 result
    - split 결정:            mc_return = 그 split이 낳은 모든 자손 손의 result 합
    - first-visit은 '라운드'가 아니라 '손 궤적' 단위로 적용한다.
    """
    transitions: list[Transition] = []

    for i, rec in enumerate(r.hands):
        # 왜: seen을 손마다 새로 만드는 것이 first-visit의 범위를 '손 궤적'으로 고정하는
        #     장치다. 라운드 단위로 두면 스플릿한 둘째 손 표본이 통째로 버려진다.
        #     (한 손 안에서는 total이 단조 증가해 키가 겹칠 수 없으므로 실제로는 안 걸린다.)
        seen: set[StateKey] = set()

        for t, (key, action) in enumerate(rec.trajectory):
            if key in seen:
                continue
            seen.add(key)

            if t + 1 < len(rec.trajectory):
                # 히트해서 같은 손이 계속된다 → 후계자는 다음 결정 상태 하나
                다음_키 = rec.trajectory[t + 1][0]
                transitions.append(
                    Transition(key, action, 0.0, (다음_키,), rec.result, False)
                )

            elif action == SPLIT:
                즉시_보상 = 0.0
                자손_합 = 0.0
                다음_키들: list[StateKey] = []
                for j in _children_of(r, i, t):
                    child = r.hands[j]
                    자손_합 += child.subtree_result
                    if len(child.trajectory) > 0:
                        다음_키들.append(child.trajectory[0][0])
                    else:
                        # 왜: 에이스 스플릿 자식은 결정이 없어 부트스트랩할 상태가 없다.
                        #     그 결과는 즉시 보상으로 넣어야 TD 타깃이 맞는다.
                        즉시_보상 += child.subtree_result
                transitions.append(
                    Transition(key, action, 즉시_보상, tuple(다음_키들), 자손_합, False)
                )

            else:
                # STAND, DOUBLE, 그리고 버스트로 끝난 HIT — 전부 종단이다
                transitions.append(
                    Transition(key, action, rec.result, (), rec.result, True)
                )

    return transitions


def assert_invariants(r: RoundResult) -> None:
    """디버그용. 손 결과 합이 라운드 순손익과 같은지, subtree 합이 일관되는지 본다."""
    손합 = 0.0
    for rec in r.hands:
        손합 += rec.result
    if abs(손합 - r.net) > 1e-9:
        raise AssertionError(f"손 결과 합 {손합}이 라운드 순손익 {r.net}과 다르다")

    for i, rec in enumerate(r.hands):
        자식합 = 0.0
        for other in r.hands:
            if other.parent == i:
                자식합 += other.subtree_result
        기대 = rec.result + 자식합
        if abs(rec.subtree_result - 기대) > 1e-9:
            raise AssertionError(
                f"{i}번 손의 subtree_result {rec.subtree_result}가 기대값 {기대}와 다르다"
            )
