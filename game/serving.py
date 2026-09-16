"""이 파일은 학습된 정책을 메모리에 올려 두고 상태마다 행동을 골라 준다.
입력: 레지스트리 표의 행과 npz 정책 파일.
출력: ServedModel 목록과 상태 하나에 대한 행동 번호.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from blackjack_rl.models.registry import load_policy
from blackjack_rl.rules import RULES_V1
from blackjack_rl.state import HIT, STAND, REACHABLE_KEYS, StateKey
from game.models import ModelRegistry

# 왜 미리 만들어 두는가: REACHABLE_KEYS는 610개 튜플이다. 결정마다 list.index()를
#   돌면 O(610)이지만 사전이면 O(1)이다. 값은 불변이므로 import 때 한 번 만든다.
KEY_INDEX: dict[StateKey, int] = {k: i for i, k in enumerate(REACHABLE_KEYS)}


@dataclass(frozen=True)
class ServedModel:
    """서빙 중인 모델 하나. 정책 배열 말고는 아무것도 들고 있지 않다."""

    id: int
    name: str
    family: str
    exact_ev: float
    policy: np.ndarray      # int8[610]


def model_action(model: ServedModel, key: StateKey, legal: np.ndarray) -> int:
    """이 모델이라면 이 상태에서 무엇을 고를지. 불법이면 합법으로 되돌아간다."""
    자리 = KEY_INDEX.get(key)
    if 자리 is None:
        # 왜 STAND인가: 모르는 자리에서 가장 덜 해로운 행동이고 언제나 합법이다.
        #   실측으로는 4,000판을 무작위로 돌려도 여기 오는 상태가 0개였지만,
        #   규칙이 바뀌면 생길 수 있으므로 죽지 않게 둔다.
        return STAND

    고른것 = int(model.policy[자리])
    if 0 <= 고른것 <= 3 and legal[고른것]:
        return 고른것
    # 왜 이 순서인가: 원래 의도가 더블이면 히트가, 스플릿이면 히트가 가장 가깝다.
    #   둘 다 막히면 STAND로 간다. 저장된 13개 모델은 여기 오지 않지만,
    #   사람의 버릇을 배우는 인간 모방 모델은 올 수 있다.
    if legal[HIT] and 고른것 in (2, 3):
        return HIT
    return STAND


class ModelStore:
    """레지스트리 표를 읽어 정책을 메모리에 올려 두는 곳. 프로세스마다 하나다."""

    def __init__(self) -> None:
        self._모델: dict[int, ServedModel] = {}

    def refresh(self, session: Session) -> int:
        """DB의 서빙 대상 모델을 전부 다시 읽는다. 읽은 개수를 돌려준다."""
        새것: dict[int, ServedModel] = {}
        행들 = session.scalars(
            select(ModelRegistry).where(ModelRegistry.is_serving.is_(True)))
        for 행 in 행들:
            # 왜 여기서 load_policy를 쓰는가: 규칙 지문과 내용 해시를 함께 본다.
            #   규칙이 다른 모델을 태연히 서빙하면 DP 정답과 EV 손실이 전부 틀린
            #   값으로 쌓인다. 예외를 잡지 않고 그대로 올린다.
            정책, _카드 = load_policy(행.artifact_path, rules=RULES_V1)
            새것[행.id] = ServedModel(id=행.id, name=행.name, family=행.family,
                                    exact_ev=float(행.exact_ev), policy=정책)
        self._모델 = 새것
        return len(새것)

    def get(self, model_id: int) -> ServedModel | None:
        """번호로 찾는다. 없으면 None."""
        return self._모델.get(model_id)

    def serving(self) -> list[ServedModel]:
        """서빙 중인 모델을 EV가 좋은 순으로 준다."""
        return sorted(self._모델.values(), key=lambda m: -m.exact_ev)

    def default(self) -> ServedModel | None:
        """상대를 고르지 않았을 때 쓸 모델. 가장 잘 두는 것을 준다."""
        후보 = self.serving()
        return 후보[0] if 후보 else None


# 왜 모듈 전역인가: 정책 13개 x 610바이트라 메모리가 거의 안 든다. 요청마다
#   파일을 다시 읽을 이유가 없고, 서버 기동 때 한 번 채운다.
store = ModelStore()
