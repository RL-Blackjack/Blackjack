"""이 파일은 학습된 정책을 메모리에 올려 두고 상태마다 행동을 골라 준다.
입력: 레지스트리 표의 행과 npz 정책 파일.
출력: ServedModel 목록과 상태 하나에 대한 행동 번호.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from blackjack_rl.models.registry import load_policy
from blackjack_rl.rules import RULES_V1
from blackjack_rl.state import HIT, STAND, REACHABLE_KEYS, StateKey
from game.config import ROOT
from game.models import ModelRegistry

log = logging.getLogger("game.serving")

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


@dataclass(frozen=True)
class LoadFailure:
    """서빙에 올리지 못한 모델 하나. refresh가 무엇이 왜 빠졌는지 돌려줄 때 쓴다."""

    id: int
    name: str
    path: Path
    error: Exception


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

    def __init__(self, models_dir: Path | None = None) -> None:
        self._모델: dict[int, ServedModel] = {}
        # 왜 환경변수를 보는가: 배포 컨테이너는 모델 볼륨을 코드와 다른 곳에 붙일 수
        #   있다. 레지스트리에는 이 폴더 기준 상대경로만 있으므로 폴더만 바꾸면 된다.
        기본 = os.environ.get("BJ_MODELS_DIR") or ROOT / "models"
        self.models_dir = Path(models_dir if models_dir is not None else 기본)
        # 왜 따로 두는가: "비어 있으면 다시 읽기"로 판단하면 전부 깨졌을 때 요청마다
        #   파일을 다시 열고 ERROR를 쏟는다. 한 번 시도했으면 명시적 refresh까지 쉰다.
        self.loaded: bool = False

    def refresh(self, session: Session) -> tuple[int, list[LoadFailure]]:
        """DB의 서빙 대상 모델을 전부 다시 읽는다. (올라온 수, 실패 목록)을 돌려준다."""
        새것: dict[int, ServedModel] = {}
        실패: list[LoadFailure] = []
        행들 = list(session.scalars(
            select(ModelRegistry).where(ModelRegistry.is_serving.is_(True))))
        for 행 in 행들:
            # 왜 이렇게 잇는가: 상대경로 행은 models_dir 기준으로 풀린다. 옛 DB의
            #   절대경로 행은 pathlib이 오른쪽 절대경로를 그대로 돌려주므로 호환된다.
            경로 = self.models_dir / 행.artifact_path
            try:
                # 왜 load_policy인가: 규칙 지문과 내용 해시를 함께 본다. 규칙이 다른
                #   모델을 태연히 서빙하면 DP 정답과 EV 손실이 전부 틀린 값으로 쌓인다.
                정책, _카드 = load_policy(경로, rules=RULES_V1)
            except Exception as 오류:
                # 왜 그 모델만 빼는가: 파일 하나가 손상되거나 규칙이 달라도 나머지
                #   모델까지 버리면 게임 API 전체가 500이 된다(실측). 조용히 넘어가지
                #   않도록 이름·번호·경로·원인을 ERROR로 남기고 실패 목록에 담는다.
                log.error("모델 %s(id=%s, 경로 %s)을 서빙에서 뺀다: %s: %s",
                          행.name, 행.id, 경로, type(오류).__name__, 오류)
                실패.append(LoadFailure(id=행.id, name=행.name, path=경로, error=오류))
                continue
            새것[행.id] = ServedModel(id=행.id, name=행.name, family=행.family,
                                    exact_ev=float(행.exact_ev), policy=정책)
        self._모델 = 새것
        self.loaded = True

        if 행들 and not 새것:
            # 왜 예외인가: 행이 있는데 하나도 못 올리면 서버가 "상대 없음"으로 조용히
            #   뜬다. 빈 표(아직 등록 전)는 실패가 아니므로 여기 오지 않는다.
            이름들 = ", ".join(f.name for f in 실패)
            raise RuntimeError(
                f"서빙 대상 모델 {len(행들)}개를 하나도 올리지 못했다: {이름들}")
        return len(새것), 실패

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
