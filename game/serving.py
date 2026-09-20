"""이 파일은 학습된 정책을 메모리에 올려 두고 상태마다 행동을 골라 준다.
입력: 레지스트리 표의 행과 npz 정책 파일.
출력: ServedModel 목록과 상태 하나에 대한 행동 번호.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from blackjack_rl.models.registry import ModelRulesMismatch, load_policy
from blackjack_rl.rules import RULES_V1
from blackjack_rl.state import HIT, STAND, REACHABLE_KEYS, StateKey
from game.config import ROOT
from game.models import ModelRegistry

log = logging.getLogger("game.serving")

# 왜 미리 만들어 두는가: REACHABLE_KEYS는 610개 튜플이다. 결정마다 list.index()를
#   돌면 O(610)이지만 사전이면 O(1)이다. 값은 불변이므로 import 때 한 번 만든다.
KEY_INDEX: dict[StateKey, int] = {k: i for i, k in enumerate(REACHABLE_KEYS)}


def file_sha256(path: Path | str) -> str:
    """파일 내용의 SHA-256. 등록 때 레지스트리에 박고 서빙 때 대조한다."""
    # 왜 여기 두는가: 등록 스크립트와 게임 서버가 같은 함수로 재야 둘이 어긋나지 않는다.
    #   스크립트에 두면 게임 서버가 scripts/를 import해야 하므로 서버 쪽에 둔다.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ModelArtifactMismatch(Exception):
    """레지스트리 행과 실제 파일이 가리키는 모델이 다를 때 난다(해시·이름)."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class ServedModel:
    """서빙 중인 모델 하나. 정책 배열과 그 배열을 읽어 온 파일의 해시를 든다."""

    id: int
    name: str
    family: str
    exact_ev: float
    policy: np.ndarray      # int8[610]
    # 왜 해시를 싣는가: 레지스트리 행은 같은 이름으로 덮어쓰일 수 있다. 게임 행에
    #   이 값을 박아야 decisions.ai_action이 어느 파일에서 나왔는지 나중에 가린다.
    artifact_sha256: str


@dataclass(frozen=True)
class LoadFailure:
    """서빙에 올리지 못한 모델 하나. refresh가 무엇이 왜 빠졌는지 돌려줄 때 쓴다."""

    id: int
    name: str
    path: Path
    reason: str             # sha_mismatch / name_mismatch / rules_mismatch / missing / load_error
    error: Exception


def _사유(오류: Exception) -> str:
    """실패 목록과 로그에 남길 짧은 사유 코드."""
    if isinstance(오류, ModelArtifactMismatch):
        return 오류.reason
    if isinstance(오류, ModelRulesMismatch):
        return "rules_mismatch"
    if isinstance(오류, FileNotFoundError):
        return "missing"
    return "load_error"


def _올리기(행: ModelRegistry, 경로: Path) -> ServedModel:
    """레지스트리 행 하나를 파일과 대조해 올린다. 어긋나면 예외를 낸다."""
    # 왜 해시를 먼저 보는가: load_policy는 파일 안의 카드와 배열이 서로 맞는지만 본다.
    #   다른 모델의 멀쩡한 파일로 통째로 바꿔치기하면 그대로 통과해 서빙됐다(누락
    #   탐색에서 재현). 등록 때 박은 해시와 대조해야 등록된 그 파일인지 안다.
    실제 = file_sha256(경로)
    if 실제 != 행.artifact_sha256:
        raise ModelArtifactMismatch(
            "sha_mismatch",
            f"파일 해시 {실제[:12]}가 레지스트리의 {행.artifact_sha256[:12]}와 다르다")
    # 왜 load_policy인가: 규칙 지문과 내용 해시를 함께 본다. 규칙이 다른
    #   모델을 태연히 서빙하면 DP 정답과 EV 손실이 전부 틀린 값으로 쌓인다.
    정책, 카드 = load_policy(경로, rules=RULES_V1)
    if 카드.name != 행.name:
        # 왜 이름도 보는가: 해시는 등록한 순간의 파일을 믿을 뿐이다. 파일을 잘못
        #   짝지어 등록하면 해시는 맞아도 행의 이름과 다른 모델이 서빙된다.
        raise ModelArtifactMismatch(
            "name_mismatch", f"카드 이름 {카드.name!r}가 레지스트리 이름과 다르다")
    return ServedModel(id=행.id, name=행.name, family=행.family,
                       exact_ev=float(행.exact_ev), policy=정책,
                       artifact_sha256=행.artifact_sha256)


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
        # 왜 지문을 들고 있는가: 레지스트리가 바뀌었는지 파일을 열지 않고 알아내려고.
        #   None이면 아직 읽지 않았거나 reset된 것이라 어떤 지문과도 다르다.
        self._지문: tuple | None = None
        # 왜 잠금인가: 동기 핸들러가 스레드 풀에서 돈다. 두 refresh가 엇갈려 새 지문에
        #   옛 모델이 붙으면 다음 레지스트리 변경 전까지 낡은 채 남는다.
        self._잠금 = threading.Lock()

    def registry_fingerprint(self, session: Session) -> tuple:
        """레지스트리의 (id, 파일 해시, 서빙 여부)를 id 순으로 모은 튜플."""
        # 왜 이 세 칸인가: 재학습 덮어쓰기는 해시를, 서빙 끄기는 is_serving을, 등록·
        #   삭제는 id 목록을 바꾼다. 모델 13행짜리 조회라 요청마다 불러도 싸다.
        행들 = session.execute(
            select(ModelRegistry.id, ModelRegistry.artifact_sha256, ModelRegistry.is_serving)
            .order_by(ModelRegistry.id))
        return tuple((int(i), str(h), bool(s)) for i, h, s in 행들)

    def ensure_fresh(self, session: Session) -> ModelStore:
        """레지스트리 지문이 들고 있는 것과 다를 때만 다시 읽는다. 자기 자신을 돌려준다."""
        # 왜 지문이 같으면 넘어가는가: 요청마다 부르는 자리다. 전부 실패했던 경우도
        #   지문이 그대로면 다시 읽지 않아, 요청마다 파일을 열고 ERROR를 쏟지 않는다.
        if self.registry_fingerprint(session) != self._지문:
            self.refresh(session)
        return self

    def reset(self) -> None:
        """올린 모델과 지문을 비운다. 다음 ensure_fresh는 반드시 다시 읽는다."""
        # 왜 필요한가: store는 모듈 전역이라 pytest 한 프로세스에서 DB를 갈아끼워도
        #   남는다. API 픽스처가 시작 때 비워야 앞 테스트의 모델·지문이 새지 않는다.
        with self._잠금:
            self._모델 = {}
            self._지문 = None
            self.loaded = False

    def refresh(self, session: Session) -> tuple[int, list[LoadFailure]]:
        """DB의 서빙 대상 모델을 전부 다시 읽는다. (올라온 수, 실패 목록)을 돌려준다."""
        with self._잠금:
            # 왜 지문을 행보다 먼저 읽는가: 둘 사이에 레지스트리가 바뀌면 옛 지문에 새
            #   모델이 붙어 다음 ensure_fresh가 한 번 더 읽을 뿐이다. 거꾸로 읽으면 새
            #   지문에 옛 모델이 붙어 다음 변경 전까지 아무도 다시 읽지 않는다.
            지문 = self.registry_fingerprint(session)
            새것: dict[int, ServedModel] = {}
            실패: list[LoadFailure] = []
            행들 = list(session.scalars(
                select(ModelRegistry).where(ModelRegistry.is_serving.is_(True))
                .order_by(ModelRegistry.id)
                # 왜 populate_existing인가: 운영 세션은 expire_on_commit=False라 한 번
                #   읽은 행이 identity map에 남는다. 그대로면 재학습으로 바뀐 DB의 해시
                #   대신 옛 해시로 파일을 대조해, 멀쩡한 모델이 sha_mismatch로 빠진다.
                .execution_options(populate_existing=True)))
            for 행 in 행들:
                # 왜 이렇게 잇는가: 상대경로 행은 models_dir 기준으로 풀린다. 옛 DB의
                #   절대경로 행은 pathlib이 오른쪽 절대경로를 그대로 돌려주므로 호환된다.
                경로 = self.models_dir / 행.artifact_path
                try:
                    새것[행.id] = _올리기(행, 경로)
                except Exception as 오류:
                    # 왜 그 모델만 빼는가: 파일 하나가 손상·변조되거나 규칙이 달라도
                    #   나머지까지 버리면 게임 API 전체가 500이 된다(실측). 조용히 넘어가지
                    #   않도록 이름·번호·경로·사유·원인을 ERROR로 남기고 실패 목록에 담는다.
                    사유 = _사유(오류)
                    log.error("모델 %s(id=%s, 경로 %s)을 서빙에서 뺀다 [%s]: %s: %s",
                              행.name, 행.id, 경로, 사유, type(오류).__name__, 오류)
                    실패.append(LoadFailure(id=행.id, name=행.name, path=경로,
                                            reason=사유, error=오류))
            self._모델 = 새것
            self._지문 = 지문
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
#   파일을 다시 읽을 이유가 없고, 기동 때 채운 뒤 레지스트리가 바뀔 때만 다시 읽는다.
store = ModelStore()
