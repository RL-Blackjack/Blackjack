"""이 파일은 학습된 정책을 파일로 저장하고 규칙·해시를 검증하며 되읽는다.
입력: 정책 배열(int8[610])과 모델 카드.
출력: npz 파일과 JSON 카드, 그리고 검증을 통과한 (정책, 카드).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from blackjack_rl.rules import RuleSet

N_STATES: int = 610
VALID_ACTIONS: frozenset[int] = frozenset({0, 1, 2, 3})


class ModelRulesMismatch(Exception):
    """저장된 모델의 규칙 지문이 지금 규칙과 다를 때 난다."""


@dataclass(frozen=True)
class ModelCard:
    """모델 하나를 설명하는 메타데이터. 게임 서버가 이것만 보고 고른다."""

    name: str
    family: str          # "rl" / "supervised" / "imitation"
    rules_fp: str
    exact_ev: float
    policy_sha256: str
    n_train_labels: int
    test_acc: float
    fit_seconds: float
    n_params: int
    created_at: str


def policy_sha256(policy_full: np.ndarray) -> str:
    """정책 배열의 내용 해시. 파일이 변조되면 달라진다."""
    # 왜 tobytes인가: 같은 값이면 항상 같은 바이트열이 나와야 한다. dtype을
    #   int8로 고정했으므로 플랫폼이 달라도 결과가 같다.
    바이트 = np.ascontiguousarray(policy_full, dtype=np.int8).tobytes()
    return hashlib.sha256(바이트).hexdigest()


def _검사(policy_full: np.ndarray) -> np.ndarray:
    배열 = np.ascontiguousarray(policy_full, dtype=np.int8)
    if 배열.shape != (N_STATES,):
        raise ValueError(f"정책은 ({N_STATES},) 모양이어야 한다: {배열.shape}")
    쓰인값 = {int(v) for v in np.unique(배열)}
    if not 쓰인값 <= VALID_ACTIONS:
        raise ValueError(f"정책에 알 수 없는 행동이 있다: {sorted(쓰인값 - VALID_ACTIONS)}")
    return 배열


def save_policy(path: Path | str, policy_full: np.ndarray, card: ModelCard) -> Path:
    """정책을 npz로, 카드를 같은 이름의 json으로 저장한다."""
    경로 = Path(path)
    배열 = _검사(policy_full)
    카드 = dataclasses.replace(card, policy_sha256=policy_sha256(배열))
    본문 = json.dumps(dataclasses.asdict(카드), ensure_ascii=False)

    경로.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(경로, policy=배열, card=본문)
    경로.with_suffix(".json").write_text(
        json.dumps(dataclasses.asdict(카드), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return 경로


def load_policy(path: Path | str, *, rules: RuleSet) -> tuple[np.ndarray, ModelCard]:
    """정책을 되읽으며 규칙 지문과 내용 해시를 함께 검증한다."""
    with np.load(Path(path), allow_pickle=False) as z:
        배열 = np.ascontiguousarray(z["policy"], dtype=np.int8)
        카드 = ModelCard(**json.loads(str(z["card"])))

    if 카드.rules_fp != rules.fingerprint():
        raise ModelRulesMismatch(
            f"모델은 규칙 {카드.rules_fp}로 학습됐는데 지금 규칙은 "
            f"{rules.fingerprint()}다. 규칙이 다른 모델은 쓸 수 없다.")

    실제해시 = policy_sha256(배열)
    if 실제해시 != 카드.policy_sha256:
        raise ValueError(
            f"정책 내용이 카드의 해시와 다르다(파일 변조 또는 손상). "
            f"카드 {카드.policy_sha256[:12]} vs 실제 {실제해시[:12]}")

    return 배열, 카드
