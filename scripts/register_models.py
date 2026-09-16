"""이 파일은 학습 산출물을 레지스트리 형식으로 바꾸고 DB 표에 등록한다.
입력: artifacts/ 의 강화학습 스냅샷과 models/ 의 정책 npz.
출력: models/rl_mc.npz 파일과 model_registry 표의 행들.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from blackjack_rl.dp.exact import evaluate_policy  # noqa: E402
from blackjack_rl.models.registry import ModelCard, load_policy, save_policy  # noqa: E402
from blackjack_rl.rules import RULES_V1  # noqa: E402
from game.db import create_schema, make_engine, make_session_factory  # noqa: E402
from game.models import ModelRegistry  # noqa: E402

RL_NAME = "rl_mc"


def file_sha256(path: Path) -> str:
    """파일 내용의 SHA-256. 레지스트리에 박아 두고 변조를 잡는다."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convert_rl_snapshot(artifacts_dir: Path, models_dir: Path) -> Path:
    """강화학습 스냅샷 중 에피소드가 가장 많은 것의 마지막 프레임을 정책으로 저장한다."""
    후보 = []
    for npz in sorted(Path(artifacts_dir).glob("*.npz")):
        meta_path = npz.with_suffix(".json")
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        후보.append((int(meta.get("n_episodes", 0)), npz, meta))
    if not 후보:
        raise FileNotFoundError(f"{artifacts_dir}에 강화학습 스냅샷이 없다")

    # 왜 최댓값인가: 계획서 ③에서 sorted(glob)[-1]이 200k짜리 시험 산출물을 집어
    #   기준선이 4.6배 틀어진 적이 있다. 이름 정렬은 학습량과 아무 상관이 없다.
    에피소드, npz, meta = max(후보, key=lambda t: t[0])

    with np.load(npz, allow_pickle=False) as z:
        정책 = np.ascontiguousarray(z["policy_full"][-1], dtype=np.int8)

    나온값 = float(evaluate_policy(정책, RULES_V1))
    출력 = Path(models_dir) / f"{RL_NAME}.npz"
    save_policy(출력, 정책, ModelCard(
        name=RL_NAME,
        family="rl",
        rules_fp=RULES_V1.fingerprint(),
        exact_ev=나온값,
        policy_sha256="",                 # save_policy가 내용으로 다시 채운다
        n_train_labels=0,                 # 강화학습은 정답 레이블을 한 장도 안 봤다
        test_acc=0.0,
        fit_seconds=float(meta.get("fit_seconds", 0.0)),
        n_params=int(정책.size),
        created_at=datetime.now(timezone.utc).isoformat(),
    ))
    print(f"  {npz.name} ({에피소드:,} 에피소드) -> {출력.name}  정확 EV {나온값:+.6f}")
    return 출력


def sync_registry(session: Session, models_dir: Path) -> int:
    """models/ 의 정책 파일을 전부 레지스트리 표에 맞춰 넣는다. 등록 개수를 돌려준다."""
    개수 = 0
    models_dir = Path(models_dir)
    for npz in sorted(models_dir.glob("*.npz")):
        try:
            _정책, 카드 = load_policy(npz, rules=RULES_V1)
        except Exception as 오류:
            # 왜 이름을 붙이는가: numpy의 "pickled data" 같은 원래 메시지에는 어느
            #   파일인지가 없다. 13개 중 무엇을 다시 만들어야 하는지 바로 보여야 한다.
            raise RuntimeError(f"{npz.name}을 레지스트리에 올리지 못했다: "
                               f"{type(오류).__name__}: {오류}") from 오류
        행 = session.scalar(select(ModelRegistry).where(ModelRegistry.name == 카드.name))
        값 = {
            "family": 카드.family,
            "rules_fp": 카드.rules_fp,
            # 왜 상대경로인가: 절대경로를 박으면 저장소 폴더를 옮기거나 윈도에서 등록한
            #   DB를 리눅스 컨테이너에서 열 때 모든 모델이 사라진다. POSIX 구분자로
            #   두어야 어느 OS의 ModelStore(models_dir)에서도 같은 파일을 가리킨다.
            "artifact_path": npz.relative_to(models_dir).as_posix(),
            "artifact_sha256": file_sha256(npz),
            "exact_ev": float(카드.exact_ev),
            "is_serving": True,
        }
        if 행 is None:
            session.add(ModelRegistry(name=카드.name, **값))
        else:
            # 왜 갱신인가: 모델을 다시 학습해 같은 이름으로 덮어쓸 수 있다. 그때
            #   해시와 EV가 따라가야 레지스트리가 거짓말을 안 한다.
            for 키, 값하나 in 값.items():
                setattr(행, 키, 값하나)
        개수 += 1
    session.commit()
    return 개수


def main() -> int:
    """스냅샷을 변환하고 레지스트리를 동기화한다."""
    print("강화학습 스냅샷 변환:")
    convert_rl_snapshot(ROOT / "artifacts", ROOT / "models")

    엔진 = make_engine()
    create_schema(엔진)
    with make_session_factory(엔진)() as 세션:
        등록 = sync_registry(세션, ROOT / "models")
    엔진.dispose()

    print(f"레지스트리에 {등록}개 모델 등록 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
