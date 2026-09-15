"""이 파일은 학습 산출물 한 벌을 npz 파일 하나로 저장하고 다시 읽는다.
입력: RunArtifact(배열 13개 + meta 딕셔너리) 또는 저장된 .npz 경로.
출력: 저장 시 q_final의 sha256 문자열, 읽기 시 RunArtifact."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from blackjack_rl.rng import STREAM_NAMES, Streams
from blackjack_rl.train.config import LOG_SCHEDULE_START, ExperimentConfig

# 설계서 §4.1의 키 목록에 chart_notation을 더한 것. 저장과 읽기가 이 튜플 하나만 본다.
# 왜 chart_notation이 필요한가: chart_action만으로는 'D'와 'Ds'가 둘 다 action=2라
#   구분이 안 되어, 저장된 npz만으로는 GIF에 넣을 ChartTable을 복원할 수 없다.
ARRAY_FIELDS: tuple[str, ...] = (
    "episodes",
    "policy_full",
    "chart_action",
    "chart_notation",
    "chart_margin",
    "chart_visits",
    "ev_greedy",
    "ev_behavior",
    "agree_a",
    "agree_b",
    "maxq_minus_vstar",
    "q_final",
    "n_final",
)

# 왜 parquet을 만들지 않는가: 곡선에 들어갈 숫자(episodes, ev_greedy, ev_behavior,
#   agree_a/b, maxq_minus_vstar)는 이미 이 npz의 1차원 배열이다. 같은 숫자를 두 번
#   저장하려고 pyarrow를 새로 깔 이유가 없다. 실측: 난수 최악 경우 200프레임이
#   압축 후 841,441 B로 설계서가 요구한 1MB 미만을 npz 하나로 만족한다.


@dataclass
class RunArtifact:
    """학습 한 번이 남기는 전부. 웹앱·노트북·GIF가 이 객체 하나만 읽는다."""

    episodes: np.ndarray          # int64[F]
    policy_full: np.ndarray       # int8[F, 610]
    chart_action: np.ndarray      # int8[F, 36, 10]
    chart_notation: np.ndarray    # '<U2'[F, 36, 10]
    chart_margin: np.ndarray      # float32[F, 36, 10]
    chart_visits: np.ndarray      # int32[F, 36, 10]
    ev_greedy: np.ndarray         # float32[F]
    ev_behavior: np.ndarray       # float32[F]
    agree_a: np.ndarray           # float32[F]
    agree_b: np.ndarray           # float32[F]
    maxq_minus_vstar: np.ndarray  # float32[F]
    q_final: np.ndarray           # float32[22,2,12,2,2,4,4]
    n_final: np.ndarray           # int32[22,2,12,2,2,4,4]
    meta: dict


def log_schedule(n: int, frames: int = 200) -> np.ndarray:
    """1000부터 n까지 로그 간격으로 frames개 지점을 찍는다."""
    if frames < 2:
        raise ValueError(f"frames는 2 이상이어야 한다: {frames}")
    최소 = LOG_SCHEDULE_START + frames
    if n < 최소:
        raise ValueError(f"n은 {최소} 이상이어야 한다: {n}")

    지점 = np.rint(
        np.geomspace(float(LOG_SCHEDULE_START), float(n), frames)).astype(np.int64)

    # 왜: 앞쪽 로그 간격이 1보다 좁아지면 반올림 때문에 같은 에피소드가 두 번 나온다.
    #     프레임 수는 설계서상 고정이라 줄일 수 없으므로 한 칸씩 밀어 올린다.
    #     n >= 1000 + frames를 위에서 강제했으므로 밀어 올려도 n을 넘지 않는다.
    for i in range(1, frames):
        if 지점[i] <= 지점[i - 1]:
            지점[i] = 지점[i - 1] + 1

    # 왜: 마지막 프레임은 반드시 '학습이 끝난 시점'이어야 한다(반올림 오차 제거).
    지점[-1] = int(n)
    return 지점


def q_sha256(q: np.ndarray) -> str:
    """저장되는 float32 Q 배열의 sha256. 재현성 검증의 기준값이다."""
    배열 = np.ascontiguousarray(q, dtype=np.float32)
    return hashlib.sha256(배열.tobytes()).hexdigest()


def git_commit_sha() -> str:
    """현재 저장소의 커밋 SHA. 못 읽으면 'unknown'."""
    저장소 = Path(__file__).resolve().parents[3]
    try:
        결과 = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=저장소, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if 결과.returncode != 0:
        return "unknown"
    return 결과.stdout.strip()


def seed_record(streams: Streams, master_seed: int) -> dict:
    """마스터 시드와 갈라진 네 스트림의 시드를 전부 받아 적는다."""
    기록: dict = {"master": int(master_seed)}
    for 이름 in STREAM_NAMES:
        시드열 = getattr(streams, 이름).bit_generator.seed_seq
        기록[이름] = {
            "entropy": int(시드열.entropy),
            "spawn_key": [int(x) for x in 시드열.spawn_key],
        }
    return 기록


def artifact_path(cfg: ExperimentConfig, root: Path | str = Path("artifacts")) -> Path:
    """artifacts/{name}__{rules_fp}__{cfg_fp}__seedNN.npz"""
    return Path(root) / f"{cfg.artifact_stem()}.npz"


def config_path(cfg: ExperimentConfig, root: Path | str = Path("artifacts")) -> Path:
    """npz 옆에 나란히 놓이는 설정 덤프 경로."""
    return Path(root) / f"{cfg.artifact_stem()}.json"


def save_run(path: Path | str, art: RunArtifact) -> str:
    """산출물을 압축 npz 하나로 저장하고 q_final의 sha256을 돌려준다."""
    경로 = Path(path)
    # 왜: np.savez_compressed는 확장자가 없으면 제멋대로 .npz를 붙인다.
    #     그러면 호출자가 들고 있는 경로와 실제 파일 이름이 달라져 load_run이 실패한다.
    if 경로.suffix != ".npz":
        raise ValueError(f"저장 경로는 .npz로 끝나야 한다: {경로}")
    경로.parent.mkdir(parents=True, exist_ok=True)

    art.meta["q_sha256"] = q_sha256(art.q_final)
    배열들 = {이름: getattr(art, 이름) for 이름 in ARRAY_FIELDS}
    메타문자열 = json.dumps(art.meta, ensure_ascii=False, sort_keys=True)
    # 왜: meta를 0차원 문자열 배열로 넣으면 pickle 없이 npz 한 파일에 같이 들어간다.
    #     '<U2' 표기 배열도 object가 아니라 고정폭 유니코드라 같은 규칙으로 들어간다.
    np.savez_compressed(경로, meta=np.array(메타문자열), **배열들)
    return art.meta["q_sha256"]


def load_run(path: Path | str) -> RunArtifact:
    """저장된 npz를 RunArtifact로 되읽는다."""
    경로 = Path(path)
    # 왜: allow_pickle=False가 설계서 §8.1의 'pickle 금지' 조항 그 자체다.
    #     남이 만든 npz를 열어도 임의 코드가 실행되지 않는다.
    with np.load(경로, allow_pickle=False) as z:
        배열들 = {이름: z[이름] for 이름 in ARRAY_FIELDS}
        메타 = json.loads(str(z["meta"]))
    return RunArtifact(meta=메타, **배열들)
