"""이 파일은 저장된 스냅샷의 해시를 다시 계산해 대조하고 재현성을 그 자리에서 실증한다
입력: --run 스냅샷 npz 경로, --expect-sha 기대 해시, --rerun 부분 재실행 플래그
출력: 표준출력 점검 결과 마크다운 표와 종료코드(0=전부 통과, 1=실패 있음)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

# 왜: 평가자가 저장소를 받아 바로 한 줄 치는 스크립트다. 설치도 PYTHONPATH도 없이 돌아야 한다.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from blackjack_rl.rules import RULES_V1  # noqa: E402
from blackjack_rl.state import LEGAL, Q_SHAPE  # noqa: E402
from blackjack_rl.train.config import ExperimentConfig  # noqa: E402
from blackjack_rl.train.runner import run  # noqa: E402

# 왜: 계획서 ①에서 DP로 직접 푼 값이다. 학습 정책이 이보다 좋게 나오면 버그다.
DP_OPTIMAL_EV = -0.005108
SANITY_FLOOR = -0.003


def sha256_of_array(a: np.ndarray) -> str:
    """배열 하나의 sha256을 돌려준다."""
    # 왜: dtype과 메모리 배치를 float32 C-연속으로 통일한 뒤에 해시한다.
    #     같은 Q를 float64로 들고 있다는 이유만으로 해시가 갈라지면
    #     "재현 실패"가 아니라 "저장 형식 차이"인데 구분이 안 된다.
    배열 = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    return hashlib.sha256(배열.tobytes()).hexdigest()


def verify_npz(path: Path) -> dict:
    """스냅샷 하나를 일곱 항목으로 점검한다."""
    with np.load(path, allow_pickle=False) as z:
        q = np.asarray(z["q_final"])
        ev_final = float(np.asarray(z["ev_greedy"])[-1])
        meta = json.loads(str(z["meta"]))

    실제해시 = sha256_of_array(q)
    기록해시 = str(meta.get("q_sha256", ""))

    점검: list[list] = [
        ["q_final sha256 == meta.q_sha256", 실제해시 == 기록해시,
         f"{실제해시[:16]} vs {기록해시[:16]}"],
        ["q_final 모양", tuple(q.shape) == tuple(Q_SHAPE), str(tuple(q.shape))],
        ["q_final에 NaN 없음", not bool(np.isnan(q).any()), ""],
        ["불법 행동이 -inf로 남아 있음", bool(np.isneginf(q[~LEGAL]).all()),
         f"{int((~LEGAL).sum())}칸"],
        ["최종 EV가 DP 상한을 넘지 않음", ev_final <= DP_OPTIMAL_EV + 1e-6,
         f"{ev_final:+.6f}"],
        [f"sanity 통과(EV <= {SANITY_FLOOR})", ev_final <= SANITY_FLOOR,
         f"{ev_final:+.6f}"],
        ["규칙 지문 == 현재 RULES_V1",
         str(meta.get("rules_fp", "")) == RULES_V1.fingerprint(),
         str(meta.get("rules_fp", ""))],
    ]
    return {
        "path": str(path),
        "q_sha256": 실제해시,
        "checks": 점검,
        "ok": all(통과 for _이름, 통과, _값 in 점검),
    }


def config_from_meta(meta: dict) -> ExperimentConfig:
    """스냅샷 meta에서 학습 설정을 통째로 되살린다."""
    # 왜 config 블록인가: train/runner.py가 cfg.to_json() 전문을 meta["config"]에
    #   넣는다. top-level에는 표를 만들 때 쓰는 요약 키만 있다. 읽는 쪽과 쓰는 쪽을
    #   한 벌로 맞춰 둬야 실제 산출물에 --rerun을 걸었을 때 KeyError가 안 난다.
    #   (구 스키마 호환: config 블록이 없으면 meta 자체를 본다.)
    본문 = meta.get("config", meta)
    if str(본문.get("rules_fp", "")) != RULES_V1.fingerprint():
        # 왜: 규칙이 다른 스냅샷을 현재 규칙으로 재실행하면 '재현 실패'로 보이지만
        #     실제로는 다른 게임이다. 조용히 섞이지 않게 여기서 막는다.
        raise ValueError(
            f"스냅샷의 규칙 지문 {본문.get('rules_fp')}가 현재 RULES_V1과 다르다")
    return ExperimentConfig(
        name=str(본문["name"]),
        algo=str(본문["algo"]),
        rules=RULES_V1,
        seed=int(본문["seed"]),
        n_episodes=int(본문["n_episodes"]),
        start_dist=str(본문["start_dist"]),
        step=str(본문["step"]),
        alpha=float(본문["alpha"]),
        eps0=float(본문["eps0"]),
        eps_final=float(본문["eps_final"]),
        eps_decay_at=int(본문["eps_decay_at"]),
        encoder=str(본문["encoder"]),
        n_frames=int(본문["n_frames"]),
    )


def rerun_twice(cfg: ExperimentConfig, n_episodes: int) -> dict:
    """같은 설정을 두 번 돌려 q_final 해시가 같은지 그 자리에서 보인다."""
    # 왜: 전체 재실행은 몇 분이 걸리고 발표 중에 할 수 없다. 대신 같은 시드로
    #     짧게 두 번 돌려 '결정론' 자체를 즉석에서 실증한다. 실측 59,919 eps/s라
    #     10만 에피소드 두 번이면 4초 안쪽이다.
    작은설정 = replace(cfg, n_episodes=int(n_episodes), n_frames=4)
    a = sha256_of_array(run(작은설정).q_final)
    b = sha256_of_array(run(작은설정).q_final)
    return {"n_episodes": int(n_episodes), "sha_a": a, "sha_b": b, "ok": a == b}


def render_report(report: dict) -> str:
    """점검 결과를 마크다운 표로 만든다."""
    줄들 = [f"## 스냅샷 점검: {report['path']}", "",
            "| 점검 항목 | 결과 | 값 |", "|---|---|---|"]
    for 이름, 통과, 값 in report["checks"]:
        줄들.append(f"| {이름} | {'통과' if 통과 else '실패'} | {값} |")
    줄들 += ["", f"q_final sha256: {report['q_sha256']}"]
    return "\n".join(줄들)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="스냅샷 해시 재검증과 부분 재실행")
    p.add_argument("--run", required=True, help="점검할 스냅샷 npz 경로")
    p.add_argument("--expect-sha", default=None, help="README에 적어 둔 기대 해시")
    p.add_argument("--rerun", action="store_true",
                   help="같은 시드로 짧게 두 번 재실행해 결정론을 실증한다")
    p.add_argument("--rerun-episodes", type=int, default=100_000)
    args = p.parse_args(argv)

    path = Path(args.run)
    보고 = verify_npz(path)
    print(render_report(보고))
    통과 = bool(보고["ok"])

    if args.expect_sha is not None:
        맞음 = 보고["q_sha256"] == args.expect_sha
        print(f"\n기대 해시 대조: {'통과' if 맞음 else '실패'}")
        통과 = 통과 and 맞음

    if args.rerun:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
        결과 = rerun_twice(config_from_meta(meta), args.rerun_episodes)
        print(f"\n{결과['n_episodes']:,} 에피소드 재실행 2회")
        print(f"  A {결과['sha_a'][:16]}")
        print(f"  B {결과['sha_b'][:16]}")
        print(f"  같은 시드 -> 같은 해시: {'통과' if 결과['ok'] else '실패'}")
        통과 = 통과 and 결과["ok"]

    print(f"\n최종: {'전부 통과' if 통과 else '실패 있음'}")
    return 0 if 통과 else 1


if __name__ == "__main__":
    raise SystemExit(main())
