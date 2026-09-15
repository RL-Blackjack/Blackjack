"""이 파일은 저장된 학습 스냅샷들을 모아 최종 평가표 한 장을 만든다
입력: --artifacts 스냅샷 폴더, --target-ev 목표 EV, --out 리포트 JSON 경로
출력: reports/final_table.json 파일 1개와 표준출력 마크다운 표 2장
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

# 왜: 맨손 실행에서도 src/를 찾게 한다(train.py와 같은 이유).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from blackjack_rl.eval.simulate import baseline_evs, sanity_alarm  # noqa: E402
from blackjack_rl.rules import RULES_V1  # noqa: E402
from blackjack_rl.state import (Q_SHAPE, REACHABLE_KEYS,  # noqa: E402
                                legal_actions)

DEFAULT_ARTIFACTS = ROOT / "artifacts"
DEFAULT_OUT = ROOT / "reports" / "final_table.json"

# 왜: DP 최적 EV -0.005108보다 0.005 나쁜 지점을 '목표'로 잡는다. 이보다 촘촘한
#     목표는 통계적 동점 칸들에 걸려 도달 여부가 시드 운에 좌우된다.
DEFAULT_TARGET_EV = -0.010

# 왜 세어서 쓰는가: 설계서 §8.3이 '파라미터 수'와 '메모리' 열을 요구한다.
#     표 기반 학습자의 파라미터는 '도달 가능한 합법 (s,a) 칸'이고 실측 1,680개다.
#     메모리는 Q_SHAPE(33,792칸) x float64 = 270,336 B다. 상수로 박지 않고 센다.
N_PARAMS_TABULAR: int = sum(int(legal_actions(k).sum()) for k in REACHABLE_KEYS)
Q_BYTES_FLOAT64: int = int(np.prod(Q_SHAPE)) * 8


def params_and_memory(algo: str) -> tuple[int, int]:
    """알고리즘별 파라미터 수와 학습 중 Q 테이블 바이트."""
    # 왜 doubleq만 두 배인가: 표를 둘 들고 번갈아 갱신하는 것이 그 알고리즘의 정의다.
    배수 = 2 if algo == "doubleq" else 1
    return N_PARAMS_TABULAR * 배수, Q_BYTES_FLOAT64 * 배수


class RunRow(NamedTuple):
    """스냅샷 하나에서 뽑아낸 평가표 한 줄."""

    path: str
    name: str
    algo: str
    seed: int
    n_episodes: int
    final_ev: float
    episodes_to_target: int | None
    wall_sec: float


def read_run_row(path: Path, target_ev: float) -> RunRow:
    """npz 하나를 읽어 최종 EV와 목표 도달 에피소드를 뽑는다."""
    with np.load(path, allow_pickle=False) as z:
        episodes = np.asarray(z["episodes"], dtype=np.int64)
        ev = np.asarray(z["ev_greedy"], dtype=np.float64)
        meta = json.loads(str(z["meta"]))

    도달 = None
    for i in range(len(ev)):
        if ev[i] >= target_ev:
            도달 = int(episodes[i])
            break

    # 왜 top-level을 읽는가: train/runner.py의 meta는 name/algo/seed/n_episodes/
    #   wall_sec 를 top-level에도 쓴다. 설정 전체를 되살리는 쪽(verify_release.py)만
    #   meta["config"] 블록을 본다. 읽는 쪽과 쓰는 쪽이 한 벌로 맞춰져 있다.
    return RunRow(
        path=str(path),
        name=str(meta["name"]),
        algo=str(meta["algo"]),
        seed=int(meta["seed"]),
        n_episodes=int(meta["n_episodes"]),
        final_ev=float(ev[-1]),
        episodes_to_target=도달,
        wall_sec=float(meta["wall_sec"]),
    )


def group_rows(rows: list[RunRow]) -> list[dict]:
    """같은 (실행 이름, 알고리즘)끼리 묶어 시드 간 통계를 낸다."""
    묶음: dict[tuple[str, str], list[RunRow]] = {}
    for r in rows:
        묶음.setdefault((r.name, r.algo), []).append(r)

    결과: list[dict] = []
    for (name, algo), 값들 in sorted(묶음.items()):
        evs = np.array([v.final_ev for v in 값들], dtype=np.float64)
        도달들 = [v.episodes_to_target for v in 값들 if v.episodes_to_target is not None]
        n_params, q_bytes = params_and_memory(algo)
        결과.append({
            "name": name,
            "algo": algo,
            "n_seeds": len(값들),
            "final_ev_mean": float(evs.mean()),
            # 왜: 시드가 하나뿐일 때 0.0이라고 적으면 '시드 간 분산이 없다'는
            #     거짓말이 된다. 표에는 '-'로 나가도록 None을 남긴다.
            "final_ev_std": float(evs.std(ddof=1)) if len(값들) >= 2 else None,
            "episodes_to_target": int(np.median(도달들)) if 도달들 else None,
            "n_reached": len(도달들),
            "wall_sec_mean": float(np.mean([v.wall_sec for v in 값들])),
            "n_episodes": int(max(v.n_episodes for v in 값들)),
            "n_params": n_params,
            "q_bytes": q_bytes,
        })
    return 결과


def render_table(groups: list[dict], target_ev: float,
                 baselines: list[tuple[str, float]]) -> str:
    """설계서 §8.3이 요구한 열을 가진 마크다운 표 2장을 만든다."""
    줄들 = [
        "## 최종 평가표 (EV는 전부 DP 정확 계산 — 시뮬레이션 0회, 분산 0)",
        "",
        f"목표 EV = {target_ev:+.4f}",
        "",
        "| 알고리즘 | 실행 이름 | 시드 수 | 에피소드 | 최종 EV(DP 정확) | "
        "시드 간 표준편차 | 목표 EV 도달 에피소드 | 파라미터 수 | 메모리 | wall_sec |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for g in groups:
        표준편차 = "-" if g["final_ev_std"] is None else f"{g['final_ev_std']:.6f}"
        도달 = ("미도달" if g["episodes_to_target"] is None
                else f"{g['episodes_to_target']:,}")
        줄들.append(
            f"| {g['algo']} | {g['name']} | {g['n_seeds']} | {g['n_episodes']:,} | "
            f"{g['final_ev_mean']:+.6f} | {표준편차} | {도달} | "
            f"{g['n_params']:,} | {g['q_bytes'] / 1024:.0f} KB | "
            f"{g['wall_sec_mean']:.1f} |")

    줄들 += [
        "",
        "## 기준 정책 (학습 결과와 똑같은 DP 평가기로 계산한 값이다)",
        "",
        "| 정책 | EV |",
        "|---|---|",
    ]
    for 이름, 값 in baselines:
        줄들.append(f"| {이름} | {값:+.6f} |")

    줄들 += [
        "",
        "벽시계 시간은 구현 의존이다(순수 파이썬 vs PyTorch). 주축은 에피소드 수다.",
        "메모리는 학습 중 Q 테이블 크기다(float64). Double Q만 표가 둘이라 두 배다.",
    ]
    return "\n".join(줄들)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="최종 평가표 한 장 생성")
    p.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    p.add_argument("--target-ev", type=float, default=DEFAULT_TARGET_EV)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args(argv)

    경로들 = sorted(Path(args.artifacts).glob("*.npz"))
    if not 경로들:
        print(f"스냅샷이 없다: {args.artifacts}/*.npz")
        return 1

    행들 = [read_run_row(p2, args.target_ev) for p2 in 경로들]
    for 행 in 행들:
        # 왜: EV가 -0.003보다 좋으면 학습을 잘한 게 아니라 버그다. 표를 만들어
        #     남기기 전에 여기서 멈춘다(설계서 §8.2 자동 버그 경보).
        sanity_alarm(행.final_ev)

    묶음 = group_rows(행들)
    기준 = baseline_evs(RULES_V1)
    print(render_table(묶음, args.target_ev, 기준))

    실을것 = {
        "schema_version": 1,
        "target_ev": args.target_ev,
        "dp_optimal_ev": 기준[0][1],
        "rules_fp": RULES_V1.fingerprint(),
        "baselines": [{"name": n, "ev": v} for n, v in 기준],
        "runs": [행._asdict() for 행 in 행들],
        "groups": 묶음,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(실을것, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"\n저장: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
