"""이 파일은 명령줄 플래그 한 줄로 학습을 돌리고 산출물 두 개를 남긴다
입력: --name --algo --rules --seed --episodes --frames --artifacts --live 등 CLI 플래그
출력: artifacts/{stem}.npz 와 artifacts/{stem}.json, 표준출력 요약
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 왜: pyproject의 pythonpath=["src"]는 pytest에만 적용된다. 발표 중
#     `python scripts/train.py`를 맨손으로 쳐도 돌아야 하므로 여기서 직접 넣는다.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blackjack_rl.chartspec import CHART_ROWS, DEALER_COLS  # noqa: E402
from blackjack_rl.dp.exact import solve_optimal  # noqa: E402
from blackjack_rl.eval.agreement import make_agreement_fn  # noqa: E402
from blackjack_rl.train.config import (HEADLINE_EPISODES, ExperimentConfig,  # noqa: E402
                                       build_parser, config_from_args)
from blackjack_rl.train.runner import run  # noqa: E402
from blackjack_rl.train.snapshot import (artifact_path, config_path,  # noqa: E402
                                         save_run)


def render_text_chart(notation: np.ndarray) -> str:
    """36x10 표기 배열을 글자만으로 그린다. 미결정 칸은 점 하나로 둔다."""
    # 왜: 라이브 화면은 matplotlib도 plotly도 쓰지 않는다. 의존성이 하나라도
    #     끼면 발표 당일 터미널에서 안 돌 위험이 생긴다.
    줄들 = ["행     " + " ".join(f"{c:>2}" for c in DEALER_COLS)]
    for i, row in enumerate(CHART_ROWS):
        칸들 = []
        for j in range(len(DEALER_COLS)):
            글자 = str(notation[i, j])
            칸들.append(f"{글자 if 글자 else '.':>2}")
        줄들.append(f"{row.label:<6} " + " ".join(칸들))
    return "\n".join(줄들)


def live_banner(cfg: ExperimentConfig) -> str:
    """라이브 화면 맨 위에 붙는 축소 배율 고백 한 줄."""
    배율 = HEADLINE_EPISODES / max(cfg.n_episodes, 1)
    return (f"[라이브] 이 화면은 최종 헤드라인({HEADLINE_EPISODES:,} 에피소드)의 "
            f"약 1/{배율:.0f} 규모 미니 학습입니다. 로컬 전용입니다.")


def _진행_출력(프레임: int, 전체: int, 에피소드: int) -> None:
    """프레임이 찍힐 때마다 한 줄을 덮어쓴다."""
    # 왜 \r로 덮어쓰는가: 매 프레임 새 줄을 찍으면 200줄이 쏟아진다.
    # 왜 표를 여기서 안 그리는가: 진행 콜백 시그니처는 정수 3개 하나뿐이다.
    #     표를 넘기려고 시그니처를 늘리면 runner와 스크립트가 갈라진다.
    print(f"\r프레임 {프레임}/{전체}  에피소드 {에피소드:,}", end="", flush=True)


def main(argv: list[str] | None = None) -> int:
    인자 = build_parser().parse_args(argv)
    cfg = config_from_args(인자)

    print(f"설정 지문 {cfg.fingerprint()} / 규칙 지문 {cfg.rules.fingerprint()}")
    print(f"알고리즘 {cfg.algo} / 시작분포 {cfg.start_dist} / "
          f"에피소드 {cfg.n_episodes:,} / 프레임 {cfg.n_frames}")
    if 인자.live:
        print(live_banner(cfg))

    훅 = None
    if not 인자.live and 인자.agreement_hands > 0:
        # 왜 여기서 꽂는가: 이 한 줄이 없으면 모든 산출물의 agree_a/agree_b가
        #     영구히 nan이고 설계서 §4.1의 '실시간 일치율 카운터'가 존재하지 않는다.
        #     참조표·사전등록·자연 빈도는 실행당 한 번만 계산된다(20만 핸드 약 8초).
        훅 = make_agreement_fn(cfg.rules, solve_optimal(cfg.rules),
                              n_freq=인자.agreement_hands)

    산출물 = run(cfg, progress=_진행_출력, agreement=훅)
    print()

    m = 산출물.meta
    print(f"에피소드      {m['n_episodes']:,}")
    print(f"학습 시간     {m['train_sec']:.1f} 초 ({m['eps_per_sec']:,.0f} eps/s)")
    print(f"프레임 시간   {m['frame_sec']:.1f} 초 ({m['n_frames']} 프레임)")
    print(f"최종 EV       {산출물.ev_greedy[-1]:+.4f}  "
          f"(DP 최적 {m['dp_ev_initial']:+.6f})")
    print(f"행동정책 EV   {산출물.ev_behavior[-1]:+.4f}")
    print(f"미결정 칸     {int((산출물.chart_visits[-1] == 0).sum())} / 360")
    print()
    print(render_text_chart(산출물.chart_notation[-1]))

    if 인자.live:
        # 왜: 라이브는 '지금 학습되는 중'을 보여주는 용도다. 여기서 만든 작은 표가
        #     artifacts/에 섞이면 어느 npz가 헤드라인인지 흐려진다.
        print("\n[라이브] 스냅샷은 저장하지 않는다.")
        return 0

    npz경로 = artifact_path(cfg, 인자.artifacts)
    json경로 = config_path(cfg, 인자.artifacts)
    해시 = save_run(npz경로, 산출물)
    json경로.parent.mkdir(parents=True, exist_ok=True)
    json경로.write_text(cfg.to_json(), encoding="utf-8")

    print(f"\nq_final sha256 {해시}")
    print(f"저장 {npz경로} ({npz경로.stat().st_size / 1024:.0f} KB)")
    print(f"저장 {json경로}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
