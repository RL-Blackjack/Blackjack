"""이 파일은 공유 학습 루프를 호출하고 로그 간격마다 스냅샷 프레임을 찍는다.
입력: ExperimentConfig 하나와 선택적 진행 콜백·일치율 훅·시작 샘플러.
출력: RunArtifact — 프레임 배열 13개와 재현 정보가 담긴 meta."""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable

import numpy as np

from blackjack_rl.agents.loop import make_env, run_episodes
from blackjack_rl.agents.tabular import TabularAgent
from blackjack_rl.chartspec import project
from blackjack_rl.dp.exact import evaluate_policy, evaluate_stochastic, solve_optimal
from blackjack_rl.eval.agreement import AgreementFn, chart_margin, chart_visits
from blackjack_rl.eval.bias import maxq_minus_vstar
from blackjack_rl.rng import make_streams
from blackjack_rl.starts import ExploringStarts, NaturalDeal, StartSampler
from blackjack_rl.state import REACHABLE_KEYS
from blackjack_rl.train.config import ExperimentConfig
from blackjack_rl.train.snapshot import (RunArtifact, git_commit_sha, log_schedule,
                                         seed_record)

# 왜 이 시그니처 하나인가: (프레임번호, 전체프레임, 에피소드) 세 정수다. 표(ChartTable)를
#   콜백으로 넘기지 않는다 — 라이브 화면이 표를 그리려고 콜백 모양을 바꾸면 scripts와
#   runner가 서로 다른 시그니처를 믿게 되어 --live가 TypeError로 즉사한다.
ProgressFn = Callable[[int, int, int], None]


def _make_sampler(cfg: ExperimentConfig) -> StartSampler:
    """cfg.start_dist 문자열을 실제 시작 상태 샘플러로 바꾼다."""
    if cfg.start_dist == "natural":
        return NaturalDeal()
    # 왜 cfg.rules를 넘기는가: ExploringStarts의 기본값은 RULES_V1이다. 인자 없이
    #   만들면 --rules h17 실행이 규칙이 어긋난 시작 쌍으로 학습하게 되고,
    #   can_double/can_split 복원이 틀리면 forced_first_action이 불법이 되어
    #   env가 ValueError를 던진다.
    return ExploringStarts(cfg.rules)


def run(
    cfg: ExperimentConfig,
    progress: ProgressFn | None = None,
    sampler: StartSampler | None = None,
    agreement: AgreementFn | None = None,
) -> RunArtifact:
    """설정 하나를 받아 끝까지 학습하고 산출물을 돌려준다.

    학습 루프는 여기 없다 — agents.loop.run_episodes 하나뿐이고 이 함수는 그것을
    호출한다. 프레임은 snapshot_at/on_snapshot으로 찍는다. 설계서 §2.12의
    "학습 루프가 다섯 알고리즘에 완전히 공유된다"가 이 구조에서 나온다.
    """
    streams = make_streams(cfg.seed)
    agent = TabularAgent(
        cfg.rules, streams,
        algo=cfg.algo, step=cfg.step, alpha=cfg.alpha,
        eps0=cfg.eps0, eps_final=cfg.eps_final,
        eps_decay_at=cfg.eps_decay_at, encoder=cfg.encoder,
    )
    env = make_env(cfg.rules, streams)
    시작샘플러 = sampler if sampler is not None else _make_sampler(cfg)
    dp = solve_optimal(cfg.rules)

    F = cfg.n_frames
    일정 = log_schedule(cfg.n_episodes, F)

    buf: dict[str, np.ndarray] = {
        "episodes": np.zeros(F, dtype=np.int64),
        "policy_full": np.zeros((F, len(REACHABLE_KEYS)), dtype=np.int8),
        "chart_action": np.zeros((F, 36, 10), dtype=np.int8),
        "chart_notation": np.full((F, 36, 10), "", dtype="<U2"),
        "chart_margin": np.zeros((F, 36, 10), dtype=np.float32),
        "chart_visits": np.zeros((F, 36, 10), dtype=np.int32),
        "ev_greedy": np.zeros(F, dtype=np.float32),
        "ev_behavior": np.zeros(F, dtype=np.float32),
        "agree_a": np.zeros(F, dtype=np.float32),
        "agree_b": np.zeros(F, dtype=np.float32),
        "maxq_minus_vstar": np.zeros(F, dtype=np.float32),
    }

    프레임 = 0
    프레임초 = 0.0

    def 스냅샷(ep: int) -> None:
        nonlocal 프레임, 프레임초
        프레임시작 = time.perf_counter()
        f = 프레임
        q = agent.q_eval_table()
        방문 = chart_visits(agent.visit_table())
        표 = project(q, cfg.rules, visits=방문)
        정책 = agent.greedy_full()

        buf["episodes"][f] = ep
        buf["policy_full"][f] = 정책
        buf["chart_action"][f] = 표.action
        buf["chart_notation"][f] = 표.notation
        buf["chart_margin"][f] = chart_margin(q)
        buf["chart_visits"][f] = 방문
        # 왜 시뮬레이션을 부르지 않는가: 설계서 §4.2 심사 지적 해소 ⑧ — 학습곡선을
        #   시뮬로 그리면 200프레임 x 5알고리즘에 수십억 핸드가 든다. DP 정확
        #   정책평가는 13.5ms고 분산이 0이라 신뢰구간조차 필요 없다.
        buf["ev_greedy"][f] = evaluate_policy(정책, cfg.rules)
        buf["ev_behavior"][f] = evaluate_stochastic(
            agent.eps_greedy_dist(cfg.eps_at(ep)), cfg.rules)
        # 왜 import해서 쓰는가: 같은 이름을 여기서 다시 계산하면 프로브 50칸이 아니라
        #   610칸 평균이 되어(V* 평균 +0.0906) 발표의 '편향 곡선'과 Task 4의 지표가
        #   서로 다른 것을 재게 된다.
        buf["maxq_minus_vstar"][f] = maxq_minus_vstar(q, dp.V)

        if agreement is None:
            # 왜: 0.0으로 채우면 '일치율 0%'로 읽힌다. 아직 안 잰 것은 nan이어야 한다.
            buf["agree_a"][f] = np.nan
            buf["agree_b"][f] = np.nan
        else:
            buf["agree_a"][f], buf["agree_b"][f] = agreement(표, 방문)

        프레임 += 1
        프레임초 += time.perf_counter() - 프레임시작
        if progress is not None:
            progress(프레임, F, ep)

    시작시각 = time.perf_counter()
    # 왜 explore 스트림을 넘기는가: ExploringStarts.sample은 rng.integers를 소비한다.
    #   deal을 쓰면 알고리즘을 바꿀 때 카드 순서가 달라져 설계서 §8.1-1의 공정 비교가
    #   깨진다. NaturalDeal은 rng를 쓰지 않으므로 어느 쪽이든 결과가 같다.
    run_episodes(env, agent, 시작샘플러, streams.explore, cfg.n_episodes,
                 snapshot_at={int(x) for x in 일정}, on_snapshot=스냅샷)
    전체초 = time.perf_counter() - 시작시각
    학습초 = max(전체초 - 프레임초, 1e-9)

    meta = {
        "schema_version": 2,
        # 왜 두 벌인가: top-level은 scripts/evaluate.py가 표를 만들 때 읽고,
        #   config 블록은 scripts/verify_release.py가 설정을 통째로 되살릴 때 읽는다.
        #   읽는 쪽과 쓰는 쪽을 한 곳에서 맞춰 둬야 실제 산출물에서 KeyError가 안 난다.
        "name": cfg.name,
        "algo": cfg.algo,
        "seed": int(cfg.seed),
        "n_episodes": int(cfg.n_episodes),
        "n_frames": int(F),
        "rules_fp": cfg.rules.fingerprint(),
        "cfg_fp": cfg.fingerprint(),
        "config": json.loads(cfg.to_json()),
        "seeds": seed_record(streams, cfg.seed),
        "git_commit": git_commit_sha(),
        # 왜: 이 칸은 save_run이 실제로 저장되는 float32 배열을 해시해 채운다.
        "q_sha256": "",
        "wall_sec": float(전체초),
        "train_sec": float(학습초),
        "frame_sec": float(프레임초),
        "eps_per_sec": float(cfg.n_episodes / 학습초),
        "dp_ev_initial": float(dp.ev_initial),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }

    # 왜 q_eval_table / visit_table인가: doubleq는 표가 둘이다. agent.Q와 agent.N을
    #   그대로 저장하면 절반만 남아 재현도 설명도 불완전해진다.
    return RunArtifact(
        **buf,
        q_final=agent.q_eval_table().astype(np.float32),
        n_final=agent.visit_table().astype(np.int32),
        meta=meta,
    )
