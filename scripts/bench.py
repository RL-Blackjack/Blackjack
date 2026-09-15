"""이 파일은 학습 전체의 계산 예산을 실측한다
입력: --rounds 측정 라운드 수, --calls 마스크 호출 횟수, --seed 시드, --out JSON 경로
출력: reports/bench.json 파일 1개와 표준출력 마크다운 표 3개
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

# 왜: 이 스크립트는 W1의 첫 산출물이라 blackjack_rl 패키지보다 먼저 존재한다.
#     그래서 패키지를 import하지 않고 최소 커널을 이 파일 안에 직접 갖는다.

CARD_RANKS = np.arange(1, 11, dtype=np.int64)
CARD_PROBS = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 4], dtype=np.float64) / 13.0

PLANNED_EPS_PER_SEC = 20_000.0

# 왜: 커널은 스플릿·불변식 검사·통계 누적을 생략했으므로 실제 env보다 빠르다.
#     처음에는 2.0으로 추정했으나, W2가 끝난 뒤 실제 BlackjackEnv.play_round를
#     측정하니 79,145 라운드/초였다. 같은 PC에서 이 커널은 250,507 eps/s이므로
#     실제 배율은 3.17이다. 추정 대신 실측값을 쓴다(2026-09-16 측정).
KERNEL_SAFETY_FACTOR = 3.2

KERNEL_SCOPE_NOTE = ("포함: 카드 뽑기, 소프트 에이스, 마스크, Q 조회, argmax, 궤적 기록, "
                     "딜러 플레이, 배당. 제외: 스플릿, 불변식 assert, 통계 누적.")

BUDGET_ROWS: tuple[tuple[str, float, float], ...] = (
    ("MC / Q / DoubleQ (자연딜)", 3e7, 1.00),
    ("2x2 스텝사이즈 추가분", 2e7, 1.00),
    ("MC / Q (exploring, 헤드라인)", 6e7, 1.00),
    ("S2 카운트 확장 (슈 모드)", 5e7, 0.60),
    ("S2 플라시보 대조", 5e7, 0.60),
    ("베팅 밴딧", 5e6, 0.85),
    ("DQN (S1 스윕 3 + S1/S2 x 3시드)", 1.8e7, 0.08),
    ("평가 시뮬 (CRN, 근사갭)", 2e7, 0.28),
)

Q_SHAPE = (22, 2, 12, 2, 2, 4, 4)
STAND, HIT, DOUBLE = 0, 1, 2


def make_card_pool(n_cards: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(CARD_RANKS, size=n_cards, p=CARD_PROBS)


def add_card(total: int, soft: bool, card: int) -> tuple[int, bool]:
    base = total - 10 if soft else total
    new_base = base + card
    has_ace = soft or card == 1
    if has_ace and new_base + 10 <= 21:
        return new_base + 10, True
    return new_base, False


def build_legal_table() -> np.ndarray:
    table = np.zeros(Q_SHAPE, dtype=bool)
    table[..., 0] = True
    table[..., 1] = True
    table[:, :, :, 1, :, :, 2] = True
    table[:, :, :, :, 1, :, 3] = True
    return table


def legal_by_table(table: np.ndarray, key: tuple[int, ...]) -> np.ndarray:
    return table[key]


def legal_by_build(key: tuple[int, ...]) -> np.ndarray:
    _total, _is_soft, _up, can_double, can_split, _depth = key
    out = np.zeros(4, dtype=bool)
    out[0] = True
    out[1] = True
    if can_double:
        out[2] = True
    if can_split:
        out[3] = True
    return out


def play_dealer(pool: np.ndarray, pos: int, up: int, hole: int) -> tuple[int, int]:
    total, soft = add_card(*add_card(0, False, up), hole)
    while total < 17:
        total, soft = add_card(total, soft, int(pool[pos]))
        pos += 1
    return total, pos


def play_one_round(pool: np.ndarray, pos: int, q_table: np.ndarray,
                   legal_table: np.ndarray, use_table: bool) -> tuple[float, int, int]:
    p1 = int(pool[pos]); p2 = int(pool[pos + 1])
    up = int(pool[pos + 2]); hole = int(pool[pos + 3])
    pos += 4
    up_idx = 11 if up == 1 else up

    total, soft = add_card(*add_card(0, False, p1), p2)
    can_double = 1
    bet = 1.0
    trajectory: list[tuple[tuple[int, ...], int]] = []

    while True:
        key = (total, int(soft), up_idx, can_double, 0, 0)
        mask = legal_table[key] if use_table else legal_by_build(key)
        values = np.where(mask, q_table[key], -np.inf)
        action = int(np.argmax(values))
        trajectory.append((key, action))
        if action == STAND:
            break
        total, soft = add_card(total, soft, int(pool[pos]))
        pos += 1
        if action == DOUBLE:
            bet = 2.0
            break
        can_double = 0
        if total > 21:
            break

    if total > 21:
        reward = -bet
    else:
        dealer_total, pos = play_dealer(pool, pos, up, hole)
        if dealer_total > 21 or total > dealer_total:
            reward = bet
        elif total < dealer_total:
            reward = -bet
        else:
            reward = 0.0

    for key, action in trajectory:
        q_table[key][action] += 0.0  # 왜: 학습 업데이트의 쓰기 비용도 예산에 포함한다.
    return reward, len(trajectory), pos


def bench_rounds(n_rounds: int, seed: int, use_table: bool) -> dict:
    pool = make_card_pool(n_rounds * 12 + 64, seed)
    rng = np.random.default_rng(seed + 1)
    q_table = rng.uniform(-1e-6, 1e-6, size=Q_SHAPE)
    legal_table = build_legal_table()
    pos = 0
    net = 0.0
    decisions = 0
    t0 = time.perf_counter()
    for _ in range(n_rounds):
        if pos + 24 >= pool.size:
            pos = 0
        reward, n_dec, pos = play_one_round(pool, pos, q_table, legal_table, use_table)
        net += reward
        decisions += n_dec
    wall = time.perf_counter() - t0
    return {
        "n_rounds": n_rounds,
        "mask_mode": "table_view" if use_table else "build_array",
        "wall_sec": wall,
        "eps_per_sec": n_rounds / wall,
        "n_decisions": decisions,
        "decisions_per_round": decisions / n_rounds,
        "us_per_decision": wall / decisions * 1e6,
        "ev_per_round": net / n_rounds,
    }


def sample_keys() -> list[tuple[int, ...]]:
    return [(total, 0, up, cd, cs, 0) for total in range(4, 22)
            for up in range(2, 12) for cd in (0, 1) for cs in (0, 1)]


def masks_are_equal() -> bool:
    table = build_legal_table()
    return all(np.array_equal(legal_by_table(table, key), legal_by_build(key))
               for key in sample_keys())


def bench_masks(n_calls: int) -> dict:
    table = build_legal_table()
    keys = sample_keys()
    n_keys = len(keys)

    t0 = time.perf_counter()
    for i in range(n_calls):
        legal_by_table(table, keys[i % n_keys])
    t_table = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(n_calls):
        legal_by_build(keys[i % n_keys])
    t_build = time.perf_counter() - t0

    return {"n_calls": n_calls, "table_view_ns": t_table / n_calls * 1e9,
            "build_array_ns": t_build / n_calls * 1e9,
            "call_speedup": t_build / t_table, "values_equal": masks_are_equal()}


def budget_rows(eps_per_sec: float) -> list[dict]:
    return [{"name": name, "episodes": episodes, "rel_speed": rel,
             "hours": episodes / (eps_per_sec * rel) / 3600.0}
            for name, episodes, rel in BUDGET_ROWS]


def total_hours(rows: list[dict]) -> float:
    return sum(row["hours"] for row in rows)


def run_bench(n_rounds: int, n_calls: int, seed: int) -> dict:
    fast = bench_rounds(n_rounds, seed, use_table=True)
    slow = bench_rounds(n_rounds, seed, use_table=False)
    masks = bench_masks(n_calls)
    masks["round_speedup"] = fast["eps_per_sec"] / slow["eps_per_sec"]

    eps = fast["eps_per_sec"]
    safe_eps = eps / KERNEL_SAFETY_FACTOR
    planned = budget_rows(PLANNED_EPS_PER_SEC)
    measured = budget_rows(eps)
    safe = budget_rows(safe_eps)
    return {
        "schema_version": 1,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "seed": seed,
        "kernel_scope": KERNEL_SCOPE_NOTE,
        "measured": fast,
        "measured_build_array": slow,
        "masks": masks,
        "budget": {
            "planned_eps_per_sec": PLANNED_EPS_PER_SEC,
            "measured_eps_per_sec": eps,
            "safety_factor": KERNEL_SAFETY_FACTOR,
            "safe_eps_per_sec": safe_eps,
            "planned_rows": planned,
            "measured_rows": measured,
            "safe_rows": safe,
            "planned_total_hours": total_hours(planned),
            "measured_total_hours": total_hours(measured),
            "safe_total_hours": total_hours(safe),
        },
    }


def render_markdown(result: dict) -> str:
    fast = result["measured"]
    slow = result["measured_build_array"]
    masks = result["masks"]
    budget = result["budget"]
    lines = [
        "## 1. 라운드 시뮬레이션 실측", "",
        "| 항목 | 테이블 뷰 | 매번 배열 생성 |", "|---|---|---|",
        f"| eps/s | {fast['eps_per_sec']:,.0f} | {slow['eps_per_sec']:,.0f} |",
        f"| 결정당 us | {fast['us_per_decision']:.2f} | {slow['us_per_decision']:.2f} |",
        f"| 라운드당 결정 수 | {fast['decisions_per_round']:.2f} | {slow['decisions_per_round']:.2f} |",
        f"| 라운드당 EV | {fast['ev_per_round']:+.4f} | {slow['ev_per_round']:+.4f} |", "",
        "## 2. legal_actions 두 방식", "",
        "| 항목 | 값 |", "|---|---|",
        f"| 테이블 뷰 호출당 | {masks['table_view_ns']:.1f} ns |",
        f"| 배열 생성 호출당 | {masks['build_array_ns']:.1f} ns |",
        f"| 호출 단위 배율 | {masks['call_speedup']:.2f} 배 |",
        f"| 라운드 전체 배율 | {masks['round_speedup']:.2f} 배 |",
        f"| 두 방식 반환값 동일 | {masks['values_equal']} |", "",
        "## 3. 계산 예산표", "",
        "| 실행 | 에피소드 | 계획(20k eps/s) | 실측 | 안전계수 적용 |",
        "|---|---|---|---|---|",
    ]
    triples = zip(budget["planned_rows"], budget["measured_rows"], budget["safe_rows"])
    for planned, measured, safe in triples:
        lines.append(
            f"| {planned['name']} | {planned['episodes']:,.0f} | "
            f"{planned['hours'] * 60:.0f} 분 | {measured['hours'] * 60:.0f} 분 | "
            f"{safe['hours'] * 60:.0f} 분 |"
        )
    lines.append(
        f"| 합계 | | {budget['planned_total_hours']:.1f} 시간 | "
        f"{budget['measured_total_hours']:.1f} 시간 | "
        f"{budget['safe_total_hours']:.1f} 시간 |"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="블랙잭 RL 계산 예산 실측")
    parser.add_argument("--rounds", type=int, default=200_000)
    parser.add_argument("--calls", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--out", type=str, default="reports/bench.json")
    args = parser.parse_args()

    result = run_bench(args.rounds, args.calls, args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(render_markdown(result))
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
