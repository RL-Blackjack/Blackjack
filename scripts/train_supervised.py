"""이 파일은 지도학습 실험 격자를 전부 돌려 모델 비교표를 만든다
입력: --seeds 시드 목록, --out 보고서 경로, --models 모델 저장 폴더
출력: reports/supervised_comparison.json 과 표준출력 마크다운 표
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# 왜: pyproject의 pythonpath=["src"]는 pytest에만 적용된다. 스크립트를 맨손으로
#     실행할 때도 패키지를 찾게 하려면 직접 넣어야 한다.
#     Path를 쓰므로 반드시 from pathlib import Path 아래여야 한다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blackjack_rl.dp.exact import greedy_policy_full, solve_optimal  # noqa: E402
from blackjack_rl.models.compare import ModelRow, make_scorer, rows_to_markdown  # noqa: E402
from blackjack_rl.models.features import build_dataset  # noqa: E402
from blackjack_rl.models.registry import ModelCard, save_policy  # noqa: E402
from blackjack_rl.models.splits import RATIOS, SPLIT_KINDS, make_split  # noqa: E402
from blackjack_rl.models.supervised import MODEL_KINDS, train  # noqa: E402
from blackjack_rl.rules import RULES_V1, RuleSet  # noqa: E402
from blackjack_rl.state import REACHABLE_KEYS  # noqa: E402

SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_REPORT = Path(__file__).resolve().parents[1] / "reports" / "supervised_comparison.json"
DEFAULT_MODELS = Path(__file__).resolve().parents[1] / "models"


def run_grid(rules: RuleSet, dp, seeds: tuple[int, ...] = SEEDS) -> list[ModelRow]:
    """모델 2종 x 비율 3가지 x 분할 2방식 x 시드를 전부 돌린다."""
    X, y, keys = build_dataset(dp)
    점수 = make_scorer(rules, dp)

    행들: list[ModelRow] = []
    for kind in MODEL_KINDS:
        for split_kind in SPLIT_KINDS:
            for ratio in RATIOS:
                for seed in seeds:
                    분할 = make_split(split_kind, keys, ratio, seed)
                    모델 = train(kind, X, y, 분할, seed)
                    이름 = f"{kind}_{split_kind}_{int(ratio * 100)}_seed{seed}"
                    행들.append(점수(
                        이름, "supervised", 모델.policy_full(X),
                        n_train_labels=모델.n_train_labels,
                        test_acc=모델.test_acc,
                        fit_seconds=모델.fit_seconds,
                        n_params=모델.n_params(),
                    ))
    return 행들


def _조건(name: str) -> tuple[str, str, float]:
    """행 이름에서 (모델, 분할, 비율)을 되뽑는다."""
    kind, split_kind, pct, _seed = name.split("_")
    return kind, split_kind, int(pct) / 100.0


def aggregate(rows: list[ModelRow]) -> list[dict]:
    """같은 조건의 시드들을 평균과 표준편차로 묶는다."""
    묶음: dict[tuple[str, str, float], list[ModelRow]] = {}
    for r in rows:
        묶음.setdefault(_조건(r.name), []).append(r)

    결과 = []
    for (kind, split_kind, ratio), 목록 in 묶음.items():
        evs = np.array([r.exact_ev for r in 목록])
        accs = np.array([r.test_acc for r in 목록])
        결과.append({
            "kind": kind, "split_kind": split_kind, "ratio": ratio,
            "n_seeds": len(목록),
            "ev_mean": float(evs.mean()), "ev_std": float(evs.std()),
            "gap_pp_mean": float(np.mean([r.gap_pp for r in 목록])),
            "test_acc_mean": float(accs.mean()), "test_acc_std": float(accs.std()),
            "tier_a_mean": float(np.mean([r.tier_a for r in 목록])),
            "fit_seconds_mean": float(np.mean([r.fit_seconds for r in 목록])),
            "n_train_labels": int(목록[0].n_train_labels),
        })
    결과.sort(key=lambda m: (m["kind"], m["split_kind"], m["ratio"]))
    return 결과


def baseline_rows(rules: RuleSet, dp) -> list[ModelRow]:
    """비교의 양 끝을 잡아 주는 기준선들."""
    점수 = make_scorer(rules, dp)
    행들 = [점수("dp_optimal", "dp", greedy_policy_full(dp)),
           점수("always_stand", "baseline", np.zeros(len(REACHABLE_KEYS), dtype=np.int8))]

    # 강화학습 결과가 있으면 함께 싣는다. 없으면 건너뛴다.
    # 왜 건너뛰는가: 이 스크립트만 따로 돌려 보는 경우에도 동작해야 한다.
    #
    # 왜 '알파벳 마지막'이 아니라 '에피소드가 가장 많은' 것을 고르는가:
    #   artifacts/ 에는 짧은 시험 학습과 본 학습이 섞여 있다. 처음에는
    #   sorted(glob)[-1] 로 골랐는데, 그러면 20만 에피소드짜리 시험 실행이
    #   뽑혀 강화학습의 EV가 -3.29%로 보고됐다(실제 본 학습은 -0.71%).
    #   비교표의 기준선이 4.6배 왜곡되는 사고였다. 의도를 코드로 못박는다.
    후보 = sorted(Path("artifacts").glob("*.npz")) if Path("artifacts").exists() else []
    if 후보:
        from blackjack_rl.train.snapshot import load_run
        실행들 = [(p, load_run(p)) for p in 후보]
        경로, art = max(실행들, key=lambda pair: int(pair[1].episodes[-1]))
        행들.append(점수(f"rl_{경로.stem.split('__')[0]}", "rl",
                       art.policy_full[-1].astype(np.int8)))
    return 행들


def build_report(rules: RuleSet, dp, seeds: tuple[int, ...] = SEEDS) -> dict:
    """격자 결과와 기준선을 한 덩어리 보고서로 묶는다."""
    행들 = run_grid(rules, dp, seeds)
    기준선 = baseline_rows(rules, dp)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rules_fp": rules.fingerprint(),
        "dp_ev": float(dp.ev_initial),
        "seeds": list(seeds),
        "rows": [r.__dict__ for r in 행들],
        "aggregates": aggregate(행들),
        "baselines": [r.__dict__ for r in 기준선],
    }


def render_markdown(payload: dict) -> str:
    """보고서를 사람이 읽을 표로 만든다."""
    줄 = [f"규칙 지문 {payload['rules_fp']} / DP 최적 EV {payload['dp_ev'] * 100:+.4f}%", ""]

    줄 += ["## 기준선", "",
          "| 모델 | 계열 | 정확 EV | DP와의 격차 |", "|---|---|---|---|"]
    for b in payload["baselines"]:
        이름 = "DP 최적" if b["name"] == "dp_optimal" else b["name"]
        줄.append(f"| {이름} | {b['family']} | {b['exact_ev'] * 100:+.4f}% | {b['gap_pp']:.4f}%p |")

    줄 += ["", f"## 지도학습 격자 (시드 {len(payload['seeds'])}개 평균)", "",
          "| 모델 | 분할 | 정답 비율 | 훈련 레이블 | 정확 EV | 시드 편차 | 시험 정확도 | 학습 시간 |",
          "|---|---|---|---|---|---|---|---|"]
    for m in payload["aggregates"]:
        분할 = "무작위" if m["split_kind"] == "random" else "구조적"
        줄.append(
            f"| {m['kind']} | {분할} | {m['ratio']:.0%} | {m['n_train_labels']} | "
            f"{m['ev_mean'] * 100:+.4f}% | {m['ev_std'] * 100:.4f} | "
            f"{m['test_acc_mean']:.3f} | {m['fit_seconds_mean']:.2f}초 |")
    return "\n".join(줄)


def save_representative_models(rules: RuleSet, dp, out_dir: Path) -> list[Path]:
    """조건마다 시드 0 모델 하나씩을 게임 서버가 쓸 수 있게 저장한다."""
    X, y, keys = build_dataset(dp)
    점수 = make_scorer(rules, dp)
    저장됨 = []
    for kind in MODEL_KINDS:
        for split_kind in SPLIT_KINDS:
            for ratio in RATIOS:
                분할 = make_split(split_kind, keys, ratio, 0)
                모델 = train(kind, X, y, 분할, 0)
                정책 = 모델.policy_full(X)
                이름 = f"{kind}_{split_kind}_{int(ratio * 100)}"
                행 = 점수(이름, "supervised", 정책)
                카드 = ModelCard(
                    name=이름, family="supervised", rules_fp=rules.fingerprint(),
                    exact_ev=행.exact_ev, policy_sha256="",
                    n_train_labels=모델.n_train_labels, test_acc=모델.test_acc,
                    fit_seconds=모델.fit_seconds, n_params=모델.n_params(),
                    created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                )
                저장됨.append(save_policy(out_dir / f"{이름}.npz", 정책, 카드))
    return 저장됨


def main() -> None:
    parser = argparse.ArgumentParser(description="지도학습 실험 격자 실행")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--out", type=str, default=str(DEFAULT_REPORT))
    parser.add_argument("--models", type=str, default=str(DEFAULT_MODELS))
    args = parser.parse_args()

    dp = solve_optimal(RULES_V1)
    payload = build_report(RULES_V1, dp, tuple(args.seeds))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    저장됨 = save_representative_models(RULES_V1, dp, Path(args.models))

    print(render_markdown(payload))
    print(f"\n보고서 저장: {out}")
    print(f"대표 모델 {len(저장됨)}개 저장: {Path(args.models)}")


if __name__ == "__main__":
    main()
