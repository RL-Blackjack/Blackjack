"""이 파일은 저장된 학습 스냅샷에서 GIF·학습곡선·헤드라인 네 숫자를 뽑는다
입력: --artifacts 스냅샷 폴더, --media 그림 폴더, --reports 리포트 폴더
출력: media/*.gif, media/learning_curves.png, media/bias_curves.png,
      reports/{cfg_fp}.json, 표준출력 실행별 헤드라인 네 숫자
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 왜: 맨손 실행에서도 src/를 찾게 한다(train.py와 같은 이유).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from blackjack_rl.chartspec import ChartTable  # noqa: E402
from blackjack_rl.dp.exact import greedy_policy_full, solve_optimal  # noqa: E402
from blackjack_rl.eval.agreement import (AgreementReport, compare,  # noqa: E402
                                         load_pre_registered)
from blackjack_rl.eval.simulate import (EVAL_SEED, make_card_stream,  # noqa: E402
                                        natural_cell_freq)
from blackjack_rl.reference import load_reference_chart  # noqa: E402
from blackjack_rl.rules import RULES_V1  # noqa: E402
from blackjack_rl.train.snapshot import RunArtifact, load_run  # noqa: E402
from blackjack_rl.viz.celldata import undecided_grid  # noqa: E402
from blackjack_rl.viz.curves import DP_OPTIMAL_EV, bias_curves, learning_curves  # noqa: E402
from blackjack_rl.viz.export import export_gif, save_png  # noqa: E402
from blackjack_rl.viz.heatmap import policy_heatmap  # noqa: E402

DEFAULT_ARTIFACTS = ROOT / "artifacts"
DEFAULT_MEDIA = ROOT / "media"
DEFAULT_REPORTS = ROOT / "reports"


def chart_of(art: RunArtifact, f: int) -> ChartTable:
    """프레임 하나를 ChartTable로 되살린다."""
    # 왜 chart_notation이 필요한가: chart_action만으로는 'D'와 'Ds'가 둘 다 2라
    #   구분되지 않는다. 표기 배열을 npz에 넣어 둔 이유가 정확히 이 한 줄이다.
    return ChartTable(action=np.asarray(art.chart_action[f]),
                      notation=np.asarray(art.chart_notation[f]),
                      rules_fp=str(art.meta["rules_fp"]))


def overlay_chart() -> ChartTable:
    """출판 참조표를 히트맵 오버레이 모양으로 감싼다."""
    ref = load_reference_chart()
    # 왜 action이 -1인가: 참조표 CSV에는 행동 번호가 없다. policy_heatmap은
    #   overlay.notation만 읽으므로 모양만 맞추면 된다.
    return ChartTable(action=np.full((36, 10), -1, dtype=np.int8),
                      notation=ref.notation, rules_fp=ref.rules_fp)


def pick_frames(n_total: int, k: int) -> list[int]:
    """프레임 n_total장에서 k장을 고르게 솎되 처음과 끝은 반드시 넣는다."""
    # 왜 솎는가: 200프레임을 그대로 GIF로 만들면 약 9 MB다(프레임당 46 KB 실측).
    #   40장이면 1.8 MB라 Git LFS 없이 커밋된다.
    if k >= n_total:
        return list(range(n_total))
    자리 = np.linspace(0, n_total - 1, k)
    return sorted({int(round(x)) for x in 자리})


def make_gif(art: RunArtifact, path: Path | str, *, overlay: ChartTable,
             k: int = 40) -> str:
    """표가 채워지는 과정을 GIF 한 장으로 굽는다."""
    프레임들 = []
    for f in pick_frames(len(art.episodes), k):
        프레임들.append(policy_heatmap(
            chart_of(art, f),
            np.asarray(art.chart_margin[f], dtype=np.float64),
            overlay=overlay,
            undecided_mask=undecided_grid(np.asarray(art.chart_visits[f])),
            title=f"{art.meta['algo']}  에피소드 {int(art.episodes[f]):,}"))
    return export_gif(프레임들, path, fps=6)


def shared_inputs(n_freq: int = 200_000):
    """헤드라인 계산에 필요한 DP·참조표·사전등록·자연 빈도를 한 번만 만든다."""
    dp = solve_optimal(RULES_V1)
    ref = load_reference_chart()
    pre = load_pre_registered()
    freq = natural_cell_freq(greedy_policy_full(dp), RULES_V1,
                             make_card_stream(EVAL_SEED, n_freq, RULES_V1))
    return dp, ref, pre, freq


def headline(art: RunArtifact, dp, ref, pre, freq) -> AgreementReport:
    """마지막 프레임의 표로 발표 헤드라인 네 숫자를 뽑는다."""
    if str(art.meta["rules_fp"]) != RULES_V1.fingerprint():
        raise ValueError(
            f"규칙 지문이 RULES_V1과 다른 산출물이다: {art.meta['rules_fp']}")
    return compare(chart_of(art, -1), ref, dp,
                   np.asarray(art.chart_visits[-1]), freq, pre)


def write_report(art: RunArtifact, report: AgreementReport,
                 reports_dir: Path | str) -> Path:
    """설계서 §4.1이 요구한 reports/{cfg_fp}.json 을 남긴다."""
    경로 = Path(reports_dir) / f"{art.meta['cfg_fp']}.json"
    경로.parent.mkdir(parents=True, exist_ok=True)
    본문 = {
        "schema_version": 1,
        "name": art.meta["name"],
        "algo": art.meta["algo"],
        "seed": art.meta["seed"],
        "n_episodes": art.meta["n_episodes"],
        "cfg_fp": art.meta["cfg_fp"],
        "rules_fp": art.meta["rules_fp"],
        "q_sha256": art.meta.get("q_sha256", ""),
        "final_ev_greedy": float(art.ev_greedy[-1]),
        "headline": {
            "tier_a": float(report.tier_a),
            "tier_b": float(report.tier_b),
            "ev_loss_pp": float(report.ev_loss_pp),
            "n_undecided": int(report.n_undecided),
        },
        "mismatches": [
            {"cell": m.cell, "dealer": m.dealer, "ai": m.ai, "ref": m.ref,
             "dp_delta": m.dp_delta, "visits": m.visits,
             "natural_freq": m.natural_freq, "ev_loss": m.ev_loss,
             "cause": m.cause}
            for m in report.mismatches
        ],
    }
    경로.write_text(json.dumps(본문, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    return 경로


def make_curves(arts: dict[str, RunArtifact], media_dir: Path | str) -> list[str]:
    """여러 실행을 한 그림에 겹쳐 V2 학습곡선과 편향 곡선을 만든다."""
    이름들 = list(arts)
    기준 = np.asarray(arts[이름들[0]].episodes)
    for 이름 in 이름들[1:]:
        if not np.array_equal(np.asarray(arts[이름].episodes), 기준):
            # 왜 막는가: x축이 다른 곡선을 한 그림에 겹치면 '언제 배웠는가'가 거짓이 된다.
            raise ValueError(
                f"'{이름}'의 에피소드 축이 '{이름들[0]}'과 다르다. "
                "같은 --episodes/--frames로 돌린 실행만 겹칠 수 있다.")

    그리디 = {이름: np.asarray(a.ev_greedy, dtype=np.float64) for 이름, a in arts.items()}
    행동 = {이름: np.asarray(a.ev_behavior, dtype=np.float64) for 이름, a in arts.items()}
    편향 = {이름: np.asarray(a.maxq_minus_vstar, dtype=np.float64)
          for 이름, a in arts.items()}

    곡선 = learning_curves(기준, 그리디, behavior=행동, dp_ev=DP_OPTIMAL_EV,
                         title="V2 학습곡선 — DP 정확 정책평가")
    편향그림 = bias_curves(기준, 편향, title="최대화 편향 max Q - V* (프로브 50칸)")
    return [save_png(곡선, Path(media_dir) / "learning_curves.png"),
            save_png(편향그림, Path(media_dir) / "bias_curves.png")]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="스냅샷에서 GIF·학습곡선·헤드라인 생성")
    p.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    p.add_argument("--media", default=str(DEFAULT_MEDIA))
    p.add_argument("--reports", default=str(DEFAULT_REPORTS))
    p.add_argument("--gif-frames", type=int, default=40)
    p.add_argument("--freq-hands", type=int, default=200_000)
    args = p.parse_args(argv)

    경로들 = sorted(Path(args.artifacts).glob("*.npz"))
    if not 경로들:
        print(f"스냅샷이 없다: {args.artifacts}/*.npz")
        return 1

    dp, ref, pre, freq = shared_inputs(args.freq_hands)
    오버레이 = overlay_chart()

    산출물들: dict[str, RunArtifact] = {}
    for 경로 in 경로들:
        art = load_run(경로)
        산출물들[str(art.meta["name"])] = art

        gif = make_gif(art, Path(args.media) / f"{경로.stem}.gif",
                       overlay=오버레이, k=args.gif_frames)
        보고 = headline(art, dp, ref, pre, freq)
        리포트 = write_report(art, 보고, args.reports)

        print(f"[{art.meta['name']}] 360칸 중 {보고.tier_a * 100:.1f}% 일치 / "
              f"비자명 칸 중 {보고.tier_b * 100:.1f}% / "
              f"EV 손실 {보고.ev_loss_pp:.2f}%p / 미결정 {보고.n_undecided}칸")
        print(f"    {gif}")
        print(f"    {리포트}")

    for 그림 in make_curves(산출물들, args.media):
        print(f"곡선 {그림}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
