"""이 파일은 실험 격자 실행기가 올바른 개수와 스키마를 내는지 확인한다.
입력: 없음(train_supervised 모듈을 직접 import해 작게 실행한다).
출력: pytest 통과/실패.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_supervised  # noqa: E402

from blackjack_rl.rules import RULES_V1  # noqa: E402


def test_시드가_다섯_개다():
    assert train_supervised.SEEDS == (0, 1, 2, 3, 4)


def test_격자가_정확히_60회다(dp):
    # 왜 60인가: 모델 2종 x 비율 3가지 x 분할 2방식 x 시드 5개.
    #     하나라도 빠지면 비교표에 구멍이 생긴다.
    행들 = train_supervised.run_grid(RULES_V1, dp, seeds=(0,))
    assert len(행들) == 2 * 3 * 2 * 1
    전체 = 2 * 3 * 2 * len(train_supervised.SEEDS)
    assert 전체 == 60


def test_행_이름이_서로_다르다(dp):
    행들 = train_supervised.run_grid(RULES_V1, dp, seeds=(0,))
    assert len({r.name for r in 행들}) == len(행들)


def test_비율_1_0_RF는_DP와_같은_EV를_낸다(dp):
    # 왜: 610칸 정답을 다 주면 랜덤포레스트는 표를 그대로 외운다. 실측에서
    #     EV -0.5108%, 시드 편차 0.000이 나왔다. 이 값이 흔들리면 인코딩이나
    #     레이블이 깨진 것이다.
    행들 = train_supervised.run_grid(RULES_V1, dp, seeds=(0,))
    대상 = [r for r in 행들 if r.name.startswith("rf_random_100")]
    assert len(대상) == 1
    assert 대상[0].exact_ev == pytest.approx(dp.ev_initial, abs=1e-6)


def test_집계가_평균과_표준편차를_낸다(dp):
    행들 = train_supervised.run_grid(RULES_V1, dp, seeds=(0, 1))
    묶음 = train_supervised.aggregate(행들)
    assert len(묶음) == 2 * 3 * 2
    for m in 묶음:
        for 키 in ("kind", "split_kind", "ratio", "ev_mean", "ev_std",
                   "test_acc_mean", "gap_pp_mean", "n_seeds"):
            assert 키 in m, f"{키}가 집계에 없다"
        assert m["n_seeds"] == 2


def test_구조적_분할이_무작위보다_EV가_나쁘다(dp):
    # 왜: 실측(5시드)에서 rf 0.6은 무작위 -1.617% vs 구조적 -2.231%였다.
    #     이 부등호가 뒤집히면 구조적 분할이 제 역할을 못 하는 것이다.
    묶음 = train_supervised.aggregate(
        train_supervised.run_grid(RULES_V1, dp, seeds=(0, 1, 2)))
    def 찾기(kind, sk, ratio):
        return next(m for m in 묶음
                    if m["kind"] == kind and m["split_kind"] == sk and m["ratio"] == ratio)
    assert 찾기("rf", "structural", 0.6)["ev_mean"] < 찾기("rf", "random", 0.6)["ev_mean"]


def test_보고서_스키마가_전부_채워진다(tmp_path, dp):
    payload = train_supervised.build_report(RULES_V1, dp, seeds=(0,))
    for 키 in ("schema_version", "generated_at", "rules_fp", "dp_ev",
               "seeds", "rows", "aggregates", "baselines"):
        assert 키 in payload
    assert payload["schema_version"] == 1
    assert payload["dp_ev"] == pytest.approx(dp.ev_initial, abs=1e-9)
    # 기준선에 강화학습과 DP 최적이 들어 있어야 비교가 성립한다.
    이름들 = {b["name"] for b in payload["baselines"]}
    assert "dp_optimal" in 이름들


def test_마크다운에_표가_들어있다(dp):
    payload = train_supervised.build_report(RULES_V1, dp, seeds=(0,))
    md = train_supervised.render_markdown(payload)
    assert "| 모델 |" in md
    assert "DP 최적" in md
