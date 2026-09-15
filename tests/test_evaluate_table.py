"""이 파일은 scripts/evaluate.py가 만드는 최종 평가표의 숫자와 열을 검증한다
입력: 최소 키만 담은 가짜 npz
출력: pytest 통과/실패
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import evaluate  # noqa: E402

from blackjack_rl.eval.simulate import SanityAlarm  # noqa: E402
from blackjack_rl.rules import RULES_V1  # noqa: E402


def 가짜_실행(path, name, algo, seed, episodes, ev, wall_sec):
    """evaluate.py가 읽는 세 배열/키만 가진 최소 npz를 만든다.
    meta의 top-level 키 이름은 train/runner.py가 실제로 쓰는 것과 같다."""
    meta = json.dumps({
        "name": name, "algo": algo, "seed": seed,
        "n_episodes": int(episodes[-1]), "wall_sec": wall_sec,
        "rules_fp": RULES_V1.fingerprint(),
    }, ensure_ascii=False)
    np.savez_compressed(
        path,
        episodes=np.asarray(episodes, dtype=np.int64),
        ev_greedy=np.asarray(ev, dtype=np.float32),
        meta=np.array(meta),
    )


def test_목표EV에_처음_도달한_프레임의_에피소드를_집는다(tmp_path):
    경로 = tmp_path / "a.npz"
    가짜_실행(경로, "mc_natural", "mc", 1,
             [1000, 10000, 100000], [-0.30, -0.02, -0.008], 12.5)
    행 = evaluate.read_run_row(경로, target_ev=-0.010)
    assert 행.algo == "mc"
    assert 행.seed == 1
    assert 행.episodes_to_target == 100000
    assert 행.final_ev == pytest.approx(-0.008, abs=1e-6)
    assert 행.wall_sec == pytest.approx(12.5)


def test_목표에_한_번도_못_닿으면_None이다(tmp_path):
    경로 = tmp_path / "a.npz"
    가짜_실행(경로, "mc_natural", "mc", 1, [1000, 100000], [-0.30, -0.05], 9.0)
    assert evaluate.read_run_row(경로, target_ev=-0.010).episodes_to_target is None


def test_시드가_둘이면_시드간_표준편차가_계산된다(tmp_path):
    가짜_실행(tmp_path / "a.npz", "mc_natural", "mc", 1,
             [1000, 100000], [-0.30, -0.008], 10.0)
    가짜_실행(tmp_path / "b.npz", "mc_natural", "mc", 2,
             [1000, 100000], [-0.30, -0.010], 11.0)
    행들 = [evaluate.read_run_row(p, -0.010) for p in sorted(tmp_path.glob("*.npz"))]
    묶음 = evaluate.group_rows(행들)
    assert len(묶음) == 1
    assert 묶음[0]["n_seeds"] == 2
    # 표본표준편차(ddof=1) = |(-0.008) - (-0.010)| / sqrt(2)
    assert 묶음[0]["final_ev_std"] == pytest.approx(0.002 / (2 ** 0.5), abs=1e-6)
    assert 묶음[0]["final_ev_mean"] == pytest.approx(-0.009, abs=1e-6)
    assert 묶음[0]["wall_sec_mean"] == pytest.approx(10.5)


def test_시드가_하나면_표준편차는_None이다(tmp_path):
    가짜_실행(tmp_path / "a.npz", "q_natural", "q", 1,
             [1000, 100000], [-0.30, -0.009], 9.0)
    묶음 = evaluate.group_rows([evaluate.read_run_row(tmp_path / "a.npz", -0.010)])
    # 왜: 0.0이라고 적으면 '시드 간 분산이 없다'는 거짓말이 된다.
    assert 묶음[0]["final_ev_std"] is None


def test_기준_정책_EV가_실측값과_맞는다():
    표 = dict(evaluate.baseline_evs(RULES_V1))
    # 왜: 아래 네 값은 전부 dp.exact로 직접 계산해 확인한 값이다.
    #     문헌값(딜러 모방 -5.96%, 항상 스탠드 -15.54%)과 다른 이유는
    #     문헌값의 규칙 전제가 우리와 다르기 때문이다. 우리 숫자를 쓴다.
    assert 표["DP 최적(상한)"] == pytest.approx(-0.005108, abs=2e-6)
    assert 표["딜러 모방"] == pytest.approx(-0.056746, abs=2e-6)
    assert 표["항상 스탠드"] == pytest.approx(-0.160377, abs=2e-6)
    assert 표["무작위(합법 균등)"] == pytest.approx(-0.460235, abs=2e-6)


def test_파라미터_수와_메모리가_실측값과_맞는다():
    # 왜: 설계서 §8.3의 보고 표에 '파라미터 수'와 '메모리' 열이 있다.
    #     도달 가능 (s,a)는 1,680개이고 Q_SHAPE는 33,792칸 x float64 = 270,336 B다.
    #     doubleq만 표가 둘이라 정확히 두 배다.
    assert evaluate.N_PARAMS_TABULAR == 1_680
    assert evaluate.Q_BYTES_FLOAT64 == 270_336
    assert evaluate.params_and_memory("mc") == (1_680, 270_336)
    assert evaluate.params_and_memory("doubleq") == (3_360, 540_672)


def test_마크다운표에_설계서가_요구한_열이_전부_있다(tmp_path):
    가짜_실행(tmp_path / "a.npz", "mc_natural", "mc", 1,
             [1000, 100000], [-0.30, -0.008], 10.0)
    묶음 = evaluate.group_rows([evaluate.read_run_row(tmp_path / "a.npz", -0.010)])
    본문 = evaluate.render_table(묶음, -0.010, evaluate.baseline_evs(RULES_V1))
    for 열 in ("알고리즘", "최종 EV", "목표 EV 도달", "시드 간 표준편차",
              "파라미터 수", "메모리", "wall_sec"):
        assert 열 in 본문
    assert "-0.005108" in 본문        # DP 상한이 같은 표에 보인다


def test_main이_리포트_json을_남긴다(tmp_path):
    가짜_실행(tmp_path / "a.npz", "mc_natural", "mc", 1,
             [1000, 100000], [-0.30, -0.008], 10.0)
    리포트 = tmp_path / "rep.json"
    코드 = evaluate.main(["--artifacts", str(tmp_path), "--out", str(리포트)])
    assert 코드 == 0
    실린것 = json.loads(리포트.read_text(encoding="utf-8"))
    assert 실린것["target_ev"] == pytest.approx(-0.010)
    assert 실린것["groups"][0]["algo"] == "mc"
    assert 실린것["groups"][0]["n_params"] == 1_680
    assert 실린것["dp_optimal_ev"] == pytest.approx(-0.005108, abs=2e-6)


def test_스냅샷이_없으면_1을_돌려준다(tmp_path):
    assert evaluate.main(["--artifacts", str(tmp_path),
                          "--out", str(tmp_path / "r.json")]) == 1


def test_EV가_너무_좋으면_표를_만들기_전에_경보가_터진다(tmp_path):
    가짜_실행(tmp_path / "bug.npz", "bug", "mc", 1,
             [1000, 100000], [-0.30, 0.02], 5.0)
    with pytest.raises(SanityAlarm):
        evaluate.main(["--artifacts", str(tmp_path), "--out", str(tmp_path / "r.json")])
