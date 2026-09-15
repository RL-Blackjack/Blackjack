"""이 파일은 발표용 한 줄 명령 scripts/explain_state.py의 출력 내용을 검증한다
입력: scripts/explain_state.py의 parse_bool / find_row / explain / main
출력: pytest 통과/실패
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import explain_state  # noqa: E402

from blackjack_rl.state import Q_SHAPE  # noqa: E402


def test_참거짓을_여러_표기로_읽는다():
    assert explain_state.parse_bool("true") == 1
    assert explain_state.parse_bool("True") == 1
    assert explain_state.parse_bool("yes") == 1
    assert explain_state.parse_bool("1") == 1
    assert explain_state.parse_bool("false") == 0
    assert explain_state.parse_bool("n") == 0
    assert explain_state.parse_bool("0") == 0
    with pytest.raises(ValueError):
        explain_state.parse_bool("아마도")


def test_표_행을_라벨로_되찾는다():
    assert explain_state.find_row(16, 0, 1).label == "8,8"
    assert explain_state.find_row(16, 0, 0).label == "16"
    assert explain_state.find_row(18, 1, 0).label == "A,7"
    assert explain_state.find_row(12, 1, 1).label == "A,A"
    with pytest.raises(ValueError):
        # 하드 4는 반드시 2,2 페어라 페어 구역에만 있다(chartspec의 하드는 5부터).
        explain_state.find_row(4, 0, 0)


def test_88_vs_10을_한_화면에_출력한다(capsys):
    코드 = explain_state.main(
        ["--total", "16", "--dealer", "10", "--soft", "false", "--pair", "true"])
    assert 코드 == 0
    찍힌글 = capsys.readouterr().out

    # 왜: 아래 네 숫자는 dp.exact.action_values(StateKey(16,0,10,1,1,0))를 직접
    #     돌려 얻은 실측값이다. 추정이 아니다.
    #     [STAND -0.540430, HIT -0.539826, DOUBLE -1.079653, SPLIT -0.480552]
    assert "-0.5404" in 찍힌글
    assert "-0.5398" in 찍힌글
    assert "-1.0797" in 찍힌글
    assert "-0.4806" in 찍힌글

    assert "8,8" in 찍힌글
    assert "vs 딜러 10" in 찍힌글

    최선줄 = [줄 for 줄 in 찍힌글.splitlines() if "DP 최선" in 줄]
    assert len(최선줄) == 1
    assert "스플릿" in 최선줄[0]

    기준줄 = [줄 for 줄 in 찍힌글.splitlines() if "기준표" in 줄][0]
    assert 기준줄.strip().endswith("Y")


def test_하드16_vs_10은_통계적_동점으로_표시된다(capsys):
    explain_state.main(
        ["--total", "16", "--dealer", "10", "--soft", "false", "--pair", "false"])
    찍힌글 = capsys.readouterr().out
    # 왜: |(-0.539826) - (-0.540430)| = 0.000604 이고 TIE_DELTA(0.005)보다 작다.
    #     "AI가 못 푼 칸 = 인간도 60년 논쟁한 칸"이라는 서사가 여기서 나온다.
    assert "0.000604" in 찍힌글
    assert "통계적 동점" in 찍힌글


def test_스플릿이_불법인_칸은_불법이라고_적는다(capsys):
    explain_state.main(
        ["--total", "18", "--dealer", "11", "--soft", "true", "--pair", "false"])
    찍힌글 = capsys.readouterr().out
    assert "A,7" in 찍힌글
    assert "vs 딜러 A" in 찍힌글      # 딜러 11을 A로 적어 준다
    # 왜: 머리말의 '스플릿 불가' 줄도 '스플릿'을 품고 있으므로 행동표의 P 행만 집는다.
    스플릿줄 = [줄 for 줄 in 찍힌글.splitlines() if "P(스플릿)" in 줄][0]
    assert "불법" in 스플릿줄


def test_스냅샷이_없어도_DP만으로_돈다(capsys):
    코드 = explain_state.main(
        ["--total", "11", "--dealer", "6", "--soft", "false", "--pair", "false"])
    assert 코드 == 0
    찍힌글 = capsys.readouterr().out
    assert "DP 최적표" in 찍힌글
    assert "학습 스냅샷 없음" in 찍힌글


def test_스냅샷을_주면_학습Q와_방문수도_같이_나온다(tmp_path, capsys):
    q = np.full(Q_SHAPE, -np.inf, dtype=np.float32)
    n = np.zeros(Q_SHAPE, dtype=np.int32)
    칸 = (16, 0, 10, 1, 1, 0)
    q[칸] = np.array([-0.55, -0.53, -1.10, -0.48], dtype=np.float32)
    n[칸] = np.array([120, 130, 40, 900], dtype=np.int32)

    경로 = tmp_path / "가짜.npz"
    np.savez_compressed(경로, q_final=q, n_final=n)

    explain_state.main([
        "--total", "16", "--dealer", "10", "--soft", "false", "--pair", "true",
        "--run", str(경로)])
    찍힌글 = capsys.readouterr().out
    assert "-0.4800" in 찍힌글      # 학습 Q
    assert "900" in 찍힌글          # 방문수 N
    assert "가짜.npz" in 찍힌글


def test_실제_스냅샷의_키_이름만_읽는다(tmp_path):
    """RunArtifact 타입이 아니라 q_final/n_final 두 키만 본다.
    스냅샷 스키마가 늘어나도 이 스크립트는 안 고쳐도 된다."""
    q = np.full(Q_SHAPE, -np.inf, dtype=np.float32)
    n = np.zeros(Q_SHAPE, dtype=np.int32)
    경로 = tmp_path / "최소.npz"
    np.savez_compressed(경로, q_final=q, n_final=n)
    학습q, 방문 = explain_state.load_q_and_n(경로)
    assert 학습q.shape == Q_SHAPE
    assert 방문.shape == Q_SHAPE


def test_발표용_한_줄_명령이_맨손으로_돈다():
    # 왜: 설계서 §9가 약속한 그 명령 그대로다. 앱이 죽어도 이건 돌아야 한다.
    뿌리 = Path(__file__).resolve().parents[1]
    환경 = dict(os.environ, PYTHONIOENCODING="utf-8")
    결과 = subprocess.run(
        [sys.executable, str(뿌리 / "scripts" / "explain_state.py"),
         "--total", "16", "--dealer", "10", "--soft", "false", "--pair", "true"],
        capture_output=True, text=True, encoding="utf-8", env=환경)
    assert 결과.returncode == 0, 결과.stderr
    assert "-0.4806" in 결과.stdout
