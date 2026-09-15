"""이 파일은 scripts/train.py의 플래그 해석·산출물 저장·라이브 화면을 검증한다
입력: scripts/train.py의 main / render_text_chart / live_banner
출력: pytest 통과/실패
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train  # noqa: E402

from blackjack_rl.chartspec import CHART_ROWS, DEALER_COLS, ROW_INDEX  # noqa: E402
from blackjack_rl.dp.exact import optimal_chart  # noqa: E402
from blackjack_rl.rules import RULES_V1  # noqa: E402
from blackjack_rl.train.config import (HEADLINE_EPISODES, build_parser,  # noqa: E402
                                       config_from_args)
from blackjack_rl.train.snapshot import load_run, q_sha256  # noqa: E402

기본인자 = ["--name", "mc_cli", "--algo", "mc", "--seed", "42",
         "--episodes", "3000", "--frames", "10", "--agreement-hands", "0"]


def test_기본_플래그가_설정_객체로_옮겨진다():
    args = build_parser().parse_args(
        ["--algo", "q", "--seed", "7", "--episodes", "20000",
         "--start-dist", "exploring", "--frames", "4"])
    cfg = config_from_args(args)
    assert cfg.algo == "q"
    assert cfg.seed == 7
    assert cfg.n_episodes == 20_000
    assert cfg.start_dist == "exploring"
    assert cfg.n_frames == 4
    assert cfg.rules == RULES_V1


def test_라이브_배너가_축소_배율을_솔직하게_적는다():
    args = build_parser().parse_args(["--live", "--live-seconds", "60"])
    cfg = config_from_args(args)
    배너 = train.live_banner(cfg)
    # 3,000만 / 360만 = 8.33 -> "1/8"
    assert f"{HEADLINE_EPISODES:,}" in 배너
    assert "1/8" in 배너
    assert "로컬 전용" in 배너


def test_텍스트표는_머리줄_한_개와_36행이다():
    표 = train.render_text_chart(optimal_chart(RULES_V1).notation)
    줄들 = 표.splitlines()
    assert len(줄들) == 1 + len(CHART_ROWS)
    assert 줄들[0].split()[1:] == [str(c) for c in DEALER_COLS]

    # 왜: 아래 두 칸은 DP표와 출판표가 모두 같은 글자인 것을 직접 확인한 칸이다.
    #     ROW_INDEX['16']=11, ROW_INDEX['8,8']=33, DEALER_COLS.index(10)=8.
    하드16 = 줄들[1 + ROW_INDEX["16"]].split()
    assert 하드16[0] == "16"
    assert 하드16[1 + 8] == "H"

    페어88 = 줄들[1 + ROW_INDEX["8,8"]].split()
    assert 페어88[0] == "8,8"
    assert 페어88[1 + 8] == "Y"


def test_명령줄_한_줄이_npz와_json_두_파일을_남긴다(tmp_path):
    코드 = train.main([*기본인자, "--artifacts", str(tmp_path)])
    assert 코드 == 0

    npz들 = sorted(tmp_path.glob("*.npz"))
    json들 = sorted(tmp_path.glob("*.json"))
    assert len(npz들) == 1 and len(json들) == 1
    assert npz들[0].stem == json들[0].stem

    조각 = npz들[0].stem.split("__")
    assert len(조각) == 4
    assert 조각[0] == "mc_cli"
    assert 조각[3] == "seed42"


def test_남은_json은_설정_전문이고_npz_meta와_지문이_같다(tmp_path):
    train.main([*기본인자, "--artifacts", str(tmp_path)])
    설정 = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    art = load_run(next(tmp_path.glob("*.npz")))
    assert 설정["cfg_fp"] == art.meta["cfg_fp"]
    assert 설정["rules_fp"] == art.meta["rules_fp"]
    assert 설정["n_episodes"] == 3000


def test_같은_명령을_두_번_돌리면_q_해시가_같다(tmp_path):
    첫번째, 두번째 = tmp_path / "a", tmp_path / "b"
    train.main([*기본인자, "--artifacts", str(첫번째)])
    train.main([*기본인자, "--artifacts", str(두번째)])
    a = load_run(next(첫번째.glob("*.npz")))
    b = load_run(next(두번째.glob("*.npz")))
    # 왜: 설계서 §8.1의 재현성 1번 장치(시드 4스트림 분리)가 실제로 먹는지 본다.
    assert a.meta["q_sha256"] == b.meta["q_sha256"]
    assert np.array_equal(a.q_final, b.q_final)


def test_저장된_npz의_q_final을_다시_해시하면_meta와_맞는다(tmp_path):
    train.main([*기본인자, "--artifacts", str(tmp_path)])
    art = load_run(next(tmp_path.glob("*.npz")))
    assert q_sha256(art.q_final) == art.meta["q_sha256"]


def test_일치율_훅이_꽂혀서_agree_a가_nan이_아니다(tmp_path):
    """훅을 꽂는 곳이 없으면 모든 산출물의 agree_a/agree_b가 영구히 nan이다."""
    train.main(["--name", "mc_hook", "--episodes", "3000", "--frames", "5",
                "--agreement-hands", "5000", "--artifacts", str(tmp_path)])
    art = load_run(next(tmp_path.glob("*.npz")))
    assert not np.any(np.isnan(art.agree_a))
    assert not np.any(np.isnan(art.agree_b))
    assert np.all((art.agree_a >= 0.0) & (art.agree_a <= 1.0))
    assert np.all((art.agree_b >= 0.0) & (art.agree_b <= 1.0))


def test_일치율_핸드가_0이면_훅을_끄고_nan으로_남는다(tmp_path):
    train.main([*기본인자, "--artifacts", str(tmp_path)])
    art = load_run(next(tmp_path.glob("*.npz")))
    assert np.all(np.isnan(art.agree_a))


def test_라이브는_스냅샷을_남기지_않는다(tmp_path, capsys):
    코드 = train.main(["--live", "--live-seconds", "1", "--live-every", "1",
                     "--artifacts", str(tmp_path)])
    assert 코드 == 0
    assert list(tmp_path.glob("*.npz")) == []
    찍힌글 = capsys.readouterr().out
    assert "[라이브]" in 찍힌글
    # 학습이 끝난 뒤 표를 한 번 그린다(콜백으로 표를 넘기지 않는다).
    assert "8,8" in 찍힌글


def test_스크립트를_그냥_실행해도_패키지를_찾는다():
    # 왜: 발표 중에는 PYTHONPATH를 손으로 넣을 시간이 없다. 맨손으로 쳐서 돌아야 한다.
    뿌리 = Path(__file__).resolve().parents[1]
    환경 = dict(os.environ, PYTHONIOENCODING="utf-8")
    for 이름 in ("train.py", "run_dp.py"):
        결과 = subprocess.run(
            [sys.executable, str(뿌리 / "scripts" / 이름), "--help"],
            capture_output=True, text=True, encoding="utf-8", env=환경)
        assert 결과.returncode == 0, f"{이름}: {결과.stderr}"
