"""이 파일은 저장된 npz에서 GIF·학습곡선·헤드라인이 실제로 나오는지 검증한다
입력: runner가 만든 작은 산출물 두 개
출력: pytest 통과/실패
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import export_media  # noqa: E402

from blackjack_rl.chartspec import ChartTable  # noqa: E402
from blackjack_rl.rules import RULES_V1  # noqa: E402
from blackjack_rl.train.config import ExperimentConfig  # noqa: E402
from blackjack_rl.train.runner import run  # noqa: E402
from blackjack_rl.train.snapshot import artifact_path, save_run  # noqa: E402


def 산출물(tmp_path, name, algo):
    cfg = ExperimentConfig(name=name, algo=algo, rules=RULES_V1, seed=42,
                           n_episodes=3_000, n_frames=6)
    art = run(cfg)
    save_run(artifact_path(cfg, tmp_path), art)
    return art


@pytest.fixture(scope="module")
def 두개(tmp_path_factory):
    폴더 = tmp_path_factory.mktemp("artifacts")
    a = 산출물(폴더, "mc_smoke", "mc")
    b = 산출물(폴더, "q_smoke", "q")
    return 폴더, {"MC": a, "Q러닝": b}


def test_프레임에서_ChartTable을_되살린다(두개):
    _폴더, arts = 두개
    art = arts["MC"]
    표 = export_media.chart_of(art, -1)
    assert isinstance(표, ChartTable)
    assert 표.notation.shape == (36, 10)
    assert 표.action.shape == (36, 10)
    assert 표.rules_fp == RULES_V1.fingerprint()
    # 왜 이 확인인가: chart_action만으로는 D와 Ds가 둘 다 2라 구분이 안 된다.
    #   되살린 표의 표기 집합이 chart_notation과 글자 단위로 같아야 한다.
    assert np.array_equal(표.notation, art.chart_notation[-1])


def test_프레임_솎기는_처음과_끝을_반드시_포함한다():
    골라낸것 = export_media.pick_frames(200, 40)
    assert len(골라낸것) == 40
    assert 골라낸것[0] == 0
    assert 골라낸것[-1] == 199
    assert 골라낸것 == sorted(set(골라낸것))
    # 프레임 수보다 많이 달라고 하면 전부 준다.
    assert export_media.pick_frames(6, 40) == list(range(6))


def test_GIF가_요청한_장수만큼_생긴다(두개, tmp_path):
    _폴더, arts = 두개
    출력 = tmp_path / "anim.gif"
    export_media.make_gif(arts["MC"], 출력, overlay=export_media.overlay_chart(), k=4)
    assert 출력.exists()
    with Image.open(출력) as im:
        assert im.format == "GIF"
        assert im.n_frames == 4


def test_헤드라인은_네_숫자를_준다(두개):
    _폴더, arts = 두개
    보고 = export_media.headline(arts["MC"], *export_media.shared_inputs(n_freq=5_000))
    assert 0.0 <= 보고.tier_a <= 1.0
    assert 0.0 <= 보고.tier_b <= 1.0
    assert 보고.ev_loss_pp >= 0.0
    # 왜 30 이상인가: 하드 20 / 하드 21 / A,10 세 행은 학습된 Q로 채울 수 없다.
    #   하드 21 행은 REACHABLE_KEYS에 아예 없고 나머지 둘은 ExploringStarts도
    #   만들지 못한다. 3,000 에피소드짜리 산출물이면 더 많이 남는다.
    assert 보고.n_undecided >= 30


def test_리포트_json에_네_숫자와_불일치_목록이_들어간다(두개, tmp_path):
    _폴더, arts = 두개
    art = arts["MC"]
    보고 = export_media.headline(art, *export_media.shared_inputs(n_freq=5_000))
    경로 = export_media.write_report(art, 보고, tmp_path)
    assert 경로.name == f"{art.meta['cfg_fp']}.json"
    본문 = json.loads(경로.read_text(encoding="utf-8"))
    assert set(본문["headline"]) == {"tier_a", "tier_b", "ev_loss_pp", "n_undecided"}
    assert 본문["headline"]["n_undecided"] == 보고.n_undecided
    assert isinstance(본문["mismatches"], list)
    assert 본문["cfg_fp"] == art.meta["cfg_fp"]


def test_학습곡선과_편향곡선_PNG가_생긴다(두개, tmp_path):
    _폴더, arts = 두개
    그림들 = export_media.make_curves(arts, tmp_path)
    assert len(그림들) == 2
    for 경로 in 그림들:
        assert Path(경로).exists()
        with Image.open(경로) as im:
            assert im.format == "PNG"


def test_에피소드_축이_다른_산출물을_섞으면_거부한다(두개, tmp_path):
    _폴더, arts = 두개
    섞인것 = dict(arts)
    가짜 = arts["MC"]
    가짜.episodes = 가짜.episodes + 1        # x축을 한 칸 밀어 버린다
    섞인것["망친것"] = 가짜
    with pytest.raises(ValueError):
        export_media.make_curves(섞인것, tmp_path)


def test_main이_미디어와_리포트를_남긴다(두개, tmp_path):
    폴더, _arts = 두개
    코드 = export_media.main(["--artifacts", str(폴더),
                             "--media", str(tmp_path / "media"),
                             "--reports", str(tmp_path / "reports"),
                             "--gif-frames", "3", "--freq-hands", "5000"])
    assert 코드 == 0
    assert len(list((tmp_path / "media").glob("*.gif"))) == 2
    assert (tmp_path / "media" / "learning_curves.png").exists()
    assert (tmp_path / "media" / "bias_curves.png").exists()
    assert len(list((tmp_path / "reports").glob("*.json"))) == 2


def test_스냅샷이_없으면_1을_돌려준다(tmp_path):
    assert export_media.main(["--artifacts", str(tmp_path),
                              "--media", str(tmp_path / "m"),
                              "--reports", str(tmp_path / "r")]) == 1
