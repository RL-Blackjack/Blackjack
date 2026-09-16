"""이 파일은 게임 API가 서버 소유 원칙과 결정 기록을 지키는지 확인한다.
입력: httpx로 보낸 HTTP 요청.
출력: 조작 시도 거부·기록 정확성에 대한 pytest 결과.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from game.config import get_settings  # noqa: E402
from game.db import make_engine, make_session_factory, reset_engine  # noqa: E402
from game.models import Decision, Round  # noqa: E402


@pytest.fixture
def 환경(tmp_path, monkeypatch):
    주소 = f"sqlite:///{tmp_path / 'game.db'}"
    monkeypatch.setenv("BJ_DATABASE_URL", 주소)
    monkeypatch.setenv("BJ_JWT_SECRET", "test-secret-at-least-32-characters-long")
    get_settings.cache_clear()
    reset_engine()

    from game.db import create_schema
    from game.deps import login_limiter
    from game.main import create_app
    from scripts.register_models import sync_registry

    엔진 = make_engine(주소)
    create_schema(엔진)
    공장 = make_session_factory(엔진)
    with 공장() as s:
        sync_registry(s, ROOT / "models")
    login_limiter.reset()

    with TestClient(create_app()) as c:
        yield c, 공장

    엔진.dispose()
    reset_engine()
    get_settings.cache_clear()


def 로그인(클라, 이메일="a@b.com"):
    r = 클라.post("/api/auth/signup",
                 json={"email": 이메일, "password": "hunter2!!", "display_name": 이메일})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def 새게임(클라, 머리):
    r = 클라.post("/api/games", json={}, headers=머리)
    assert r.status_code == 201, r.text
    return r.json()


def 끝까지두기(클라, 머리, 게임):
    """열린 라운드를 스탠드로 끝내고 마지막 응답을 돌려준다."""
    라운드 = 게임["round"]
    마지막 = None
    while 라운드["status"] == "pending":
        r = 클라.post(f"/api/games/{게임['game_id']}/act",
                     json={"seq": 라운드["seq"], "action": 0}, headers=머리)
        assert r.status_code == 200, r.text
        마지막 = r.json()
        라운드 = 마지막["round"]
    return 마지막


# ---------- 모델 목록 ----------

def test_상대_모델_목록을_준다(환경):
    클라, _ = 환경
    r = 클라.get("/api/models")
    assert r.status_code == 200
    모델들 = r.json()
    assert len(모델들) >= 12
    assert {"id", "name", "family", "exact_ev"} <= set(모델들[0])
    # 왜: 가장 잘 두는 모델이 맨 앞이어야 기본 상대가 정해진다.
    assert 모델들 == sorted(모델들, key=lambda m: -m["exact_ev"])


# ---------- 서버가 게임 상태의 주인이다 ----------

def test_클라이언트가_보낸_카드는_무시된다(환경):
    클라, _ = 환경
    머리 = 로그인(클라)
    게임 = 새게임(클라, 머리)
    원래카드 = 게임["round"]["player_cards"]

    r = 클라.post(f"/api/games/{게임['game_id']}/act",
                 json={"seq": 0, "action": 1,
                       "player_cards": [1, 10], "dealer_cards": [2, 2],
                       "net": 999.0, "total": 21},
                 headers=머리)
    assert r.status_code == 200, r.text
    # 왜: 몸체에 카드를 끼워 넣어도 서버는 시드로 재현한 상태만 쓴다. 스키마가
    #     모르는 칸을 무시하므로 200이 나오고, 상태는 조작되지 않는다.
    보기 = r.json()["round"]
    if 보기["status"] == "pending":
        assert 보기["player_cards"][:2] == 원래카드[:2]
        assert 보기["total"] != 21 or 보기["player_cards"] != [1, 10]


def test_카드는_DB에_저장되지_않고_시드로_재현된다(환경):
    클라, 공장 = 환경
    머리 = 로그인(클라)
    게임 = 새게임(클라, 머리)
    with 공장() as s:
        라운드 = s.scalar(select(Round))
        # 왜: 이게 "서버가 상태의 주인"의 구현이다. 카드를 저장하지 않으므로
        #     DB를 고쳐도 게임을 조작할 수 없다.
        assert 라운드.seed > 0
        assert 라운드.dealer_cards == ""      # 아직 안 끝났다
        assert 라운드.dealer_up == 게임["round"]["dealer_up"]


def test_불법_행동은_400이다(환경):
    클라, _ = 환경
    머리 = 로그인(클라)
    for _ in range(40):
        게임 = 새게임(클라, 머리)
        라운드 = 게임["round"]
        if 라운드["status"] == "pending" and 3 not in 라운드["legal"]:
            r = 클라.post(f"/api/games/{게임['game_id']}/act",
                         json={"seq": 0, "action": 3}, headers=머리)
            assert r.status_code == 400, r.text
            return
    pytest.fail("스플릿이 불법인 첫 상태를 40판 안에 못 찾았다")


def test_범위_밖_행동_번호는_422다(환경):
    클라, _ = 환경
    머리 = 로그인(클라)
    게임 = 새게임(클라, 머리)
    r = 클라.post(f"/api/games/{게임['game_id']}/act",
                 json={"seq": 0, "action": 99}, headers=머리)
    assert r.status_code == 422


def test_같은_결정을_두_번_보내면_409다(환경):
    클라, _ = 환경
    머리 = 로그인(클라)
    게임 = 새게임(클라, 머리)
    첫번째 = 클라.post(f"/api/games/{게임['game_id']}/act",
                    json={"seq": 0, "action": 0}, headers=머리)
    assert 첫번째.status_code == 200, 첫번째.text
    # 왜: 새로고침이나 뒤로가기로 같은 요청이 두 번 갈 수 있다. seq가 낙관적
    #     잠금이 되어 두 번째를 막는다.
    두번째 = 클라.post(f"/api/games/{게임['game_id']}/act",
                    json={"seq": 0, "action": 0}, headers=머리)
    assert 두번째.status_code == 409


def test_남의_게임에는_404다(환경):
    클라, _ = 환경
    주인 = 로그인(클라, "owner@b.com")
    게임 = 새게임(클라, 주인)
    남 = 로그인(클라, "other@b.com")

    # 왜 403이 아니라 404인가: 403은 "그 번호의 게임이 존재한다"를 알려 준다.
    #     게임 번호를 훑어 남이 몇 판 했는지 셀 수 있다.
    assert 클라.get(f"/api/games/{게임['game_id']}", headers=남).status_code == 404
    assert 클라.post(f"/api/games/{게임['game_id']}/act",
                    json={"seq": 0, "action": 0}, headers=남).status_code == 404


def test_로그인하지_않으면_401이다(환경):
    클라, _ = 환경
    assert 클라.post("/api/games", json={}).status_code == 401
    assert 클라.get("/api/games/1").status_code == 401


def test_끝난_라운드에_또_두면_409다(환경):
    클라, _ = 환경
    머리 = 로그인(클라)
    게임 = 새게임(클라, 머리)
    마지막 = 끝까지두기(클라, 머리, 게임)
    if 마지막 is None:
        pytest.skip("결정 없이 끝난 라운드다")
    # 왜 seq를 0~60 안에서 보내는가: ActIn.seq 는 le=60 이다. 범위를 넘기면 409에
    #     닿기 전에 입력 검증(422)에서 막혀 이 테스트가 검사하려는 것을 못 본다.
    for 순번 in (0, 1, len(마지막["round"]["hands"])):
        r = 클라.post(f"/api/games/{게임['game_id']}/act",
                     json={"seq": 순번, "action": 0}, headers=머리)
        assert r.status_code == 409, (순번, r.status_code, r.text)


# ---------- 결정 기록 ----------

def test_결정마다_DP_정답과_EV_손실이_저장된다(환경):
    클라, 공장 = 환경
    머리 = 로그인(클라)
    게임 = 새게임(클라, 머리)
    라운드 = 게임["round"]
    if 라운드["status"] != "pending":
        pytest.skip("결정 없이 끝난 라운드다")

    r = 클라.post(f"/api/games/{게임['game_id']}/act",
                 json={"seq": 0, "action": 라운드["legal"][0]}, headers=머리)
    assert r.status_code == 200, r.text
    되돌림 = r.json()["feedback"]

    with 공장() as s:
        d = s.scalar(select(Decision))
        assert d is not None
        assert d.seq == 0
        assert (d.total, d.is_soft, d.dealer_up) == (
            라운드["total"], int(라운드["is_soft"]), 라운드["dealer_up"])
        assert d.action_taken == 라운드["legal"][0]
        assert d.dp_optimal_action == 되돌림["dp_optimal_action"]
        assert d.dp_ev_loss == pytest.approx(되돌림["dp_ev_loss"])
        # 왜 0 이상인가: EV 손실은 최적 대비 손해다. 음수면 DP보다 잘 뒀다는
        #     뜻이 되어 정의가 깨진다.
        assert d.dp_ev_loss >= -1e-12
        assert d.ai_action in (0, 1, 2, 3)


def test_최적_행동을_고르면_EV_손실이_0이다(환경):
    클라, _ = 환경
    머리 = 로그인(클라)
    맞춘적있다 = False
    for _ in range(30):
        게임 = 새게임(클라, 머리)
        라운드 = 게임["round"]
        if 라운드["status"] != "pending":
            continue
        # 먼저 아무거나 눌러 정답을 알아낸 뒤, 다음 판에서 그 상태가 나오면 맞춘다.
        r = 클라.post(f"/api/games/{게임['game_id']}/act",
                     json={"seq": 0, "action": 라운드["legal"][0]}, headers=머리)
        피드백 = r.json()["feedback"]
        if 피드백["was_optimal"]:
            assert 피드백["dp_ev_loss"] == pytest.approx(0.0, abs=1e-12)
            맞춘적있다 = True
    assert 맞춘적있다, "30판을 뒀는데 최적 행동을 한 번도 못 골랐다"


def test_라운드가_끝나면_딜러_카드와_순손익이_저장된다(환경):
    클라, 공장 = 환경
    머리 = 로그인(클라)
    게임 = 새게임(클라, 머리)
    끝까지두기(클라, 머리, 게임)

    with 공장() as s:
        r = s.scalar(select(Round))
        assert r.is_open is False
        assert r.dealer_cards != ""
        assert len(r.dealer_cards.split(",")) >= 2
        assert r.net in (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)


def test_한_게임에서_여러_라운드를_둔다(환경):
    클라, 공장 = 환경
    머리 = 로그인(클라)
    게임 = 새게임(클라, 머리)
    끝까지두기(클라, 머리, 게임)

    for 회차 in range(2, 5):
        r = 클라.post(f"/api/games/{게임['game_id']}/rounds", headers=머리)
        assert r.status_code == 201, r.text
        assert r.json()["round_no"] == 회차
        끝까지두기(클라, 머리, {"game_id": 게임["game_id"], "round": r.json()["round"]})

    with 공장() as s:
        assert len(list(s.scalars(select(Round)))) == 4


def test_라운드가_열려_있으면_새_라운드를_못_연다(환경):
    클라, _ = 환경
    머리 = 로그인(클라)
    게임 = 새게임(클라, 머리)
    if 게임["round"]["status"] != "pending":
        pytest.skip("결정 없이 끝난 라운드다")
    # 왜: 라운드 두 개가 동시에 열려 있으면 어느 쪽에 두는지 알 수 없다.
    assert 클라.post(f"/api/games/{게임['game_id']}/rounds",
                    headers=머리).status_code == 409


def test_게임에_규칙_지문이_박힌다(환경):
    클라, 공장 = 환경
    from blackjack_rl.rules import RULES_V1
    머리 = 로그인(클라)
    게임 = 새게임(클라, 머리)
    from game.models import Game
    with 공장() as s:
        assert s.get(Game, 게임["game_id"]).rules_fp == RULES_V1.fingerprint()


def test_상대_모델을_고를_수_있다(환경):
    클라, _ = 환경
    머리 = 로그인(클라)
    모델들 = 클라.get("/api/models").json()
    고른것 = 모델들[-1]
    r = 클라.post("/api/games", json={"opponent_model_id": 고른것["id"]}, headers=머리)
    assert r.status_code == 201, r.text
    assert r.json()["opponent"]["name"] == 고른것["name"]


def test_없는_모델을_고르면_404다(환경):
    클라, _ = 환경
    머리 = 로그인(클라)
    r = 클라.post("/api/games", json={"opponent_model_id": 999999}, headers=머리)
    assert r.status_code == 404


def test_게임을_끝낼_수_있다(환경):
    클라, 공장 = 환경
    머리 = 로그인(클라)
    게임 = 새게임(클라, 머리)
    끝까지두기(클라, 머리, 게임)
    assert 클라.post(f"/api/games/{게임['game_id']}/finish",
                    headers=머리).status_code == 204
    from game.models import Game
    with 공장() as s:
        assert s.get(Game, 게임["game_id"]).ended_at is not None
