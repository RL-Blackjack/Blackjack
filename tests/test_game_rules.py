"""이 파일은 열린 라운드 규칙·규칙 지문·입력 경계·상대 모델 무결성을 확인한다.
입력: httpx로 보낸 HTTP 요청과 optimal 모듈 직접 호출.
출력: 409·422 응답과 EV 손실 계산에 대한 pytest 결과.
"""

import sys
from pathlib import Path

import pytest
from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from blackjack_rl.rules import RULES_H17  # noqa: E402
from blackjack_rl.state import STAND, StateKey, legal_actions  # noqa: E402
from game.models import Decision, Game, ModelRegistry, Round  # noqa: E402
from game.optimal import ev_loss, optimal_action  # noqa: E402
# 왜 테스트 파일에서 가져오는가: 픽스처와 도우미를 파일마다 베끼면 세 벌이 따로
#   낡는다(가입 제한기 리셋 한 줄을 빠뜨리면 그 파일만 429로 무너진다). conftest는
#   이 태스크의 소유 파일이 아니라 한 곳에 두고 나눠 쓴다.
from test_game_api import (  # noqa: E402
    끝까지두기, 다음라운드, 대기라운드, 로그인, 새게임, 환경)  # noqa: F401

# 왜 이 시드인가: 2,2 대 6에서 SPLIT, SPLIT, STAND 를 두면 hand_index 3의 실제
#   스플릿 깊이가 2다(누락 탐색 hunt[3] 재현 시드). 깊이 2 Q 기준 STAND 손실은
#   0.3485, 깊이 0 Q 기준은 0.3811이라 어느 표를 봤는지 값으로 갈린다.
깊이시드 = 351727117773760389
깊이키 = StateKey(total=4, is_soft=0, dealer_up=6, can_double=1, can_split=1,
                split_depth=0)


def 상세(응답):
    """409 응답의 detail 객체."""
    return 응답.json()["detail"]


def test_불법_행동은_400이다(환경):
    클라, _ = 환경
    머리 = 로그인(클라)
    게임 = 새게임(클라, 머리)
    for _ in range(40):
        라운드 = 게임["round"]
        if 라운드["status"] == "pending" and 3 not in 라운드["legal"]:
            r = 클라.post(f"/api/games/{게임['game_id']}/act",
                         json={"seq": 라운드["seq"], "action": 3}, headers=머리)
            assert r.status_code == 400, r.text
            return
        # 왜 새 게임이 아니라 새 라운드인가: 대기 라운드를 두고 게임을 또 만들면
        #   이제 409다(결정 건너뛰기 금지). 같은 게임에서 라운드를 넘긴다.
        게임 = 다음라운드(클라, 머리, 게임)
    pytest.fail("스플릿이 불법인 첫 상태를 40판 안에 못 찾았다")


def test_라운드가_열려_있으면_새_라운드를_못_연다(환경):
    클라, _ = 환경
    머리 = 로그인(클라)
    게임 = 대기라운드(클라, 머리, 새게임(클라, 머리))
    # 왜: 라운드 두 개가 동시에 열려 있으면 어느 쪽에 두는지 알 수 없다.
    r = 클라.post(f"/api/games/{게임['game_id']}/rounds", headers=머리)
    assert r.status_code == 409, r.text
    assert 상세(r)["code"] == "open_round"


def test_새_게임으로_결정을_건너뛸_수_없다(환경):
    클라, _ = 환경
    머리 = 로그인(클라)
    게임 = 대기라운드(클라, 머리, 새게임(클라, 머리))
    r = 클라.post("/api/games", json={}, headers=머리)
    # 왜: 새 게임으로 대기 라운드를 버리면 그 결정이 기록되지 않는다. 오답을 지우는
    #   길이 열려 있으면 일치율이 아무 뜻도 없어진다.
    assert r.status_code == 409, r.text
    assert 상세(r)["code"] == "open_round"
    assert 상세(r)["game_id"] == 게임["game_id"]


def test_다른_게임의_딜로도_결정을_건너뛸_수_없다(환경):
    클라, _ = 환경
    머리 = 로그인(클라)
    먼저 = 끝까지두기(클라, 머리, 새게임(클라, 머리))
    나중 = 대기라운드(클라, 머리, 새게임(클라, 머리))
    r = 클라.post(f"/api/games/{먼저['game_id']}/rounds", headers=머리)
    assert r.status_code == 409, r.text
    assert 상세(r)["code"] == "open_round"
    assert 상세(r)["game_id"] == 나중["game_id"]


def test_브라우저를_닫았다_와도_이어_둘_수_있다(환경):
    클라, _ = 환경
    머리 = 로그인(클라)
    게임 = 대기라운드(클라, 머리, 새게임(클라, 머리))
    r = 클라.post("/api/games", json={}, headers=머리)
    assert r.status_code == 409
    # 왜 game_id를 함께 주는가: 프런트엔드가 "이어 두기" 버튼을 만들 수 있어야 한다.
    다시 = 클라.get(f"/api/games/{상세(r)['game_id']}", headers=머리)
    assert 다시.status_code == 200, 다시.text
    assert 다시.json()["round"]["player_cards"] == 게임["round"]["player_cards"]
    assert 다시.json()["round"]["seq"] == 게임["round"]["seq"]


def test_열린_라운드가_있으면_게임을_끝낼_수_없다(환경):
    클라, _ = 환경
    머리 = 로그인(클라)
    게임 = 대기라운드(클라, 머리, 새게임(클라, 머리))
    r = 클라.post(f"/api/games/{게임['game_id']}/finish", headers=머리)
    # 왜: 열린 라운드를 둔 채 끝내면 그 결정이 기록 없이 사라진다.
    assert r.status_code == 409, r.text
    assert 상세(r)["code"] == "open_round"
    끝까지두기(클라, 머리, 게임)
    assert 클라.post(f"/api/games/{게임['game_id']}/finish",
                    headers=머리).status_code == 204


def test_사용자마다_자기_라운드를_따로_연다(환경):
    클라, 공장 = 환경
    첫째, 둘째 = 로그인(클라, "one@b.com"), 로그인(클라, "two@b.com")
    게임1 = 대기라운드(클라, 첫째, 새게임(클라, 첫째))
    # 왜 첫째의 대기 라운드를 둔 채 만드는가: 라운드 행의 user_id 가 게임 주인이
    #   아니라 상수(예: 1)면 사용자 단위 유일 인덱스가 둘째의 첫 딜을 409로 막는다.
    #   첫째 외 전원이 영원히 잠기는 그 변이가 기존 테스트 38개를 전부 통과했다.
    게임2 = 대기라운드(클라, 둘째, 새게임(클라, 둘째))
    assert 게임1["game_id"] != 게임2["game_id"]
    assert 게임1["round"]["status"] == 게임2["round"]["status"] == "pending"
    # 왜: 열린 라운드 잠금은 사용자 단위다. 둘째를 막는 game_id는 둘째 자신의 것이다.
    r = 클라.post("/api/games", json={}, headers=둘째)
    assert r.status_code == 409, r.text
    assert 상세(r)["game_id"] == 게임2["game_id"]
    끝까지두기(클라, 둘째, 게임2)          # 닫힌 행도 같은 불변식을 지켜야 한다
    with 공장() as s:
        라운드들 = list(s.scalars(select(Round)))
        assert len(라운드들) >= 2
        for 행 in 라운드들:
            assert 행.user_id == s.get(Game, 행.game_id).user_id, (행.id, 행.user_id)
        주인들 = {s.get(Game, g["game_id"]).user_id for g in (게임1, 게임2)}
        assert len(주인들) == 2 and {행.user_id for 행 in 라운드들} == 주인들


def test_끝난_게임에서는_둘_수_없다(환경):
    클라, _ = 환경
    머리 = 로그인(클라)
    게임 = 끝까지두기(클라, 머리, 새게임(클라, 머리))
    assert 클라.post(f"/api/games/{게임['game_id']}/finish",
                    headers=머리).status_code == 204
    # 왜: 끝낸 뒤에도 딜이나 행동이 되면 전적에 잡히지 않는 라운드가 생긴다.
    행동 = 클라.post(f"/api/games/{게임['game_id']}/act",
                  json={"seq": 0, "action": 0}, headers=머리)
    assert 행동.status_code == 409, 행동.text
    assert 상세(행동)["code"] == "game_ended"
    딜 = 클라.post(f"/api/games/{게임['game_id']}/rounds", headers=머리)
    assert 딜.status_code == 409, 딜.text
    assert 상세(딜)["code"] == "game_ended"


def test_거대한_게임_번호는_422다(환경):
    클라, _ = 환경
    머리 = 로그인(클라)
    # 왜: 파이썬 int는 무한정이라 BigInteger 범위를 넘는 번호가 DB 드라이버에
    #   닿아 500이 났다. 경로에서 걸러 422로 돌려준다.
    for 번호 in ("99999999999999999999", str(2**63), "-1", "0"):
        assert 클라.get(f"/api/games/{번호}", headers=머리).status_code == 422, 번호
        assert 클라.post(f"/api/games/{번호}/rounds",
                        headers=머리).status_code == 422, 번호
        assert 클라.post(f"/api/games/{번호}/act", json={"seq": 0, "action": 0},
                        headers=머리).status_code == 422, 번호
        assert 클라.post(f"/api/games/{번호}/finish",
                        headers=머리).status_code == 422, 번호


def test_행동에_true나_문자열을_보내면_422다(환경):
    클라, _ = 환경
    머리 = 로그인(클라)
    게임 = 대기라운드(클라, 머리, 새게임(클라, 머리))
    # 왜 StrictInt인가: 파이썬에서 True == 1 이라 bool이 행동 1(히트)로 통과했다.
    #   "0"과 0.0도 조용히 정수로 바뀌어 기록이 클라이언트 표기에 휘둘렸다.
    for 몸체 in ({"seq": 0, "action": True}, {"seq": 0, "action": "0"},
                {"seq": 0, "action": 0.0}, {"seq": True, "action": 0},
                {"seq": "0", "action": 0}):
        r = 클라.post(f"/api/games/{게임['game_id']}/act", json=몸체, headers=머리)
        assert r.status_code == 422, (몸체, r.status_code, r.text)


def test_상대_모델_번호도_범위를_본다(환경):
    클라, _ = 환경
    머리 = 로그인(클라)
    for 번호 in (0, -1, 2**63):
        r = 클라.post("/api/games", json={"opponent_model_id": 번호}, headers=머리)
        assert r.status_code == 422, (번호, r.status_code, r.text)


def test_규칙이_바뀐_게임은_조회_행동_딜이_409다(환경, monkeypatch):
    클라, _ = 환경
    머리 = 로그인(클라)
    끝난게임 = 끝까지두기(클라, 머리, 새게임(클라, 머리))
    번호 = 대기라운드(클라, 머리, 새게임(클라, 머리))["game_id"]
    # 왜 이렇게 흉내 내는가: 규칙을 바꾼 배포에서 옛 게임을 재생하면 딜러 카드와
    #   정산이 저장된 값과 달라진다. 지문 비교 대상만 바꾸면 그 배포와 같다.
    monkeypatch.setattr("game.rounds.RULES_V1", RULES_H17)
    부름들 = [
        lambda: 클라.get(f"/api/games/{번호}", headers=머리),
        lambda: 클라.post(f"/api/games/{번호}/act", json={"seq": 0, "action": 0},
                         headers=머리),
        lambda: 클라.post(f"/api/games/{번호}/rounds", headers=머리),
    ]
    for 부름 in 부름들:
        r = 부름()
        assert r.status_code == 409, r.text
        assert 상세(r)["code"] == "rules_changed"
    # 왜 끝내기는 허용하는가: 끝내기는 라운드를 재생하지 않는다. 막으면 옛 게임에
    #   갇혀 새 게임도 못 만든다.
    assert 클라.post(f"/api/games/{끝난게임['game_id']}/finish",
                    headers=머리).status_code == 204


def test_상대_모델을_고를_수_있다(환경):
    클라, _ = 환경
    머리 = 로그인(클라)
    고른것 = 클라.get("/api/models").json()[-1]
    게임 = 새게임(클라, 머리, opponent_model_id=고른것["id"])
    assert 게임["opponent"]["name"] == 고른것["name"]


def test_없는_모델을_고르면_404다(환경):
    클라, _ = 환경
    r = 클라.post("/api/games", json={"opponent_model_id": 999999},
                 headers=로그인(클라))
    assert r.status_code == 404


def test_게임은_시작_시점의_상대_모델_해시를_기록한다(환경):
    클라, 공장 = 환경
    번호 = 새게임(클라, 로그인(클라))["game_id"]     # 안 고르면 기본 상대가 붙는다
    with 공장() as s:
        g = s.get(Game, 번호)
        assert g.opponent_model_id is not None
        # 왜 기본 상대도 보는가: 칸이 NULL 허용이라 DB가 빠뜨림을 못 막는다.
        assert g.opponent_artifact_sha256 == s.get(
            ModelRegistry, g.opponent_model_id).artifact_sha256
        assert len(g.opponent_artifact_sha256) == 64


def test_서빙을_끈_모델로는_게임을_만들_수_없다(환경):
    클라, 공장 = 환경
    머리 = 로그인(클라)
    끈것 = 클라.get("/api/models").json()[-1]["id"]
    with 공장() as s:
        s.get(ModelRegistry, 끈것).is_serving = False
        s.commit()
    # 왜: 레지스트리 행은 그대로 남는다. 행으로 판정하면 서빙을 끈 모델로 게임이
    #   시작돼 상대 정책이 무엇이었는지 아무도 모르게 된다.
    r = 클라.post("/api/games", json={"opponent_model_id": 끈것}, headers=머리)
    assert r.status_code == 404, r.text
    assert 끈것 not in [m["id"] for m in 클라.get("/api/models").json()]


def test_스플릿한_손의_EV_손실은_실제_깊이로_계산된다(환경, monkeypatch):
    클라, 공장 = 환경
    머리 = 로그인(클라)
    monkeypatch.setattr("game.rounds.new_seed", lambda: 깊이시드)
    번호 = 새게임(클라, 머리)["game_id"]
    for 순번, 행동 in enumerate((3, 3, 0)):
        r = 클라.post(f"/api/games/{번호}/act",
                     json={"seq": 순번, "action": 행동}, headers=머리)
        assert r.status_code == 200, r.text
    r = 클라.post(f"/api/games/{번호}/act", json={"seq": 3, "action": 0}, headers=머리)
    assert r.status_code == 200, r.text
    # 왜 이 값인가: 이 손의 실제 깊이는 2다. 깊이 0 표를 보면 0.3811이 나온다.
    assert r.json()["feedback"]["dp_ev_loss"] == pytest.approx(0.3485, abs=1e-3)
    with 공장() as s:
        d = s.scalar(select(Decision).where(Decision.seq == 3))
        assert d.dp_ev_loss == pytest.approx(0.3485, abs=1e-3)
        # 왜 칸은 0인가: decisions의 6필드는 상태 키 그대로다(데이터셋 정의).
        assert d.split_depth == 0


def test_깊이를_주면_그_깊이의_Q로_손실을_잰다():
    합법 = legal_actions(깊이키)
    assert ev_loss(깊이키, STAND) == pytest.approx(0.3811, abs=1e-3)
    assert ev_loss(깊이키, STAND, depth=0) == pytest.approx(0.3811, abs=1e-3)
    assert ev_loss(깊이키, STAND, depth=2) == pytest.approx(0.3485, abs=1e-3)
    assert 합법[optimal_action(깊이키, 합법, depth=2)]


def test_깊이가_범위를_벗어나면_거부한다():
    합법 = legal_actions(깊이키)
    # 왜 예외인가: DP 표의 깊이 축은 0..3뿐이다. 조용히 자르면 엉뚱한 칸의 Q로
    #   손실을 재고도 아무도 모른다.
    for 깊이 in (-1, 4, 99):
        with pytest.raises(ValueError):
            ev_loss(깊이키, STAND, depth=깊이)
        with pytest.raises(ValueError):
            optimal_action(깊이키, 합법, depth=깊이)
