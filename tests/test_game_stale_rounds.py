"""이 파일은 규칙이 바뀐 뒤 옛 열린 라운드가 사용자를 잠그지 않는지 확인한다.
입력: 규칙 지문을 손으로 바꾼 게임 행과 그 뒤의 API 요청.
출력: 201·204 응답과 옛 라운드·게임의 닫힘 상태에 대한 pytest 결과.
"""

import sys
from pathlib import Path

from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from game.models import Game, Round  # noqa: E402
from test_game_api import 대기라운드, 로그인, 새게임, 환경  # noqa: E402,F401

옛지문 = "000000000000"


def 옛게임만들기(클라, 공장, 머리):
    """열린 라운드가 있는 게임을 만들고 규칙 지문을 옛것으로 바꿔 둔다."""
    게임 = 대기라운드(클라, 머리, 새게임(클라, 머리))
    with 공장() as s:
        s.get(Game, 게임["game_id"]).rules_fp = 옛지문
        s.commit()
    return 게임["game_id"]


def test_옛_열린_라운드가_있어도_새_게임을_만들_수_있다(환경):
    클라, 공장 = 환경
    머리 = 로그인(클라)
    옛번호 = 옛게임만들기(클라, 공장, 머리)
    r = 클라.post("/api/games", json={}, headers=머리)
    assert r.status_code == 201, r.text
    with 공장() as s:
        옛게임 = s.get(Game, 옛번호)
        옛라운드들 = list(s.scalars(select(Round).where(Round.game_id == 옛번호)))
        assert 옛게임.ended_at is not None
        assert not any(x.is_open for x in 옛라운드들)
        # 왜: 정산 없이 버린 라운드다. 손익 0이어야 전적·순위가 흔들리지 않는다.
        assert all(x.net == 0.0 for x in 옛라운드들 if x.dealer_cards == "")


def test_옛_게임은_끝내기로_닫을_수_있다(환경):
    클라, 공장 = 환경
    머리 = 로그인(클라)
    옛번호 = 옛게임만들기(클라, 공장, 머리)
    r = 클라.post(f"/api/games/{옛번호}/finish", headers=머리)
    assert r.status_code == 204, r.text
    with 공장() as s:
        assert s.get(Game, 옛번호).ended_at is not None


def test_지금_규칙의_열린_라운드는_정리하지_않는다(환경):
    클라, 공장 = 환경
    머리 = 로그인(클라)
    게임 = 대기라운드(클라, 머리, 새게임(클라, 머리))
    r = 클라.post("/api/games", json={}, headers=머리)
    assert r.status_code == 409
    assert r.json()["detail"] == {"code": "open_round",
                                  "message": "아직 끝나지 않은 라운드가 있다",
                                  "game_id": 게임["game_id"]}


def test_옛_라운드_정리는_그_사용자_것만_건드린다(환경):
    클라, 공장 = 환경
    나, 남 = 로그인(클라, "me@b.com"), 로그인(클라, "other@b.com")
    남의옛번호 = 옛게임만들기(클라, 공장, 남)
    assert 클라.post("/api/games", json={}, headers=나).status_code == 201
    with 공장() as s:
        assert s.get(Game, 남의옛번호).ended_at is None
