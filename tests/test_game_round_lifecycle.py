"""이 파일은 끝내기와 딜이 겹쳐도 사용자가 잠기지 않는지, 409 코드와 회차가 맞는지 확인한다.
입력: 잠금 자리에서 멈춘 딜과 그 사이에 보낸 끝내기·딜 요청.
출력: 상태코드 조합·409 코드·DB 일관성에 대한 pytest 결과.
"""

import sys
import threading
from pathlib import Path

from sqlalchemy import func, select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import game.rounds as 라운드모듈  # noqa: E402
from game.models import Game, Round  # noqa: E402
# 왜 테스트 파일에서 가져오는가: 픽스처와 도우미를 파일마다 베끼면 낡는다.
#   conftest는 이 태스크의 소유 파일이 아니라 한 곳에 두고 나눠 쓴다.
from test_game_api import 끝까지두기, 로그인, 새게임, 환경  # noqa: E402,F401
from test_game_concurrency import 대기시드, 동시에, 즉시끝시드  # noqa: E402


def 딜(클라, 머리, 번호):
    return 클라.post(f"/api/games/{번호}/rounds", headers=머리)


def 끝내기(클라, 머리, 번호):
    return 클라.post(f"/api/games/{번호}/finish", headers=머리)


def 잠긴게임수(공장):
    """끝났는데 열린 라운드가 남은 게임 수. 하나라도 있으면 그 사용자는 영구히 막힌다."""
    with 공장() as s:
        return s.scalar(
            select(func.count(func.distinct(Round.game_id)))
            .join(Game, Game.id == Round.game_id)
            .where(Round.is_open.is_(True), Game.ended_at.is_not(None)))


def test_딜이_잠근_사이_들어온_끝내기는_기다렸다가_열린_라운드를_본다(환경, monkeypatch):
    클라, 공장 = 환경
    머리 = 로그인(클라)
    monkeypatch.setattr("game.rounds.new_seed", 즉시끝시드)
    번호 = 새게임(클라, 머리)["game_id"]     # 첫 라운드는 내추럴이라 닫혀 있다
    monkeypatch.setattr("game.rounds.new_seed", 대기시드)
    원래, 잠갔다, 재개 = 라운드모듈.게임잠금, threading.Event(), threading.Event()

    def 느린잠금(session, game_id, **값):
        결과 = 원래(session, game_id, **값)
        # 왜 여기서 멈추는가: 딜이 게임 행을 잠갔지만 라운드를 아직 커밋하지 않은
        #   자리다. 끝내기가 이 잠금을 기다리지 않으면 열린 라운드를 못 보고 끝낸다.
        잠갔다.set()
        재개.wait(10)
        return 결과

    monkeypatch.setattr(라운드모듈, "게임잠금", 느린잠금)
    결과 = {}
    딜실 = threading.Thread(target=lambda: 결과.update(딜=딜(클라, 머리, 번호)))
    딜실.start()
    assert 잠갔다.wait(30)
    끝실 = threading.Thread(target=lambda: 결과.update(끝=끝내기(클라, 머리, 번호)))
    끝실.start()
    끝실.join(timeout=1.0)     # 끝내기가 잠금을 기다리는 동안 딜을 붙잡아 둔다
    재개.set()
    딜실.join(timeout=60)
    끝실.join(timeout=60)

    assert 결과["딜"].status_code == 201, 결과["딜"].text
    assert 결과["끝"].status_code == 409, 결과["끝"].text
    상세 = 결과["끝"].json()["detail"]
    assert (상세["code"], 상세["game_id"]) == ("open_round", 번호)
    assert 잠긴게임수(공장) == 0
    with 공장() as s:
        assert s.get(Game, 번호).ended_at is None
    # 왜 이어서 두는가: 잠기지 않았다는 증거는 그 라운드를 끝까지 두고 끝낼 수 있다는 것.
    끝까지두기(클라, 머리, {"game_id": 번호, "round": 결과["딜"].json()["round"]})
    assert 끝내기(클라, 머리, 번호).status_code == 204


def test_끝낸_뒤_늦게_들어온_딜은_끝난_게임이라고_거절된다(환경, monkeypatch):
    클라, 공장 = 환경
    머리 = 로그인(클라)
    monkeypatch.setattr("game.rounds.new_seed", 즉시끝시드)
    번호 = 새게임(클라, 머리)["game_id"]
    멈춤, 재개 = threading.Event(), threading.Event()

    def 느린시드():
        # 왜 여기인가: 딜이 "끝나지 않은 게임"을 확인한 뒤, 잠그기 전이다. 이 사이에
        #   끝내기가 커밋되면 딜은 잠금 UPDATE에서 0행을 보고 물러나야 한다.
        멈춤.set()
        재개.wait(10)
        return 대기시드()

    monkeypatch.setattr("game.rounds.new_seed", 느린시드)
    결과 = {}
    딜실 = threading.Thread(target=lambda: 결과.update(딜=딜(클라, 머리, 번호)))
    딜실.start()
    assert 멈춤.wait(30)
    끝 = 끝내기(클라, 머리, 번호)
    재개.set()
    딜실.join(timeout=60)

    assert 끝.status_code == 204, 끝.text
    assert 결과["딜"].status_code == 409, 결과["딜"].text
    상세 = 결과["딜"].json()["detail"]
    assert (상세["code"], 상세["game_id"]) == ("game_ended", 번호)
    assert 잠긴게임수(공장) == 0
    with 공장() as s:
        assert s.get(Game, 번호).ended_at is not None
    # 왜: 잠기지 않았다는 증거는 새 게임을 만들 수 있다는 것.
    새게임(클라, 머리)


def test_이미_끝난_게임을_다시_끝내면_204다(환경, monkeypatch):
    클라, 공장 = 환경
    머리 = 로그인(클라)
    monkeypatch.setattr("game.rounds.new_seed", 즉시끝시드)
    번호 = 새게임(클라, 머리)["game_id"]
    assert 끝내기(클라, 머리, 번호).status_code == 204
    with 공장() as s:
        처음 = s.get(Game, 번호).ended_at
    assert 끝내기(클라, 머리, 번호).status_code == 204
    with 공장() as s:
        # 왜: 두 번째 끝내기가 시각을 덮어쓰면 전적의 "끝난 때"가 뒤로 밀린다.
        assert s.get(Game, 번호).ended_at == 처음


def test_이긴_딜이_바로_닫히면_진_딜은_conflict_다(환경, monkeypatch):
    클라, 공장 = 환경
    머리 = 로그인(클라)
    monkeypatch.setattr("game.rounds.new_seed", 즉시끝시드)
    번호 = 새게임(클라, 머리)["game_id"]
    관문, 잠금 = threading.Barrier(2, timeout=30), threading.Lock()
    시드들 = [즉시끝시드(), 즉시끝시드()]

    def 겹치는시드():
        # 왜 둘 다 내추럴인가: 이긴 딜의 라운드가 바로 닫히면 진 딜을 막는 것은
        #   사용자 열린 라운드 인덱스가 아니라 (game_id, round_no) 제약이다.
        관문.wait()
        with 잠금:
            return 시드들.pop()

    monkeypatch.setattr("game.rounds.new_seed", 겹치는시드)
    응답들 = 동시에([lambda: 딜(클라, 머리, 번호)] * 2)
    assert sorted(r.status_code for r in 응답들) == [201, 409], [r.text for r in 응답들]
    상세 = next(r for r in 응답들 if r.status_code == 409).json()["detail"]
    # 왜 open_round가 아닌가: 이을 라운드가 없는데 "이어 두기"를 띄우면 화면이 헤맨다.
    assert (상세["code"], 상세["game_id"]) == ("conflict", 번호)
    with 공장() as s:
        라운드들 = list(s.scalars(select(Round).where(Round.game_id == 번호)))
        assert sorted(r.round_no for r in 라운드들) == [1, 2]
        assert not any(r.is_open for r in 라운드들)


def test_가운데_라운드_행이_없어져도_회차는_이어진다(환경, monkeypatch):
    클라, 공장 = 환경
    머리 = 로그인(클라)
    monkeypatch.setattr("game.rounds.new_seed", 즉시끝시드)
    번호 = 새게임(클라, 머리)["game_id"]
    for _ in range(2):
        assert 딜(클라, 머리, 번호).status_code == 201
    with 공장() as s:
        가운데 = s.scalar(select(Round).where(Round.game_id == 번호, Round.round_no == 2))
        s.delete(가운데)
        s.commit()
    # 왜: 회차를 개수로 세면 3이 나와 남은 3회차와 부딪혀 500이 난다. 마지막 행의
    #   회차 + 1 이어야 한다.
    r = 딜(클라, 머리, 번호)
    assert r.status_code == 201, r.text
    assert r.json()["round_no"] == 4
