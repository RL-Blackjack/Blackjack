"""이 파일은 겹친 요청에서도 결정과 라운드가 한 번만 기록되는지 확인한다.
입력: 배리어로 결정적으로 겹친 요청과, 실제 uvicorn에 퍼붓는 병렬 HTTP.
출력: 상태코드 조합·DB 일관성에 대한 pytest 결과.
"""

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from sqlalchemy import func, select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import game.api.games as 게임모듈  # noqa: E402
from blackjack_rl.state import STAND  # noqa: E402
from game.db import (  # noqa: E402
    create_schema, make_engine, make_session_factory)
from game.engine import Finished, Pending, new_seed, play  # noqa: E402
from game.models import Decision, Game, Round  # noqa: E402
from scripts.register_models import sync_registry  # noqa: E402
# 왜 테스트 파일에서 가져오는가: 픽스처와 도우미를 파일마다 베끼면 세 벌이 따로
#   낡는다. conftest는 이 태스크의 소유 파일이 아니라 한 곳에 두고 나눠 쓴다.
from test_game_api import 로그인, 새게임, 환경  # noqa: E402,F401

비밀 = "test-secret-at-least-32-characters-long"


def 시드찾기(조건):
    """조건(시드, 첫 보기)이 참인 시드가 나올 때까지 새로 뽑는다."""
    # 왜 뽑아서 거르는가: 시드는 서버가 만든다. 원하는 상황을 결정적으로 만들려면
    #   그 상황이 나오는 시드를 미리 찾아 new_seed를 그것으로 바꿔치기해야 한다.
    while True:
        시드 = new_seed()
        if 조건(시드, play(시드, [])):
            return 시드


def 대기시드():
    """딜하자마자 끝나지 않는(결정이 남는) 라운드의 시드."""
    return 시드찾기(lambda _s, 보기: isinstance(보기, Pending))


def 즉시끝시드():
    """결정 없이 바로 끝나는(내추럴) 라운드의 시드. 무승부는 뺀다."""
    # 왜 무승부를 빼는가: 순손익 0인 라운드는 갱신을 덮어써도 합이 같아 티가 안 난다.
    return 시드찾기(lambda _s, v: isinstance(v, Finished) and abs(v.net) > 1e-9)


def 스탠드로끝나는시드():
    """스탠드 한 번으로 끝나고 무승부도 아닌 라운드의 시드."""
    def 조건(시드, 보기):
        끝 = play(시드, [STAND]) if isinstance(보기, Pending) else None
        return isinstance(끝, Finished) and abs(끝.net) > 1e-9
    return 시드찾기(조건)


def 동시에(부름들):
    """여러 요청을 스레드로 겹쳐 보내고 결과를 순서대로 돌려준다."""
    결과 = [None] * len(부름들)

    def 실행(i, 부름):
        try:
            결과[i] = 부름()
        except Exception as 오류:      # noqa: BLE001
            결과[i] = f"EXC {type(오류).__name__}: {오류}"

    실들 = [threading.Thread(target=실행, args=(i, f)) for i, f in enumerate(부름들)]
    for t in 실들:
        t.start()
    for t in 실들:
        t.join(timeout=60)
    return 결과


def 열린수(공장):
    """DB에 열려 있는 라운드 수."""
    with 공장() as s:
        return s.scalar(select(func.count()).select_from(Round)
                        .where(Round.is_open.is_(True)))


def test_같은_결정을_동시에_보내면_한쪽만_성공한다(환경, monkeypatch):
    클라, 공장 = 환경
    머리 = 로그인(클라)
    monkeypatch.setattr("game.rounds.new_seed", 대기시드)
    번호 = 새게임(클라, 머리)["game_id"]
    관문, 원래 = threading.Barrier(2, timeout=30), 게임모듈.optimal_action

    def 겹치기(key, legal, *, depth=None):
        관문.wait()
        return 원래(key, legal, depth=depth)

    monkeypatch.setattr(게임모듈, "optimal_action", 겹치기)
    응답들 = 동시에([
        lambda a=a: 클라.post(f"/api/games/{번호}/act",
                             json={"seq": 0, "action": a}, headers=머리)
        for a in (STAND, 1)])
    코드 = [r.status_code for r in 응답들]
    assert sorted(코드) == [200, 409], [getattr(r, "text", r) for r in 응답들]
    이긴행동 = [a for a, r in zip((STAND, 1), 응답들) if r.status_code == 200][0]
    with 공장() as s:
        결정들 = list(s.scalars(select(Decision)))
        assert len(결정들) == 1
        assert 결정들[0].action_taken == 이긴행동
        라운드 = s.scalar(select(Round))
        assert [int(a) for a in 라운드.actions.split(",") if a] == [이긴행동]
        기대 = play(라운드.seed, [이긴행동])
        assert 라운드.is_open is isinstance(기대, Pending)
        if isinstance(기대, Finished):
            assert 라운드.net == pytest.approx(기대.net)
    보기 = 클라.get(f"/api/games/{번호}", headers=머리).json()["round"]
    assert 보기["status"] == ("pending" if isinstance(기대, Pending) else "finished")


def test_새_라운드를_동시에_열면_한쪽만_성공한다(환경, monkeypatch):
    클라, 공장 = 환경
    머리 = 로그인(클라)
    monkeypatch.setattr("game.rounds.new_seed", 즉시끝시드)
    번호 = 새게임(클라, 머리)["game_id"]     # 첫 라운드는 내추럴이라 바로 닫힌다
    관문, 잠금 = threading.Barrier(2, timeout=30), threading.Lock()
    시드들 = [대기시드(), 대기시드()]

    def 겹치는시드():
        # 왜 여기서 겹치는가: 열림 확인과 회차 읽기가 모두 끝난 지점이다. 확인과
        #   INSERT 사이가 가장 넓게 벌어져 두 라운드가 함께 열릴 수 있다.
        관문.wait()
        with 잠금:
            return 시드들.pop()

    monkeypatch.setattr("game.rounds.new_seed", 겹치는시드)
    응답들 = 동시에([lambda: 클라.post(f"/api/games/{번호}/rounds", headers=머리)] * 2)
    assert sorted(r.status_code for r in 응답들) == [201, 409], [r.text for r in 응답들]
    with 공장() as s:
        라운드들 = list(s.scalars(select(Round).where(Round.game_id == 번호)))
        회차들 = [r.round_no for r in 라운드들]
        assert len(회차들) == len(set(회차들))
        assert sum(r.is_open for r in 라운드들) == 1


def test_열림_확인_뒤_다른_딜이_끼어들어도_열린_라운드는_하나다(환경, monkeypatch):
    클라, 공장 = 환경
    머리 = 로그인(클라)
    monkeypatch.setattr("game.rounds.new_seed", 즉시끝시드)
    첫째 = 새게임(클라, 머리)["game_id"]
    둘째 = 새게임(클라, 머리)["game_id"]
    멈춤, 재개, 호출 = threading.Event(), threading.Event(), {"수": 0}
    시드들 = [대기시드(), 대기시드()]

    def 느린시드():
        호출["수"] += 1
        if 호출["수"] == 1:
            멈춤.set()
            재개.wait(30)
        return 시드들.pop()

    monkeypatch.setattr("game.rounds.new_seed", 느린시드)
    결과 = {}
    실 = threading.Thread(target=lambda: 결과.update(
        늦은쪽=클라.post(f"/api/games/{첫째}/rounds", headers=머리)))
    실.start()
    assert 멈춤.wait(30)
    # 왜: 늦은 쪽이 "열린 라운드 없음"을 읽은 뒤에 다른 게임의 딜이 커밋된다.
    #   확인만으로는 못 막는 자리이고, 여기서 DB의 부분 유일 인덱스가 판정한다.
    빠른쪽 = 클라.post(f"/api/games/{둘째}/rounds", headers=머리)
    재개.set()
    실.join(timeout=60)
    assert 빠른쪽.status_code == 201, 빠른쪽.text
    assert 결과["늦은쪽"].status_code == 409, 결과["늦은쪽"].text
    assert 결과["늦은쪽"].json()["detail"]["code"] == "open_round"
    assert 열린수(공장) == 1


def test_게임_손익_갱신이_사라지지_않는다(환경, monkeypatch):
    클라, 공장 = 환경
    머리 = 로그인(클라)
    시드큐 = [스탠드로끝나는시드(), 즉시끝시드()]
    monkeypatch.setattr("game.rounds.new_seed", lambda: 시드큐.pop(0))
    번호 = 새게임(클라, 머리)["game_id"]
    멈춤, 재개, 호출 = threading.Event(), threading.Event(), {"수": 0}
    원래찾기 = 게임모듈.게임찾기

    def 느린찾기(session, game_id, user):
        게임 = 원래찾기(session, game_id, user)
        호출["수"] += 1
        if 호출["수"] == 1:
            멈춤.set()
            재개.wait(30)
        return 게임

    monkeypatch.setattr(게임모듈, "게임찾기", 느린찾기)
    결과 = {}
    실 = threading.Thread(target=lambda: 결과.update(
        딜=클라.post(f"/api/games/{번호}/rounds", headers=머리)))
    실.start()
    assert 멈춤.wait(30)
    # 왜: 딜이 Game을 읽은 뒤에 행동이 라운드를 닫고 순손익을 올린다. 파이썬에서
    #   읽어 더하면 딜이 그 증가분을 덮어쓴다(실측 3/3 손실).
    행동 = 클라.post(f"/api/games/{번호}/act", json={"seq": 0, "action": STAND},
                  headers=머리)
    재개.set()
    실.join(timeout=60)
    assert 행동.status_code == 200, 행동.text
    assert 결과["딜"].status_code == 201, 결과["딜"].text
    with 공장() as s:
        게임 = s.get(Game, 번호)
        라운드들 = list(s.scalars(select(Round).where(Round.game_id == 번호)))
        assert len(라운드들) == 2
        assert 게임.net_result == pytest.approx(sum(r.net for r in 라운드들))
        # 왜 이 단언까지 두는가: 덮어쓰기가 일어나면 합이 나중 라운드의 net과 같다.
        assert abs(게임.net_result - 라운드들[1].net) > 1e-9


def 빈포트():
    """지금 비어 있는 TCP 포트 하나."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.slow
def test_실제_서버에_병렬_요청을_퍼부어도_결정이_중복되지_않는다(tmp_path):
    """진짜 uvicorn에 판마다 동시 8개 /act 를 보내 200이 하나뿐인지 본다."""
    import httpx
    주소 = f"sqlite:///{(tmp_path / 'srv.db').as_posix()}"
    엔진 = make_engine(주소)
    create_schema(엔진)
    공장 = make_session_factory(엔진)
    with 공장() as s:
        sync_registry(s, ROOT / "models")
    환경변수 = dict(os.environ)
    환경변수.update({"BJ_DATABASE_URL": 주소, "BJ_JWT_SECRET": 비밀,
                   "PYTHONPATH": f"{ROOT}{os.pathsep}{ROOT / 'src'}",
                   "PYTHONIOENCODING": "utf-8"})
    포트 = 빈포트()
    기지 = f"http://127.0.0.1:{포트}"
    로그 = (tmp_path / "uvicorn.log").open("w", encoding="utf-8")
    서버 = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "game.main:app", "--host", "127.0.0.1",
         "--port", str(포트), "--log-level", "warning"],
        env=환경변수, cwd=str(ROOT), stdout=로그, stderr=subprocess.STDOUT)
    try:
        for _ in range(150):
            try:
                if httpx.get(기지 + "/api/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        else:
            pytest.fail("uvicorn이 뜨지 않았다")
        c = httpx.Client(base_url=기지, timeout=60)
        r = c.post("/api/auth/signup", json={
            "email": "p@b.com", "password": "hunter2!!", "display_name": "p"})
        assert r.status_code == 201, r.text
        머리 = {"Authorization": f"Bearer {r.json()['access_token']}"}
        번호 = c.post("/api/games", json={}, headers=머리).json()["game_id"]
        판수 = 20
        for _ in range(판수):
            보기 = c.get(f"/api/games/{번호}", headers=머리).json()["round"]
            while 보기["status"] != "pending":
                보기 = c.post(f"/api/games/{번호}/rounds", headers=머리).json()["round"]
            코드 = 동시에([
                lambda a=a: c.post(f"/api/games/{번호}/act", json={"seq": 0, "action": a},
                                   headers=머리).status_code
                for a in (0, 1, 0, 1, 0, 1, 0, 1)])
            assert 코드.count(200) == 1, 코드
            assert all(x in (200, 409) for x in 코드), 코드
            보기 = c.get(f"/api/games/{번호}", headers=머리).json()["round"]
            while 보기["status"] == "pending":
                보기 = c.post(f"/api/games/{번호}/act", headers=머리,
                            json={"seq": 보기["seq"], "action": 0}).json()["round"]
        c.close()
        with 공장() as s:
            # 왜 seq=0 행만 세는가: 라운드를 닫으려면 결정을 더 둬야 하고 중간에
            #   내추럴 라운드도 끼어든다. 판마다 첫 결정이 정확히 하나면 겹친 8개
            #   요청 중 하나만 기록된 것이다.
            assert s.scalar(select(func.count()).select_from(Decision)
                            .where(Decision.seq == 0)) == 판수
            for 라운드 in s.scalars(select(Round).where(Round.game_id == 번호)):
                행동 = [int(a) for a in 라운드.actions.split(",") if a]
                결정들 = list(s.scalars(select(Decision).order_by(Decision.seq)
                                     .where(Decision.round_id == 라운드.id)))
                assert [d.seq for d in 결정들] == list(range(len(행동)))
                assert [d.action_taken for d in 결정들] == 행동
        assert 열린수(공장) <= 1
    finally:
        서버.terminate()
        try:
            서버.wait(timeout=15)
        except subprocess.TimeoutExpired:
            서버.kill()
        로그.close()
        엔진.dispose()
