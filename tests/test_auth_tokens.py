"""이 파일은 리프레시 회전·재사용 감지·토큰 버전·전체 로그아웃·비밀번호 변경을 확인한다.
입력: httpx로 보낸 인증 요청과 손으로 만든 액세스 토큰.
출력: 401·204·200 응답과 refresh_tokens 표 상태에 대한 pytest 결과.
"""

import sys
import threading
from datetime import timedelta
from pathlib import Path

from sqlalchemy import func, select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from game.models import RefreshToken, User  # noqa: E402
from game.security import make_access_token  # noqa: E402
from test_game_api import 환경  # noqa: E402,F401

가입 = {"email": "a@b.com", "password": "hunter2!!", "display_name": "지우"}


def 토큰들(클라):
    r = 클라.post("/api/auth/signup", json=가입)
    assert r.status_code == 201, r.text
    return r.json()


def 머리(액세스):
    return {"Authorization": f"Bearer {액세스}"}


def 갱신(클라, 리프레시):
    return 클라.post("/api/auth/refresh", json={"refresh_token": 리프레시})


def 살아있는수(공장):
    with 공장() as s:
        return s.scalar(select(func.count()).select_from(RefreshToken)
                        .where(RefreshToken.revoked_at.is_(None)))


def test_리프레시는_새_토큰을_주고_옛_토큰을_폐기한다(환경):
    클라, 공장 = 환경
    처음 = 토큰들(클라)
    r = 갱신(클라, 처음["refresh_token"])
    assert r.status_code == 200, r.text
    새것 = r.json()
    assert 새것["refresh_token"] != 처음["refresh_token"]
    assert 살아있는수(공장) == 1
    # 왜: 옛 토큰은 이제 쓸 수 없어야 회전의 의미가 있다.
    assert 갱신(클라, 처음["refresh_token"]).status_code == 401
    # 왜: 회전 직후(유예 안)의 재사용은 다른 탭일 수 있다. 새 토큰까지 죽이면 안 된다.
    assert 갱신(클라, 새것["refresh_token"]).status_code == 200


def test_같은_토큰으로_동시에_갱신하면_한쪽만_성공하고_아무도_쫓겨나지_않는다(환경):
    클라, 공장 = 환경
    t = 토큰들(클라)
    결과 = [None, None]

    def 보내기(i):
        결과[i] = 갱신(클라, t["refresh_token"])

    실들 = [threading.Thread(target=보내기, args=(i,)) for i in range(2)]
    for x in 실들:
        x.start()
    for x in 실들:
        x.join(timeout=30)
    assert sorted(r.status_code for r in 결과) == [200, 401], [r.text for r in 결과]
    새것 = next(r for r in 결과 if r.status_code == 200).json()
    # 왜: 진 쪽은 유예 안이라 도난이 아니다. 이긴 쪽의 새 토큰은 살아 있어야 한다.
    assert 갱신(클라, 새것["refresh_token"]).status_code == 200
    assert 살아있는수(공장) == 1


def test_유예가_지난_뒤의_재사용만_전부_폐기한다(환경):
    클라, 공장 = 환경
    처음 = 토큰들(클라)
    둘째 = 갱신(클라, 처음["refresh_token"]).json()
    with 공장() as s:
        for 행 in s.scalars(select(RefreshToken).where(RefreshToken.revoked_at.is_not(None))):
            행.revoked_at = 행.revoked_at - timedelta(seconds=60)
        s.commit()
    # 왜: 회전한 지 30초가 넘은 옛 토큰이 다시 오는 것은 복사해 간 쪽이다. 둘 다 끊는다.
    assert 갱신(클라, 처음["refresh_token"]).status_code == 401
    assert 살아있는수(공장) == 0
    assert 갱신(클라, 둘째["refresh_token"]).status_code == 401


def test_로그아웃한_토큰을_다시_써도_다른_기기는_살아있다(환경):
    클라, 공장 = 환경
    폰 = 토큰들(클라)
    노트북 = 클라.post("/api/auth/login", json={
        "email": "a@b.com", "password": "hunter2!!"}).json()
    assert 클라.post("/api/auth/logout",
                    json={"refresh_token": 폰["refresh_token"]}).status_code == 204
    with 공장() as s:
        # 왜 시각을 되돌리는가: 로그아웃이 행을 지우지 않고 폐기 표시만 했다면, 유예가
        #   지난 재사용을 도난으로 봐 노트북까지 끊는다. 그 차이를 드러낸다.
        for 행 in s.scalars(select(RefreshToken).where(RefreshToken.revoked_at.is_not(None))):
            행.revoked_at = 행.revoked_at - timedelta(seconds=60)
        s.commit()
    assert 갱신(클라, 폰["refresh_token"]).status_code == 401
    # 왜: 로그아웃한 토큰의 재사용은 정상(다른 탭)이다. 도난으로 보면 안 된다.
    assert 갱신(클라, 노트북["refresh_token"]).status_code == 200


def test_전체_로그아웃은_액세스_토큰까지_죽인다(환경):
    클라, 공장 = 환경
    t = 토큰들(클라)
    assert 클라.get("/api/auth/me", headers=머리(t["access_token"])).status_code == 200
    r = 클라.post("/api/auth/logout-all", headers=머리(t["access_token"]))
    assert r.status_code == 204, r.text
    # 왜: 아직 15분이 안 지난 액세스 토큰도 버전이 어긋나 401이어야 한다.
    assert 클라.get("/api/auth/me", headers=머리(t["access_token"])).status_code == 401
    assert 갱신(클라, t["refresh_token"]).status_code == 401
    assert 살아있는수(공장) == 0


def test_버전이_다른_액세스_토큰은_401이다(환경):
    클라, 공장 = 환경
    t = 토큰들(클라)
    with 공장() as s:
        번호 = s.scalar(select(User.id))
    옛버전 = make_access_token(번호, version=0)
    assert 클라.get("/api/auth/me", headers=머리(옛버전)).status_code == 200
    다른버전 = make_access_token(번호, version=7)
    assert 클라.get("/api/auth/me", headers=머리(다른버전)).status_code == 401


def test_비밀번호를_바꾸면_새_토큰을_받고_옛_것은_전부_죽는다(환경):
    클라, 공장 = 환경
    t = 토큰들(클라)
    r = 클라.post("/api/auth/password", headers=머리(t["access_token"]),
                 json={"current_password": "hunter2!!", "new_password": "newpass99!"})
    assert r.status_code == 200, r.text
    새것 = r.json()
    assert 클라.get("/api/auth/me", headers=머리(t["access_token"])).status_code == 401
    assert 클라.get("/api/auth/me", headers=머리(새것["access_token"])).status_code == 200
    assert 살아있는수(공장) == 1
    assert 클라.post("/api/auth/login", json={
        "email": "a@b.com", "password": "hunter2!!"}).status_code == 401
    assert 클라.post("/api/auth/login", json={
        "email": "a@b.com", "password": "newpass99!"}).status_code == 200


def test_현재_비밀번호가_틀리면_아무것도_바뀌지_않는다(환경):
    클라, 공장 = 환경
    t = 토큰들(클라)
    r = 클라.post("/api/auth/password", headers=머리(t["access_token"]),
                 json={"current_password": "wrong!!!!", "new_password": "newpass99!"})
    assert r.status_code == 401
    assert 클라.get("/api/auth/me", headers=머리(t["access_token"])).status_code == 200
    assert 살아있는수(공장) == 1


def test_짧은_새_비밀번호는_422다(환경):
    클라, _ = 환경
    t = 토큰들(클라)
    r = 클라.post("/api/auth/password", headers=머리(t["access_token"]),
                 json={"current_password": "hunter2!!", "new_password": "short"})
    assert r.status_code == 422
