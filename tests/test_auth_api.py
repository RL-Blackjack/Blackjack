"""이 파일은 회원가입·로그인 API가 설계서 §5.2대로 도는지 확인한다.
입력: httpx로 보낸 HTTP 요청.
출력: 상태코드·토큰·중복·제한에 대한 pytest 결과.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game.config import get_settings  # noqa: E402
from game.db import reset_engine  # noqa: E402

가입 = {"email": "Jiwoo@Example.com", "password": "hunter2!!", "display_name": "지우"}


@pytest.fixture
def 클라(tmp_path, monkeypatch):
    monkeypatch.setenv("BJ_DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")
    monkeypatch.setenv("BJ_JWT_SECRET", "test-secret-at-least-32-characters-long")
    get_settings.cache_clear()
    reset_engine()

    from game.deps import login_limiter
    from game.main import create_app
    login_limiter.reset()

    with TestClient(create_app()) as c:
        yield c

    reset_engine()
    get_settings.cache_clear()


def test_가입하면_토큰이_나온다(클라):
    r = 클라.post("/api/auth/signup", json=가입)
    assert r.status_code == 201, r.text
    몸체 = r.json()
    assert 몸체["token_type"] == "bearer"
    assert 몸체["access_token"] and 몸체["refresh_token"]


def test_가입하면_이메일이_소문자로_저장된다(클라):
    클라.post("/api/auth/signup", json=가입)
    토큰 = 클라.post("/api/auth/login",
                   json={"email": "JIWOO@EXAMPLE.COM", "password": "hunter2!!"})
    # 왜: 대소문자가 다른 같은 이메일로 로그인이 돼야 한다. citext 대신 정규화를
    #     쓰기로 했으므로 이게 그 결정을 지키는 검사다.
    assert 토큰.status_code == 200, 토큰.text
    r = 클라.get("/api/auth/me",
                headers={"Authorization": f"Bearer {토큰.json()['access_token']}"})
    assert r.json()["email"] == "jiwoo@example.com"


def test_같은_이메일로_두_번_가입할_수_없다(클라):
    클라.post("/api/auth/signup", json=가입)
    r = 클라.post("/api/auth/signup", json=가입)
    assert r.status_code == 409


def test_짧은_비밀번호는_거부된다(클라):
    r = 클라.post("/api/auth/signup", json={**가입, "password": "1234"})
    assert r.status_code == 422


def test_이메일_형식이_아니면_거부된다(클라):
    r = 클라.post("/api/auth/signup", json={**가입, "email": "이건이메일이아니다"})
    assert r.status_code == 422


def test_비밀번호는_응답에_절대_나오지_않는다(클라):
    r = 클라.post("/api/auth/signup", json=가입)
    assert "hunter2" not in r.text
    토큰 = r.json()["access_token"]
    me = 클라.get("/api/auth/me", headers={"Authorization": f"Bearer {토큰}"})
    assert "password" not in me.text and "hunter2" not in me.text


def test_틀린_비밀번호는_401이고_없는_계정과_메시지가_같다(클라):
    클라.post("/api/auth/signup", json=가입)
    틀림 = 클라.post("/api/auth/login",
                   json={"email": 가입["email"], "password": "틀린비밀번호"})
    없음 = 클라.post("/api/auth/login",
                   json={"email": "nobody@example.com", "password": "hunter2!!"})
    assert 틀림.status_code == 401 and 없음.status_code == 401
    # 왜: 메시지가 다르면 "이 이메일은 가입돼 있다"를 알아낼 수 있다.
    assert 틀림.json()["detail"] == 없음.json()["detail"]


def test_토큰_없이_me를_부르면_401(클라):
    assert 클라.get("/api/auth/me").status_code == 401


def test_엉터리_토큰은_401(클라):
    # 왜 영문인가: HTTP 헤더는 ASCII만 된다. "Bearer 엉터리"를 넣으면 httpx가 요청을
    #     보내기도 전에 UnicodeEncodeError로 죽어 서버의 401을 검사하지 못한다.
    for 나쁜것 in ["Bearer not.a.jwt", "Bearer ", "Basic YTpi", "bearer"]:
        r = 클라.get("/api/auth/me", headers={"Authorization": 나쁜것})
        assert r.status_code == 401, (나쁜것, r.status_code)


def test_리프레시로_새_액세스_토큰을_받는다(클라):
    발급 = 클라.post("/api/auth/signup", json=가입).json()
    r = 클라.post("/api/auth/refresh", json={"refresh_token": 발급["refresh_token"]})
    assert r.status_code == 200, r.text
    새토큰 = r.json()["access_token"]
    assert 클라.get("/api/auth/me",
                   headers={"Authorization": f"Bearer {새토큰}"}).status_code == 200


def test_로그아웃하면_그_리프레시_토큰은_못_쓴다(클라):
    발급 = 클라.post("/api/auth/signup", json=가입).json()
    assert 클라.post("/api/auth/logout",
                    json={"refresh_token": 발급["refresh_token"]}).status_code == 204
    r = 클라.post("/api/auth/refresh", json={"refresh_token": 발급["refresh_token"]})
    assert r.status_code == 401


def test_없는_리프레시_토큰은_401(클라):
    r = 클라.post("/api/auth/refresh", json={"refresh_token": "이런토큰은없다"})
    assert r.status_code == 401


def test_로그인_시도가_분당_10회를_넘으면_429(클라):
    클라.post("/api/auth/signup", json=가입)
    from game.deps import login_limiter
    login_limiter.reset()
    상태들 = [클라.post("/api/auth/login",
                     json={"email": 가입["email"], "password": "틀림"}).status_code
            for _ in range(12)]
    # 왜: 설계서 §5.2가 IP당 분당 10회로 못박았다. 가입 요청은 세지 않는다.
    assert 상태들[:10] == [401] * 10
    assert 상태들[10:] == [429, 429]


def test_제한에_걸려도_성공_로그인이_한도_안에서는_통과한다(클라):
    클라.post("/api/auth/signup", json=가입)
    from game.deps import login_limiter
    login_limiter.reset()
    r = 클라.post("/api/auth/login",
                 json={"email": 가입["email"], "password": "hunter2!!"})
    assert r.status_code == 200


def test_헬스체크가_있다(클라):
    r = 클라.get("/api/health")
    assert r.status_code == 200
    assert r.json()["rules_fp"] == "47c403568aa3"
