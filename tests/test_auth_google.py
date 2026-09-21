"""이 파일은 구글 ID 토큰 검증과 계정 합치기를 확인한다.
입력: 테스트용 RSA 키로 서명한 가짜 구글 토큰.
출력: 200·201·401·404·429 응답과 users 표 상태에 대한 pytest 결과.
"""

import sys
import time
from pathlib import Path

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import game.google_auth as 구글  # noqa: E402
from game.config import get_settings  # noqa: E402
from game.models import User  # noqa: E402
from test_game_api import 환경  # noqa: E402,F401

클라이언트ID = "1234-test.apps.googleusercontent.com"
개인키 = rsa.generate_private_key(public_exponent=65537, key_size=2048)
공개키PEM = 개인키.public_key().public_bytes(
    serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)


class 가짜키집:
    """PyJWKClient 흉내. 어떤 토큰이든 테스트 공개키를 준다."""

    class 키:
        key = 공개키PEM

    def get_signing_key_from_jwt(self, token):
        return self.키()


@pytest.fixture
def 구글환경(환경, monkeypatch):
    monkeypatch.setenv("BJ_GOOGLE_CLIENT_ID", 클라이언트ID)
    get_settings.cache_clear()
    monkeypatch.setattr(구글, "_키집", lambda: 가짜키집())
    yield 환경
    get_settings.cache_clear()


def 토큰(sub="g-1", email="g@b.com", verified=True, name="구글지우", **덮어씀):
    지금 = int(time.time())
    본문 = {"iss": "https://accounts.google.com", "aud": 클라이언트ID, "sub": sub,
          "email": email, "email_verified": verified, "name": name,
          "iat": 지금, "exp": 지금 + 300, **덮어씀}
    return jwt.encode(본문, 개인키, algorithm="RS256", headers={"kid": "test"})


def test_설정이_없으면_구글_로그인은_404다(환경):
    클라, _ = 환경
    assert 클라.get("/api/auth/config").json() == {"google_client_id": None}
    assert 클라.post("/api/auth/google", json={"credential": "x"}).status_code == 404


def test_처음_온_구글_계정은_만들어진다(구글환경):
    클라, 공장 = 구글환경
    assert 클라.get("/api/auth/config").json() == {"google_client_id": 클라이언트ID}
    r = 클라.post("/api/auth/google", json={"credential": 토큰()})
    assert r.status_code == 201, r.text
    assert set(r.json()) == {"access_token", "refresh_token", "token_type"}
    with 공장() as s:
        u = s.scalar(select(User))
        assert (u.email, u.auth_provider, u.google_sub, u.password_hash,
                u.display_name) == ("g@b.com", "google", "g-1", None, "구글지우")
    assert 클라.post("/api/auth/google", json={"credential": 토큰()}).status_code == 200


def test_같은_이메일의_자체_계정에_구글을_붙인다(구글환경):
    클라, 공장 = 구글환경
    assert 클라.post("/api/auth/signup", json={
        "email": "G@b.com", "password": "hunter2!!", "display_name": "지우"}).status_code == 201
    r = 클라.post("/api/auth/google", json={"credential": 토큰(email="g@b.com")})
    assert r.status_code == 200, r.text
    with 공장() as s:
        u = s.scalar(select(User))
        assert (u.auth_provider, u.google_sub, u.display_name) == ("local", "g-1", "지우")
        assert u.password_hash is not None
    # 왜: 합쳐진 계정은 비밀번호 로그인도 여전히 된다.
    assert 클라.post("/api/auth/login", json={
        "email": "g@b.com", "password": "hunter2!!"}).status_code == 200


# 왜 호출식이 아니라 인자 사전인가: parametrize 값은 수집 시각에 만들어진다. 토큰을 그때
#   만들면 exp가 수집 시각 +300초로 굳어, 10분짜리 전체 회귀 뒤쪽에서는 전부 만료돼
#   "만료" 사유로만 401이 나 다른 사유를 검사하지 못한다.
@pytest.mark.parametrize("덮어씀", [
    {"verified": False},
    {"aud": "other-client"},
    {"iss": "https://evil.example"},
    {"exp": "past"},
    {"raw": "not-a-jwt"},
])
def test_검증에_실패한_토큰은_401이다(구글환경, 덮어씀):
    클라, 공장 = 구글환경
    if "raw" in 덮어씀:
        나쁜토큰 = 덮어씀["raw"]
    elif 덮어씀.get("exp") == "past":
        나쁜토큰 = 토큰(exp=int(time.time()) - 10)
    else:
        나쁜토큰 = 토큰(**덮어씀)
    r = 클라.post("/api/auth/google", json={"credential": 나쁜토큰})
    assert r.status_code == 401, r.text
    with 공장() as s:
        assert s.scalar(select(User)) is None


def test_이름이_비정상이면_이메일_앞부분을_쓴다(구글환경):
    클라, 공장 = 구글환경
    # 왜 \N 표기인가: U+200B는 눈에 보이지 않아 복사·붙여넣기에서 사라진다. 이름으로 적는다.
    r = 클라.post("/api/auth/google", json={"credential": 토큰(name="\N{ZERO WIDTH SPACE}")})
    assert r.status_code == 201, r.text
    with 공장() as s:
        assert s.scalar(select(User)).display_name == "g"


def test_구글_로그인도_IP_제한을_받는다(구글환경):
    클라, _ = 구글환경
    from game.deps import login_limiter
    for _ in range(10):
        login_limiter.allow("testclient")
    assert 클라.post("/api/auth/google", json={"credential": 토큰()}).status_code == 429
