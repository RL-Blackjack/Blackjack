"""이 파일은 회원가입·로그인 API가 설계서 §5.2대로 도는지 확인한다.
입력: httpx로 보낸 HTTP 요청.
출력: 상태코드·토큰·중복·제한에 대한 pytest 결과.
"""

import sys
import unicodedata
from datetime import datetime, timedelta, timezone
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

    from game.deps import login_limiter, signup_limiter
    from game.main import create_app
    login_limiter.reset()
    signup_limiter.reset()

    with TestClient(create_app()) as c:
        yield c

    # 왜 끝나고도 비우는가: 제한기는 프로세스 전역이다. 남겨 두면 뒤에 도는 다른
    #   테스트 파일의 가입이 같은 "testclient" 키로 429를 맞는다.
    login_limiter.reset()
    signup_limiter.reset()
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


def test_가입도_분당_5회로_제한된다(클라, monkeypatch):
    from game.api import auth
    해싱 = []
    원래 = auth.hash_password

    def 세는_해싱(평문: str) -> str:
        해싱.append(평문)
        return 원래(평문)

    monkeypatch.setattr(auth, "hash_password", 세는_해싱)
    상태들 = [클라.post("/api/auth/signup", json=가입).status_code for _ in range(6)]
    # 왜 같은 이메일로 보내는가: 409로 끝나는 시도도 argon2 비용은 다 치른다.
    #     그래서 성공이 아니라 시도를 세고, 6번째는 해싱하기 전에 막아야 한다.
    assert 상태들 == [201, 409, 409, 409, 409, 429]
    assert len(해싱) == 5


def test_없는_계정도_비밀번호_검증을_한_번_한다(클라, monkeypatch):
    from sqlalchemy import insert

    from game.api import auth
    from game.db import make_engine
    from game.models import User
    클라.post("/api/auth/signup", json=가입)
    엔진 = make_engine()
    with 엔진.begin() as 연결:  # 구글로만 가입해 password_hash가 NULL인 계정
        연결.execute(insert(User).values(
            email="g@example.com", display_name="구글", auth_provider="google",
            password_hash=None, google_sub="google-1"))
    엔진.dispose()
    검증 = []
    원래 = auth.verify_password

    def 세는_검증(해시: str, 평문: str) -> bool:
        검증.append(해시)
        return 원래(해시, 평문)

    monkeypatch.setattr(auth, "verify_password", 세는_검증)
    for 이메일 in [가입["email"], "nobody@example.com", "g@example.com"]:
        검증.clear()
        r = 클라.post("/api/auth/login", json={"email": 이메일, "password": "틀린비밀번호"})
        assert r.status_code == 401 and r.json()["detail"] == auth.로그인실패
        # 왜: 없는 계정에서 argon2를 건너뛰면 응답이 약 20배 빨라, 메시지를
        #     통일해도 응답 시간만으로 가입 여부가 샌다(실측 37.6ms 대 1.7ms).
        assert len(검증) == 1 and 검증[0].startswith("$argon2id$"), 이메일


def _만료를_옮긴다(시각: datetime) -> None:
    """DB에 있는 리프레시 토큰의 만료 시각을 모두 바꾼다."""
    from sqlalchemy import update

    from game.db import make_engine
    from game.models import RefreshToken
    엔진 = make_engine()
    with 엔진.begin() as 연결:
        연결.execute(update(RefreshToken).values(expires_at=시각))
    엔진.dispose()


# 왜 한국 시간(+09:00)으로 넣는가: 시간대를 무시하고 UTC로 되붙이면 9시간이
#   어긋난다. 3시간 전 만료가 살아 있고 1시간 남은 토큰이 죽는 바로 그 경우다.
KST = timezone(timedelta(hours=9))


def test_만료된_리프레시_토큰은_401(클라):
    발급 = 클라.post("/api/auth/signup", json=가입).json()
    _만료를_옮긴다(datetime.now(KST) - timedelta(hours=3))
    r = 클라.post("/api/auth/refresh", json={"refresh_token": 발급["refresh_token"]})
    assert r.status_code == 401


def test_만료_전_리프레시_토큰은_200(클라):
    발급 = 클라.post("/api/auth/signup", json=가입).json()
    _만료를_옮긴다(datetime.now(KST) + timedelta(hours=1))
    r = 클라.post("/api/auth/refresh", json={"refresh_token": 발급["refresh_token"]})
    assert r.status_code == 200, r.text


def test_공백뿐인_표시명은_거부된다(클라):
    # 왜: 길이 검사가 strip보다 먼저면 "   "이 통과해 빈 이름('')으로 저장된다.
    for 이름 in ["   ", "\t\n", "\u3000", "\u00a0", " " * 64]:
        r = 클라.post("/api/auth/signup", json={**가입, "display_name": 이름})
        assert r.status_code == 422, (repr(이름), r.status_code)


def test_제어문자와_보이지_않는_문자는_거부된다(클라):
    # 왜: 리더보드에 이름이 그대로 나간다. 폭 없는 공백·한글 채움 문자로 남의
    #     이름을 흉내 내거나, 방향 제어(U+202E)·NUL로 화면을 깨뜨릴 수 있다.
    for 이름 in ["지\u200b우", "\u3164", "a\x07b", "\u202eevil", "a\x00b",
                "\ufeff지우", "\u115f", "a\u2028b", "\x1b[31mred"]:
        r = 클라.post("/api/auth/signup", json={**가입, "display_name": 이름})
        assert r.status_code == 422, (repr(이름), r.status_code)


def test_표시명은_NFC로_저장하고_길이는_정규화_뒤에_잰다(클라):
    분해형 = "  " + unicodedata.normalize("NFD", "지우") + "  "
    r = 클라.post("/api/auth/signup", json={**가입, "display_name": 분해형})
    assert r.status_code == 201, r.text
    me = 클라.get("/api/auth/me",
                 headers={"Authorization": f"Bearer {r.json()['access_token']}"})
    assert me.json()["display_name"] == "지우"
    # 왜: U+FB2C는 NFC로 바꾸면 세 글자가 된다. 22자로 들어와 66자로 늘어나므로
    #     정규화 뒤에 다시 재지 않으면 64자 칸을 넘는 이름이 저장된다.
    긴것 = {**가입, "email": "long@example.com", "display_name": "\ufb2c" * 22}
    assert 클라.post("/api/auth/signup", json=긴것).status_code == 422
