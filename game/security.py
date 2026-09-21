"""이 파일은 비밀번호 해싱, JWT 발급·검증, 로그인·가입 요청 제한을 맡는다.
입력: 평문 비밀번호, 사용자 번호, 토큰 문자열, 호출자 IP.
출력: argon2 해시, JWT 문자열, 사용자 번호, 허용 여부.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from game.config import get_settings

ALGORITHM: str = "HS256"
REFRESH_BYTES: int = 32

# 왜 모듈 수준에서 한 번만 만드는가: PasswordHasher는 상태가 없고 만드는 비용이
#   있다. 실측 해싱 1회가 29ms이므로 객체 생성까지 요청마다 할 이유가 없다.
_해셔 = PasswordHasher()

# 왜 4개인가: 해시 하나가 64MiB다. 워커 스레드 40개가 전부 argon2에 들어가면
#   2.5GiB인데(검증 I4 실측 2.7GiB), 4개면 256MiB로 묶인다. 사람 로그인은 초당
#   4 × (1000/29) ≈ 130건까지 받으므로 발표 규모에 충분하다.
ARGON2_CONCURRENCY: int = 4
# 왜 2초인가: 자리가 2초 안에 안 나면 폭주 중이다. 기다리게 두면 스레드 풀이 말라
#   health까지 멈춘다. 빨리 503을 주고 Retry-After로 물러나게 한다.
ARGON2_WAIT_SECONDS: float = 2.0
_해싱자리 = threading.BoundedSemaphore(ARGON2_CONCURRENCY)


class TokenInvalid(Exception):
    """토큰이 만료됐거나 서명이 맞지 않거나 형식이 틀렸다."""


class TooBusy(Exception):
    """argon2 자리가 다 차서 기다리다 포기했다. main.py가 503으로 바꾼다."""


def _자리잡고(작업):
    """세마포어 자리를 얻어 작업을 돌린다. 못 얻으면 TooBusy."""
    # 왜 모듈 전역을 호출 시점에 읽는가: 테스트가 _해싱자리·ARGON2_WAIT_SECONDS를
    #   바꿔 끼운다. 기본 인자로 굳히면 바꿔 끼운 값이 안 보인다.
    if not _해싱자리.acquire(timeout=ARGON2_WAIT_SECONDS):
        raise TooBusy
    try:
        return 작업()
    finally:
        _해싱자리.release()


def hash_password(plain: str) -> str:
    """평문 비밀번호를 argon2id 해시로 바꾼다. 같은 값도 매번 다르게 나온다."""
    return _자리잡고(lambda: _해셔.hash(plain))


def verify_password(hashed: str, plain: str) -> bool:
    """해시와 평문이 맞는지 본다. 어떤 실패든 False다(예외를 밖으로 내지 않는다)."""
    def 확인() -> bool:
        try:
            return _해셔.verify(hashed, plain)
        except (VerifyMismatchError, VerificationError, InvalidHashError, UnicodeError):
            # 왜 전부 False인가: 예외가 밖으로 나가면 500이 되고, 틀린 비밀번호(401)와
            #   구분되는 그 차이만으로 계정이 있는지 없는지가 샌다.
            # 왜 UnicodeError도 잡는가: argon2-cffi 25.1.0의 verify()는 형식 검사보다
            #   먼저 hash 문자열을 ascii로 무조건 인코딩한다(_password_hasher.py:244).
            #   비ASCII 문자가 든 가짜 해시는 InvalidHashError가 아니라
            #   UnicodeEncodeError로 먼저 죽는다. 라이브러리 docstring이 약속한
            #   예외 목록에는 없지만 실측으로 확인했다.
            return False
    # 왜 TooBusy는 여기서 안 잡는가: 자리가 없는 것은 비밀번호 문제가 아니라 서버
    #   사정이다. False로 바꾸면 폭주 중에 모든 로그인이 "틀린 비밀번호"가 된다.
    return _자리잡고(확인)


def normalize_email(raw: str) -> str:
    """이메일을 소문자로 정규화한다. citext를 쓰지 않기로 했으므로 코드가 책임진다."""
    return raw.strip().lower()


def make_access_token(user_id: int, version: int = 0, *,
                      now: datetime | None = None) -> str:
    """사용자 번호와 토큰 버전을 담은 액세스 토큰을 만든다. 기본 유효기간은 15분이다."""
    설정 = get_settings()
    기준 = now if now is not None else datetime.now(timezone.utc)
    페이로드 = {
        "sub": str(user_id),
        "ver": int(version),
        "iat": int(기준.timestamp()),
        "exp": int((기준 + timedelta(minutes=설정.access_token_minutes)).timestamp()),
    }
    return jwt.encode(페이로드, 설정.jwt_secret, algorithm=ALGORITHM)


def read_access_token(token: str) -> tuple[int, int]:
    """토큰을 검증하고 (사용자 번호, 토큰 버전)을 꺼낸다. 문제가 있으면 TokenInvalid."""
    try:
        # 왜 세 클레임을 요구하는가: PyJWT는 exp가 없으면 만료 검사를 건너뛴다.
        #   지금은 서버만 서명하지만, 빠진 토큰을 받아 주면 영원히 유효해진다.
        페이로드 = jwt.decode(token, get_settings().jwt_secret, algorithms=[ALGORITHM],
                          options={"require": ["exp", "sub", "iat"]})
        # 왜 ver를 require에 안 넣는가: 서명은 서버만 한다. ver 없는 토큰은 버전 0으로
        #   읽으면 되고, 그래야 손으로 만든 페이로드를 쓰는 기존 테스트가 그대로 산다.
        return int(페이로드["sub"]), int(페이로드.get("ver", 0))
    except (jwt.InvalidTokenError, KeyError, TypeError, ValueError) as e:
        # 왜 사유를 응답에 담지 않는가: "만료됨"과 "서명 불일치"를 구분해 주면
        #   공격자에게 힌트가 된다. 하나로 합친다.
        raise TokenInvalid("토큰을 쓸 수 없다") from e


def hash_refresh_token(raw: str) -> str:
    """리프레시 토큰 원문의 SHA-256. DB에는 이것만 저장한다."""
    # 왜 argon2가 아닌가: 리프레시 토큰은 32바이트 난수라 사전 공격 대상이 아니다.
    #   느린 해시가 필요 없고, 갱신 요청마다 29ms를 쓸 이유는 더더욱 없다.
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def new_refresh_token() -> tuple[str, str]:
    """새 리프레시 토큰을 만들어 (원문, 해시)로 돌려준다. 원문은 이때만 존재한다."""
    원문 = secrets.token_urlsafe(REFRESH_BYTES)
    return 원문, hash_refresh_token(원문)


class RateLimiter:
    """키(보통 IP)마다 시간 창 안의 요청 수를 센다. 프로세스 안에서만 유효하다."""

    # 왜 프로세스 1개(워커 1개) 전제인가: 기록이 이 객체의 메모리에만 있다.
    #   uvicorn 워커를 N개 띄우면 워커마다 따로 세어 한도가 N배가 된다.
    #   여러 워커가 필요해지면 Redis 같은 공유 저장소로 옮겨야 한다.

    def __init__(self, limit: int, window_seconds: float) -> None:
        self.limit = limit
        self.window = window_seconds
        self._기록: dict[str, deque[float]] = {}
        # 왜 잠금인가: 동기 핸들러는 스레드 풀에서 동시에 돈다. "세고 → 비교하고
        #   → 기록하는" 사이에 다른 스레드가 끼면 한도를 넘겨 허용한다.
        self._잠금 = threading.Lock()
        self._마지막청소 = float("-inf")

    def allow(self, key: str, *, now: float | None = None) -> bool:
        """지금 요청을 허용할지 판단하고, 허용하면 센다."""
        시각 = now if now is not None else time.monotonic()
        with self._잠금:
            self._빈키청소(시각)
            큐 = self._기록.get(key, deque())
            while 큐 and 시각 - 큐[0] >= self.window:
                큐.popleft()
            if len(큐) >= self.limit:
                return False
            큐.append(시각)
            self._기록[key] = 큐
            return True

    def _빈키청소(self, 시각: float) -> None:
        """창이 지나 비어 버린 키를 지운다. 창 하나에 한 번만 훑는다."""
        # 왜 다른 키까지 훑는가: 자기 키만 정리하면 IP를 바꿔 한 번씩만 보내는
        #   경우 키가 영원히 남아 메모리가 계속 는다.
        # 왜 창마다 한 번인가: 요청마다 전체를 훑으면 키 수에 비례해 느려진다.
        #   창 하나에 한 번이면 남는 키는 최근 두 창 안에 온 것뿐이다.
        if 시각 - self._마지막청소 < self.window:
            return
        self._마지막청소 = 시각
        지울것 = [키 for 키, 큐 in self._기록.items()
               if not 큐 or 시각 - 큐[-1] >= self.window]
        for 키 in 지울것:
            del self._기록[키]

    def key_count(self) -> int:
        """지금 기억하고 있는 키 수. 메모리가 새지 않는지 확인할 때 쓴다."""
        with self._잠금:
            return len(self._기록)

    def reset(self) -> None:
        """전부 지운다. 테스트에서만 쓴다."""
        with self._잠금:
            self._기록.clear()
            self._마지막청소 = float("-inf")
