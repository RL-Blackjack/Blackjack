"""이 파일은 여러 라우터가 함께 쓰는 FastAPI 의존성을 모은다.
입력: Authorization 헤더와 DB 세션.
출력: 인증된 User, 그리고 로그인 요청 제한기.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from game.db import get_session
from game.models import User
from game.security import RateLimiter, TokenInvalid, read_access_token

# 왜 모듈 전역인가: 요청 제한은 프로세스 하나가 기억해야 의미가 있다. 요청마다
#   새로 만들면 아무도 제한되지 않는다. 테스트는 reset()으로 비운다.
# 왜 설정에서 읽지 않고 상수를 두는가: 모듈 import 시점에 get_settings()를 부르면
#   BJ_JWT_SECRET이 아직 없는 테스트 수집 단계에서 터진다. 설계서 §5.2가 못박은
#   값이므로 상수로 두고, 바꿀 일이 생기면 config.login_rate_per_minute와 함께 고친다.
LOGIN_RATE_PER_MINUTE: int = 10
login_limiter = RateLimiter(limit=LOGIN_RATE_PER_MINUTE, window_seconds=60.0)


def current_user(
    session: Annotated[Session, Depends(get_session)],
    authorization: Annotated[str | None, Header()] = None,
) -> User:
    """Authorization 헤더의 액세스 토큰으로 사용자를 찾는다. 실패하면 401."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "로그인이 필요하다")
    try:
        사용자번호 = read_access_token(authorization.split(" ", 1)[1].strip())
    except TokenInvalid as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "로그인이 필요하다") from e

    사용자 = session.get(User, 사용자번호)
    if 사용자 is None:
        # 왜: 토큰은 멀쩡한데 계정이 지워졌을 수 있다. 그때도 401이다.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "로그인이 필요하다")
    return 사용자
