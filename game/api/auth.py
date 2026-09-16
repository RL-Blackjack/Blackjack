"""이 파일은 회원가입·로그인·토큰 갱신·로그아웃 엔드포인트를 제공한다.
입력: 이메일·비밀번호·리프레시 토큰.
출력: 액세스/리프레시 토큰과 사용자 공개 정보.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from game.config import get_settings
from game.db import get_session
from game.deps import current_user, login_limiter, signup_limiter
from game.models import RefreshToken, User
from game.schemas import LoginIn, RefreshIn, SignupIn, TokenOut, UserOut
from game.security import (
    hash_password,
    hash_refresh_token,
    make_access_token,
    new_refresh_token,
    normalize_email,
    verify_password,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])

# 왜 메시지를 하나로 고정하는가: 로그인은 상태·본문·응답 시간으로 계정 유무를
#   구분하지 않는다(없는 계정도 더미 해시를 검증한다). "이메일 없음"과
#   "비밀번호 틀림"이 다르면 그 차이만으로 가입된 이메일 목록을 캐낼 수 있다.
로그인실패 = "이메일 또는 비밀번호가 맞지 않는다"

# 왜 더미 해시인가: 없는 계정에서 argon2를 건너뛰면 응답이 약 20배 빨라(실측
#   37.6ms 대 1.7ms) 메시지를 통일해도 시간으로 가입 여부가 샌다. 같은 해셔로
#   만든 해시를 한 번 검증해 두 경로의 비용을 맞춘다. 해셔 설정이 바뀌어도
#   import 때 새로 만들므로 비용이 저절로 따라간다.
_더미해시 = hash_password(secrets.token_urlsafe(32))


def _토큰발급(session: Session, user: User) -> TokenOut:
    """액세스 토큰을 만들고 리프레시 토큰을 DB에 해시로 저장한다."""
    원문, 해시 = new_refresh_token()
    session.add(RefreshToken(
        user_id=user.id,
        token_hash=해시,
        expires_at=datetime.now(timezone.utc)
        + timedelta(days=get_settings().refresh_token_days),
    ))
    session.commit()
    return TokenOut(access_token=make_access_token(user.id), refresh_token=원문)


@router.post("/signup", response_model=TokenOut,
             status_code=status.HTTP_201_CREATED)
def signup(몸체: SignupIn, request: Request,
           session: Annotated[Session, Depends(get_session)]) -> TokenOut:
    """새 계정을 만들고 바로 로그인 상태로 만들어 준다. IP당 분당 5회로 제한한다."""
    아이피 = request.client.host if request.client else "unknown"
    # 왜 해싱보다 먼저, 시도마다 세는가: 409로 끝나는 가입도 argon2 비용은 다
    #   치른다. 성공만 세면 같은 이메일을 반복해 서버를 묶을 수 있다.
    if not signup_limiter.allow(아이피):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            "가입 시도가 너무 잦다. 잠시 뒤 다시 시도하라")

    사용자 = User(
        email=normalize_email(str(몸체.email)),
        display_name=몸체.display_name,  # 검증기가 NFC·strip을 마쳤다
        auth_provider="local",
        password_hash=hash_password(몸체.password),
        created_at=datetime.now(timezone.utc),
    )
    session.add(사용자)
    try:
        session.commit()
    except IntegrityError as e:
        session.rollback()
        # 왜 가입만 중복을 알려 주는가: 사용자가 로그인으로 가야 하기 때문이다.
        #   그래서 계정 존재 여부는 가입 409로 확인할 수 있다. 메일 인증이 범위
        #   밖이라 완전히 숨기지 않고 요청 제한으로 속도만 늦춘다.
        raise HTTPException(status.HTTP_409_CONFLICT, "이미 가입된 이메일이다") from e

    return _토큰발급(session, 사용자)


@router.post("/login", response_model=TokenOut)
def login(몸체: LoginIn, request: Request,
          session: Annotated[Session, Depends(get_session)]) -> TokenOut:
    """이메일과 비밀번호로 로그인한다. IP당 분당 10회로 제한한다."""
    아이피 = request.client.host if request.client else "unknown"
    if not login_limiter.allow(아이피):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            "로그인 시도가 너무 잦다. 잠시 뒤 다시 시도하라")

    사용자 = session.scalar(
        select(User).where(User.email == normalize_email(str(몸체.email))))
    if 사용자 is None or 사용자.password_hash is None:
        # 왜 결과를 버릴 검증을 하는가: 있는 계정과 같은 argon2 비용을 치러
        #   응답 시간을 맞춘다. 구글로만 가입한 계정(해시 없음)도 같다.
        verify_password(_더미해시, 몸체.password)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, 로그인실패)
    if not verify_password(사용자.password_hash, 몸체.password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, 로그인실패)

    return _토큰발급(session, 사용자)


@router.post("/refresh", response_model=TokenOut)
def refresh(몸체: RefreshIn,
            session: Annotated[Session, Depends(get_session)]) -> TokenOut:
    """리프레시 토큰으로 새 액세스 토큰을 받는다."""
    기록 = session.scalar(select(RefreshToken).where(
        RefreshToken.token_hash == hash_refresh_token(몸체.refresh_token)))
    if 기록 is None or 기록.revoked_at is not None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "다시 로그인하라")
    if 기록.expires_at <= datetime.now(timezone.utc):
        # 왜 그대로 비교하는가: expires_at은 UTCDateTime 칸이라 SQLite(naive)든
        #   PG(연결 TimeZone 기준 aware)든 aware UTC로 온다. 예전처럼 tzinfo를
        #   덮어쓰면 Asia/Seoul 연결에서 만료가 9시간 늦어진다.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "다시 로그인하라")

    사용자 = session.get(User, 기록.user_id)
    if 사용자 is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "다시 로그인하라")
    return TokenOut(access_token=make_access_token(사용자.id),
                    refresh_token=몸체.refresh_token)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(몸체: RefreshIn,
           session: Annotated[Session, Depends(get_session)]) -> Response:
    """리프레시 토큰을 폐기한다. 없는 토큰이어도 204다."""
    기록 = session.scalar(select(RefreshToken).where(
        RefreshToken.token_hash == hash_refresh_token(몸체.refresh_token)))
    if 기록 is not None and 기록.revoked_at is None:
        기록.revoked_at = datetime.now(timezone.utc)
        session.commit()
    # 왜 없어도 204인가: "그 토큰은 존재하지 않는다"를 알려 주면 토큰을 찍어
    #   맞히는 데 쓸 수 있다.
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/me", response_model=UserOut)
def me(사용자: Annotated[User, Depends(current_user)]) -> UserOut:
    """지금 로그인한 사용자의 공개 정보."""
    return UserOut(id=사용자.id, email=사용자.email,
                   display_name=사용자.display_name,
                   auth_provider=사용자.auth_provider,
                   created_at=사용자.created_at)
