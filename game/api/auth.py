"""이 파일은 회원가입·로그인·토큰 갱신·로그아웃 엔드포인트를 제공한다.
입력: 이메일·비밀번호·리프레시 토큰.
출력: 액세스/리프레시 토큰과 사용자 공개 정보.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic_core import PydanticCustomError
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from game.config import get_settings
from game.db import get_session
from game.deps import current_user, login_limiter, signup_limiter
from game.google_auth import verify_google_id_token
from game.models import RefreshToken, User
from game.schemas import (
    DISPLAY_NAME_MAX, AuthConfigOut, GoogleIn, LoginIn, PasswordChangeIn, RefreshIn,
    SignupIn, TokenOut, UserOut, 표시명정리)
from game.security import (
    TokenInvalid,
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

# 왜 30초인가: 같은 브라우저의 다른 탭이 옛 리프레시 토큰을 들고 있다가 보내는 데
#   걸리는 시간은 길어야 몇 초다. 30초를 넘겨 옛 토큰이 오면 복사해 간 쪽이라고 본다.
ROTATION_GRACE_SECONDS: int = 30


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
    return TokenOut(access_token=make_access_token(user.id, user.token_version),
                    refresh_token=원문)


def _전부폐기(session: Session, user: User) -> None:
    """이 사용자의 살아 있는 리프레시 토큰을 모두 폐기하고 액세스 토큰 버전을 올린다."""
    session.execute(
        update(RefreshToken)
        .where(RefreshToken.user_id == user.id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=datetime.now(timezone.utc))
        .execution_options(synchronize_session=False))
    # 왜 SQL 식인가: 두 요청이 겹쳐도 +1이 두 번 적용돼야 한다.
    session.execute(
        update(User).where(User.id == user.id)
        .values(token_version=User.token_version + 1)
        .execution_options(synchronize_session=False))
    session.commit()
    session.refresh(user)


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
    """리프레시 토큰을 새것으로 바꾸고 새 액세스 토큰을 준다. 옛 토큰은 폐기한다."""
    기록 = session.scalar(select(RefreshToken).where(
        RefreshToken.token_hash == hash_refresh_token(몸체.refresh_token)))
    if 기록 is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "다시 로그인하라")
    사용자 = session.get(User, 기록.user_id)
    지금 = datetime.now(timezone.utc)
    if 기록.revoked_at is not None:
        # 왜 유예를 두는가: 회전 직후 30초 안의 재사용은 겹친 요청(다른 탭)이다.
        #   그 뒤의 재사용은 누군가 복사해 갔다는 신호라 전부 끊는다.
        # 왜 그대로 비교하는가: revoked_at은 UTCDateTime 칸이라 SQLite든 PG든
        #   aware UTC로 온다. tzinfo를 덮어쓰면 Asia/Seoul 연결에서 9시간 어긋난다.
        if (사용자 is not None
                and (지금 - 기록.revoked_at).total_seconds() > ROTATION_GRACE_SECONDS):
            _전부폐기(session, 사용자)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "다시 로그인하라")
    if 기록.expires_at <= 지금 or 사용자 is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "다시 로그인하라")
    # 왜 조건부 UPDATE인가: 같은 토큰이 동시에 두 번 오면 파이썬 확인은 둘 다 통과한다.
    #   DB가 한 번만 성공시키고, 진 쪽은 위의 유예 규칙으로 401만 받는다.
    회전 = session.execute(
        update(RefreshToken)
        .where(RefreshToken.id == 기록.id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=지금).execution_options(synchronize_session=False))
    if 회전.rowcount != 1:
        session.rollback()
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "다시 로그인하라")
    session.commit()
    return _토큰발급(session, 사용자)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(몸체: RefreshIn,
           session: Annotated[Session, Depends(get_session)]) -> Response:
    """리프레시 토큰을 지운다. 없는 토큰이어도 204다."""
    기록 = session.scalar(select(RefreshToken).where(
        RefreshToken.token_hash == hash_refresh_token(몸체.refresh_token)))
    if 기록 is not None:
        # 왜 폐기 표시가 아니라 삭제인가: 로그아웃한 토큰을 다른 탭이 다시 보내는 것은
        #   정상이다. 행이 없으면 평범한 401로 끝나고, 도난 감지는 회전된 토큰에만 걸린다.
        session.delete(기록)
        session.commit()
    # 왜 없어도 204인가: "그 토큰은 존재하지 않는다"를 알려 주면 토큰을 찍어
    #   맞히는 데 쓸 수 있다.
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/logout-all", status_code=status.HTTP_204_NO_CONTENT)
def logout_all(사용자: Annotated[User, Depends(current_user)],
               session: Annotated[Session, Depends(get_session)]) -> Response:
    """모든 기기에서 로그아웃한다. 리프레시 토큰 전부 폐기 + 액세스 토큰 버전 올림."""
    _전부폐기(session, 사용자)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/password", response_model=TokenOut)
def change_password(몸체: PasswordChangeIn, request: Request,
                    사용자: Annotated[User, Depends(current_user)],
                    session: Annotated[Session, Depends(get_session)]) -> TokenOut:
    """비밀번호를 바꾼다. 옛 토큰은 전부 죽고 새 토큰 한 쌍을 받는다."""
    아이피 = request.client.host if request.client else "unknown"
    # 왜 로그인 제한기를 같이 쓰는가: 현재 비밀번호 확인은 로그인과 같은 공격면이다.
    if not login_limiter.allow(아이피):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            "시도가 너무 잦다. 잠시 뒤 다시 시도하라")
    if 사용자.password_hash is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "구글로만 가입한 계정은 비밀번호가 없다")
    if not verify_password(사용자.password_hash, 몸체.current_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "현재 비밀번호가 틀리다")
    사용자.password_hash = hash_password(몸체.new_password)
    session.commit()
    _전부폐기(session, 사용자)
    return _토큰발급(session, 사용자)


@router.get("/config", response_model=AuthConfigOut)
def auth_config() -> AuthConfigOut:
    """구글 버튼을 그릴지 화면이 묻는다. 클라이언트 ID는 비밀이 아니다."""
    아이디 = get_settings().google_client_id
    return AuthConfigOut(google_client_id=아이디 or None)


@router.post("/google", response_model=TokenOut)
def google_login(몸체: GoogleIn, request: Request, response: Response,
                 session: Annotated[Session, Depends(get_session)]) -> TokenOut:
    """구글 ID 토큰으로 로그인한다. 처음이면 계정을 만들고(201) 같은 이메일이면 합친다."""
    설정 = get_settings()
    if not 설정.google_client_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "구글 로그인을 켜지 않았다")
    아이피 = request.client.host if request.client else "unknown"
    if not login_limiter.allow(아이피):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            "로그인 시도가 너무 잦다. 잠시 뒤 다시 시도하라")
    try:
        신원 = verify_google_id_token(몸체.credential, client_id=설정.google_client_id)
    except TokenInvalid as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "구글 로그인에 실패했다") from e
    if not 신원.email_verified:
        # 왜: 검증 안 된 이메일로 합치면 남의 자체 계정을 가로챌 수 있다.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "구글 로그인에 실패했다")

    이메일 = normalize_email(신원.email)
    사용자 = session.scalar(select(User).where(User.google_sub == 신원.sub))
    if 사용자 is None:
        사용자 = session.scalar(select(User).where(User.email == 이메일))
        if 사용자 is not None:
            # 합치기(설계서 §5.2): 만든 방식(auth_provider)은 그대로 두고 구글만 붙인다.
            사용자.google_sub = 신원.sub
        else:
            try:
                이름 = 표시명정리(신원.name)
            except PydanticCustomError:
                # 왜: 구글이 준 이름이 비었거나 보이지 않는 글자면 이메일 앞부분을 쓴다.
                이름 = 이메일.split("@", 1)[0][:DISPLAY_NAME_MAX] or "player"
            사용자 = User(email=이메일, display_name=이름, auth_provider="google",
                        password_hash=None, google_sub=신원.sub,
                        created_at=datetime.now(timezone.utc))
            session.add(사용자)
            response.status_code = status.HTTP_201_CREATED
        try:
            session.commit()
        except IntegrityError:
            # 왜: 같은 구글 계정이 동시에 두 번 오면 한쪽의 INSERT가 유일 제약에 걸린다.
            session.rollback()
            사용자 = session.scalar(select(User).where(User.google_sub == 신원.sub))
            if 사용자 is None:
                raise HTTPException(status.HTTP_409_CONFLICT, "다시 시도하라") from None
    return _토큰발급(session, 사용자)


@router.get("/me", response_model=UserOut)
def me(사용자: Annotated[User, Depends(current_user)]) -> UserOut:
    """지금 로그인한 사용자의 공개 정보."""
    return UserOut(id=사용자.id, email=사용자.email,
                   display_name=사용자.display_name,
                   auth_provider=사용자.auth_provider,
                   created_at=사용자.created_at)
