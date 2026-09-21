"""이 파일은 데이터베이스 엔진과 세션을 만들고 스키마를 세우거나 지운다.
입력: 접속 URL(없으면 설정에서 읽는다).
출력: Engine, 세션 공장, FastAPI 의존성으로 쓰는 세션 생성기.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from game.config import get_settings
from game.models import Base

_엔진: Engine | None = None
_공장: sessionmaker[Session] | None = None


def make_engine(url: str | None = None) -> Engine:
    """접속 URL로 엔진을 만든다. SQLite면 외래키 강제를 켠다."""
    주소 = url if url is not None else get_settings().database_url
    if 주소.startswith("sqlite"):
        # 왜 check_same_thread=False 인가: FastAPI는 동기 핸들러를 스레드 풀에서
        #   돌리므로 한 연결이 다른 스레드에서 쓰일 수 있다. 세션을 요청마다 새로
        #   열기 때문에 실제로 공유되지는 않는다.
        engine = create_engine(주소, connect_args={"check_same_thread": False},
                               future=True)

        @event.listens_for(engine, "connect")
        def _외래키_켜기(dbapi_conn, _rec):  # noqa: ANN001, ANN202
            # 왜: SQLite는 기본적으로 외래키를 무시한다. 연결마다 켜 주지 않으면
            #   테스트는 통과하는데 배포(PostgreSQL)에서만 제약 위반이 터진다.
            커서 = dbapi_conn.cursor()
            커서.execute("PRAGMA foreign_keys=ON")
            커서.close()

        return engine

    # 왜 pool_pre_ping인가: 집 PC가 절전에서 깨거나 PG가 재시작하면 죽은 연결이
    #   풀에 남는다. 쓰기 직전에 한 번 확인하면 첫 요청만 실패하는 일을 막는다.
    # 왜 5+5인가: uvicorn 워커 1개의 스레드 풀이 40이지만 요청 대부분이 1ms 안에
    #   끝나 연결 10개면 충분하다. PG 기본 max_connections 100 안에서 백업·psql 자리를 남긴다.
    return create_engine(주소, pool_pre_ping=True, pool_size=5, max_overflow=5,
                         pool_timeout=10, future=True)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """엔진 하나에 묶인 세션 공장을 만든다."""
    # 왜 expire_on_commit=False 인가: 커밋한 뒤에도 객체 필드를 읽어 응답을
    #   만들어야 하는데, 기본값이면 커밋 순간 만료돼 DB를 다시 친다.
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def create_schema(engine: Engine) -> None:
    """표를 만든다. 이미 있으면 아무 일도 하지 않는다."""
    Base.metadata.create_all(engine)


def drop_schema(engine: Engine) -> None:
    """표를 전부 지운다. 개발과 테스트에서만 쓴다."""
    Base.metadata.drop_all(engine)


def reset_engine() -> None:
    """모듈에 캐시된 엔진을 버린다. 테스트가 DB를 갈아끼울 때 쓴다."""
    global _엔진, _공장
    if _엔진 is not None:
        _엔진.dispose()
    _엔진 = None
    _공장 = None


def _세션공장() -> sessionmaker[Session]:
    """엔진과 공장을 처음 쓸 때 한 번 만들고 계속 재사용한다."""
    global _엔진, _공장
    if _공장 is None:
        _엔진 = make_engine()
        create_schema(_엔진)
        _공장 = make_session_factory(_엔진)
    return _공장


def get_session() -> Iterator[Session]:
    """FastAPI 의존성. 요청마다 세션을 열고 끝나면 닫는다."""
    세션 = _세션공장()()
    try:
        yield 세션
    finally:
        세션.close()
