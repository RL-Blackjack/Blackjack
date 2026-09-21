"""이 파일은 게임 서버 테스트가 함께 쓰는 DB 주소 스위치와 스키마 준비를 모은다.
입력: pytest 의 tmp_path 와 환경변수 BJ_TEST_PG_URL.
출력: 이번 테스트가 쓸 DB 주소, 그리고 비운 스키마.
"""

import os
from pathlib import Path

from sqlalchemy import Engine

from game.db import create_schema, drop_schema


def test_db_url(tmp_path: Path, name: str = "game.db") -> str:
    """BJ_TEST_PG_URL 이 있으면 그 PostgreSQL 을, 없으면 임시 SQLite 파일을 쓴다."""
    # 왜 환경변수인가: 같은 테스트를 SQLite(빠름, 기본)와 실제 PG(배포 전 확인)로
    #   돌린다. PG 는 TimeZone=Asia/Seoul 로 띄워 시각 정규화까지 검증한다.
    return os.environ.get("BJ_TEST_PG_URL") or f"sqlite:///{tmp_path / name}"


def using_pg() -> bool:
    """지금 PG 로 돌고 있는가. 인메모리 SQLite 만 가능한 픽스처가 분기할 때 쓴다."""
    return bool(os.environ.get("BJ_TEST_PG_URL"))


def fresh_schema(engine: Engine) -> None:
    """표를 비우고 새로 만든다. PG 는 앞 테스트의 행이 남아 있으므로 먼저 지운다."""
    if using_pg():
        drop_schema(engine)
    create_schema(engine)
