"""이 파일은 게임 서버가 쓰는 데이터베이스 표들을 정의한다.
입력: 없음(SQLAlchemy 선언만).
출력: Base와 여섯 개의 매핑 클래스.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """모든 표의 공통 조상."""


class User(Base):
    """회원 한 명. 자체 가입과 구글 로그인을 같은 표에 담는다."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    # 왜 citext가 아닌가: PostgreSQL 전용 타입이라 SQLite에서 테스트할 수 없다.
    #   대신 저장 전에 소문자로 정규화한다(game/auth.py 의 normalize_email).
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(64), nullable=False)
    auth_provider: Mapped[str] = mapped_column(String(16), nullable=False)
    # 왜 nullable인가: 구글로만 가입한 사람은 비밀번호가 아예 없다.
    password_hash: Mapped[str | None] = mapped_column(String(256), nullable=True)
    google_sub: Mapped[str | None] = mapped_column(String(64), unique=True, nullable=True)
    # 왜 default가 있는가: brief의 테스트(사용자(), 구글 가입 테스트)가
    #   created_at을 넘기지 않는다. NOT NULL 제약은 그대로 두고
    #   ORM 쪽에서 가입 시각을 자동으로 채운다.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc))

    games: Mapped[list["Game"]] = relationship(back_populates="user")


class RefreshToken(Base):
    """리프레시 토큰. 원문이 아니라 해시만 저장한다."""

    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    # 왜 해시인가: 원문을 저장하면 DB가 유출될 때 그대로 로그인에 쓰인다.
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True)


class ModelRegistry(Base):
    """서빙 중인 모델 목록. 게임은 여기 있는 것만 상대로 고를 수 있다."""

    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    family: Mapped[str] = mapped_column(String(16), nullable=False)
    rules_fp: Mapped[str] = mapped_column(String(12), nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(256), nullable=False)
    artifact_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    exact_ev: Mapped[float] = mapped_column(Float, nullable=False)
    is_serving: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class Game(Base):
    """한 사람이 한 모델을 상대로 진행한 게임 한 판(여러 라운드)."""

    __tablename__ = "games"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    opponent_model_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_registry.id"), nullable=True)
    # 왜 게임마다 규칙 지문을 박는가: 규칙이 바뀌면 옛 기록과 새 기록을 섞어
    #   집계하면 안 된다. 지문이 다르면 통계에서 분리할 수 있다.
    rules_fp: Mapped[str] = mapped_column(String(12), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True)
    net_result: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    user: Mapped[User] = relationship(back_populates="games")
    rounds: Mapped[list["Round"]] = relationship(back_populates="game")


class Round(Base):
    """게임 안의 한 라운드. 카드가 아니라 '시드와 행동 목록'을 저장한다."""

    __tablename__ = "rounds"

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), nullable=False)
    round_no: Mapped[int] = mapped_column(Integer, nullable=False)

    # 왜 시드와 행동만 저장하는가: 서버가 게임 상태의 주인이라는 원칙을 가장 싸게
    #   지키는 방법이다. 카드를 저장하면 그 값을 믿어야 하지만, 시드는 코어로 다시
    #   돌려 재현하므로 믿을 것이 없다. 재생 1회 실측 55.8us, 한 라운드의 결정은
    #   최대 9개이므로 요청당 비용이 사실상 0이다.
    seed: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 지금까지 고른 행동을 쉼표로 이은 것. 아직 아무것도 안 골랐으면 빈 문자열이다.
    # 왜 128자인가: 스플릿·히트를 최대한 밀어붙이는 플레이로 20,000판을 돌렸을 때
    #   한 라운드의 최대 결정이 16개였고 직렬화하면 32자다. 이론상 손 4개가 각자
    #   열 번쯤 히트하면 40결정(80자)까지 갈 수 있으므로 128자로 여유를 둔다.
    actions: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    is_open: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    dealer_up: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    # 왜 배열이 아니라 문자열인가: PostgreSQL 배열 타입은 SQLite에 없다.
    #   "10,7,5" 처럼 쉼표로 이어 두면 양쪽에서 같은 코드가 돈다.
    #   라운드가 끝나기 전에는 빈 문자열이다.
    dealer_cards: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    net: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    game: Mapped[Game] = relationship(back_populates="rounds")
    decisions: Mapped[list["Decision"]] = relationship(back_populates="round")


class Decision(Base):
    """사람이 내린 결정 하나. 이 표가 전적·리더보드·모방 모델의 원천이다."""

    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(primary_key=True)
    round_id: Mapped[int] = mapped_column(ForeignKey("rounds.id"), nullable=False)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)

    # 상태 키 여섯 필드를 그대로 담는다. 그래야 (X, y) 데이터셋이 바로 나온다.
    total: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    is_soft: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    dealer_up: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    can_double: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    can_split: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    # 왜 늘 0이 들어가는가: RULES_V1은 include_split_depth_in_state=False라
    #   상태 키가 스플릿 깊이를 담지 않는다(실측 확인). 설계서 §6.2가 요구하는
    #   칸이므로 두되, 깊이를 켜는 규칙으로 바꾸면 그때 의미가 생긴다.
    #   스플릿 여부는 이 칸이 아니라 라운드의 손 개수로 판단한다.
    split_depth: Mapped[int] = mapped_column(SmallInteger, nullable=False)

    action_taken: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    # 왜 저장 시점에 계산해 넣는가: 전적과 리더보드를 재계산 없이 집계할 수 있고,
    #   나중에 DP를 다시 풀지 않아도 "그때 정답이 무엇이었나"가 남는다.
    dp_optimal_action: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    dp_ev_loss: Mapped[float] = mapped_column(Float, nullable=False)
    ai_action: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    round: Mapped[Round] = relationship(back_populates="decisions")


# 왜 이 인덱스들인가: 전적은 사용자의 최근 게임을, 일치율은 라운드별 결정을
#   훑는다. 이 두 경로가 전체 조회의 대부분이다.
Index("ix_games_user_started", Game.user_id, Game.started_at.desc())
Index("ix_rounds_game", Round.game_id)
# 왜: 행동 API는 매번 "이 게임의 열린 라운드"를 찾는다. 가장 잦은 조회다.
Index("ix_rounds_open", Round.game_id, Round.is_open)
Index("ix_decisions_round", Decision.round_id)
