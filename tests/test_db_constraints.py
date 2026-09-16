"""이 파일은 DB가 겹친 요청과 잘못된 행을 스스로 막는 제약들을 확인한다.
입력: 메모리 SQLite(부분 인덱스 DDL은 SQLite·PG 방언으로도 컴파일한다).
출력: 유일 제약·사용자당 열린 라운드 인덱스·라운드 사용자 NOT NULL·외래키에 대한 pytest 결과.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import func, inspect, select, text
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateIndex

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game.db import make_engine  # noqa: E402
from game.models import Base, Decision, Game, Round, User  # noqa: E402

지금 = datetime.now(timezone.utc)


@pytest.fixture
def 세션():
    # 왜 make_engine인가: SQLite는 외래키를 기본으로 무시한다. 서버의 엔진 공장을
    #     거쳐야 PRAGMA foreign_keys=ON 이 켜져 외래키 테스트가 의미를 갖는다.
    engine = make_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def 사용자(세션, email="a@b.com"):
    u = User(email=email, display_name="지우", auth_provider="local",
             password_hash="$argon2id$dummy")
    세션.add(u)
    세션.commit()
    return u


def 게임(세션, u):
    g = Game(user_id=u.id, rules_fp="47c403568aa3", started_at=지금)
    세션.add(g)
    세션.commit()
    return g


def 라운드(g, 회차, 열림):
    return Round(game_id=g.id, user_id=g.user_id, round_no=회차, seed=회차,
                 is_open=열림, dealer_up=10)


def 결정(r, 순번):
    return Decision(round_id=r.id, seq=순번, total=16, is_soft=0, dealer_up=10,
                    can_double=1, can_split=0, split_depth=0, action_taken=1,
                    dp_optimal_action=1, dp_ev_loss=0.0, created_at=지금)


def 제약_위반(세션, 메시지):
    # 왜 원인 메시지를 보는가: IntegrityError의 문자열에는 INSERT 문 전체가 붙어
    #   칸 이름이 늘 들어 있다. 어느 제약이 막았는지는 드라이버 메시지에만 있다.
    with pytest.raises(IntegrityError) as 오류:
        세션.commit()
    assert 메시지 in str(오류.value.orig)
    세션.rollback()


def 열린_라운드_수(세션):
    return 세션.scalar(select(func.count()).select_from(Round).where(Round.is_open.is_(True)))


def test_같은_라운드의_같은_순번은_두_번_기록되지_않는다(세션):
    # 왜: 같은 결정을 동시에 두 번 보내면 둘 다 확인을 통과한다. DB가 막아야 한다.
    g = 게임(세션, 사용자(세션))
    r1, r2 = 라운드(g, 1, False), 라운드(g, 2, False)
    세션.add_all([r1, r2])
    세션.commit()
    세션.add_all([결정(r1, 0), 결정(r1, 1), 결정(r2, 0)])  # 다른 라운드의 0번은 된다
    세션.commit()
    세션.add(결정(r1, 1))
    제약_위반(세션, "UNIQUE constraint failed: decisions.round_id, decisions.seq")


def test_같은_게임의_같은_회차는_두_번_만들어지지_않는다(세션):
    # 왜 닫힌 라운드로 시험하는가: 열린 라운드면 사용자당 열린 라운드 인덱스가
    #   먼저 막아, 회차 제약이 없어도 통과해 버린다.
    u = 사용자(세션)
    g1, g2 = 게임(세션, u), 게임(세션, u)
    세션.add_all([라운드(g1, 1, False), 라운드(g2, 1, False)])  # 다른 게임의 1회차는 된다
    세션.commit()
    세션.add(라운드(g1, 1, False))
    제약_위반(세션, "UNIQUE constraint failed: rounds.game_id, rounds.round_no")


def test_한_사용자는_열린_라운드를_둘_가질_수_없다(세션):
    # 왜: 대기 중인 결정을 새 게임으로 버리는 우회를 DB가 끝까지 막는다.
    u = 사용자(세션)
    g1, g2 = 게임(세션, u), 게임(세션, u)
    첫판 = 라운드(g1, 1, True)
    세션.add(첫판)
    세션.commit()
    세션.add(라운드(g2, 1, True))  # 다른 게임이어도 막힌다
    제약_위반(세션, "UNIQUE constraint failed: rounds.user_id")

    세션.add_all([라운드(g1, 2, False), 라운드(g2, 1, False), 라운드(g2, 2, False)])
    세션.commit()  # 닫힌 라운드는 여러 개여도 된다
    첫판.is_open = False
    세션.commit()
    세션.add(라운드(g2, 3, True))  # 닫고 나면 다시 열 수 있다
    세션.commit()
    assert 열린_라운드_수(세션) == 1


def test_다른_사용자는_각자_열린_라운드를_가질_수_있다(세션):
    가 = 게임(세션, 사용자(세션, "a@b.com"))
    나 = 게임(세션, 사용자(세션, "c@d.com"))
    세션.add_all([라운드(가, 1, True), 라운드(나, 1, True)])
    세션.commit()
    assert 열린_라운드_수(세션) == 2


def test_부분_인덱스가_SQLite와_PG에서_WHERE를_가진다(세션):
    # 왜 문장 전체를 비교하는가: WHERE가 빠지면 사용자당 라운드가 평생 하나로 묶인다.
    #   조건이 is_(False)나 IS NOT NULL로 바뀌어도 "WHERE ... is_open"은 남으므로
    #   부분 문자열로는 못 잡는다. IS 형태여야 조회식 Round.is_open.is_(True)와 같아
    #   SQLite가 부분 인덱스를 쓴다(= 1 이면 SCAN).
    인덱스 = {i.name: i for i in Round.__table__.indexes}["uq_rounds_one_open_per_user"]
    머리 = "CREATE UNIQUE INDEX uq_rounds_one_open_per_user ON rounds (user_id) WHERE "
    기대 = {"sqlite": 머리 + "is_open IS 1", "postgresql": 머리 + "is_open IS true"}
    for 방언 in (sqlite.dialect(), postgresql.dialect()):
        assert str(CreateIndex(인덱스).compile(dialect=방언)) == 기대[방언.name]
    # 왜 실제 DB도 보는가: create_all이 실제로 낸 DDL이 이 문장이어야 제약이 선다.
    실제 = 세션.scalar(text(
        "SELECT sql FROM sqlite_master WHERE name = 'uq_rounds_one_open_per_user'"))
    assert 실제 == 기대["sqlite"]


def test_라운드는_사용자_없이_만들_수_없다(세션):
    # 왜: 유일 인덱스는 SQLite·PG 모두 NULL끼리 다르다고 본다. 사용자 칸이 비면
    #   열린 라운드를 몇 개든 만들 수 있어 사용자당 하나 규칙이 뚫린다.
    g = 게임(세션, 사용자(세션))
    세션.add(Round(game_id=g.id, round_no=1, seed=1, is_open=True, dealer_up=10))
    제약_위반(세션, "NOT NULL constraint failed: rounds.user_id")
    assert 세션.scalar(select(func.count()).select_from(Round)) == 0


def test_없는_사용자의_라운드는_만들_수_없다(세션):
    # 왜: 게임은 실제로 있으므로 이 INSERT를 막을 수 있는 것은 user_id 외래키뿐이다.
    #   유령 사용자에게 걸린 라운드는 전적에도, 열린 라운드 판정에도 잡히지 않는다.
    g = 게임(세션, 사용자(세션))
    세션.add(Round(game_id=g.id, user_id=99999, round_no=1, seed=1,
                  is_open=False, dealer_up=10))
    제약_위반(세션, "FOREIGN KEY constraint failed")
    assert 세션.scalar(select(func.count()).select_from(Round)) == 0
    # 왜 반영된 스키마도 보는가: SQLite 메시지는 어느 외래키인지 말하지 않는다.
    외래키 = {(tuple(f["constrained_columns"]), f["referred_table"], tuple(f["referred_columns"]))
            for f in inspect(세션.get_bind()).get_foreign_keys("rounds")}
    assert (("user_id",), "users", ("id",)) in 외래키
