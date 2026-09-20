"""이 파일은 라운드를 열고 닫는 규칙과 게임 조회 도우미를 모은다.
입력: DB 세션과 게임 행, 그리고 엔진이 재현한 라운드 모습.
출력: 새 라운드 행, 응답 모양, 그리고 409·404 예외.
"""

from __future__ import annotations

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from blackjack_rl.rules import RULES_V1
from game.engine import Finished, Pending, RoundView, new_seed, play
from game.models import Game, ModelRegistry, Round, User
from game.play_schemas import FinishedOut, HandOut, OpponentOut, PendingOut


def 충돌(code: str, message: str, game_id: int | None = None) -> HTTPException:
    """409 하나를 만든다. detail은 문자열이 아니라 객체다."""
    # 왜 객체인가: 프런트엔드가 code로 갈라 "이어 두기" 버튼을 만들 수 있어야 한다.
    #   문자열이면 화면이 메시지 문구를 파싱해야 하고 문구를 못 바꾸게 된다.
    상세: dict[str, object] = {"code": code, "message": message}
    if game_id is not None:
        상세["game_id"] = game_id
    return HTTPException(status.HTTP_409_CONFLICT, 상세)


def 행동목록(round_: Round) -> list[int]:
    """저장된 행동 문자열을 행동 번호 목록으로 푼다."""
    return [int(a) for a in round_.actions.split(",") if a]


def 보기(view: RoundView, seq: int) -> PendingOut | FinishedOut:
    """엔진의 결과를 응답 모양으로 바꾼다."""
    if isinstance(view, Pending):
        return PendingOut(
            seq=seq, total=view.key.total, is_soft=bool(view.key.is_soft),
            dealer_up=view.dealer_up, player_cards=list(view.player_cards),
            legal=list(view.legal), hand_index=view.hand_index,
            n_hands=view.n_hands, bet=view.bet)
    return FinishedOut(
        dealer_up=view.dealer_up, dealer_cards=list(view.dealer_cards),
        net=view.net,
        hands=[HandOut(cards=list(h.cards), total=h.total, bet=h.bet,
                       result=h.result) for h in view.hands])


def 닫기(game: Game, round_: Round, view: Finished) -> None:
    """끝난 라운드에 딜러 카드와 순손익을 박고 게임 합계를 올린다."""
    round_.is_open = False
    round_.dealer_cards = ",".join(str(c) for c in view.dealer_cards)
    round_.net = float(view.net)
    # 왜 SQL 식인가: 파이썬에서 읽어 더하면 그 사이에 다른 요청이 커밋한 증가분을
    #   덮어쓴다(실측 3/3 손실). DB가 행을 잠그고 더하게 맡긴다.
    game.net_result = Game.net_result + float(view.net)


def 사용자열린라운드(session: Session, user_id: int) -> Round | None:
    """이 사용자가 지금 열어 둔 라운드. 규칙상 많아야 하나다."""
    # 왜 is_(True)인가: 부분 유일 인덱스가 WHERE is_open IS 1 로 걸려 있다.
    #   SQLite는 조회 조건이 인덱스 조건과 같은 식일 때만 그 인덱스를 쓴다.
    return session.scalar(
        select(Round).where(Round.user_id == user_id, Round.is_open.is_(True)))


def 열린라운드충돌(session: Session, user_id: int) -> HTTPException:
    """열린 라운드 때문에 막을 때 쓰는 409. 이어 둘 game_id를 함께 준다."""
    열린것 = 사용자열린라운드(session, user_id)
    return 충돌("open_round", "아직 끝나지 않은 라운드가 있다",
               None if 열린것 is None else 열린것.game_id)


def 새라운드(session: Session, game: Game) -> Round:
    """새 라운드를 딜한다. 결정 없이 끝나는 라운드면 바로 닫는다."""
    # 왜 여기서 사용자 전체를 보는가: 라운드를 만드는 유일한 길이다. 게임 생성에서만
    #   막으면 다른 게임의 딜로 대기 라운드를 버릴 수 있었다(우회 봇 99회 성공).
    if 사용자열린라운드(session, game.user_id) is not None:
        session.rollback()
        raise 열린라운드충돌(session, game.user_id)
    # 왜 한 행만 읽는가: 회차와 열림 여부를 같은 행에서 함께 정해야 한다. 개수로
    #   세면 지운 행이나 겹친 딜에서 회차가 어긋나 500이 났다.
    마지막 = session.scalar(
        select(Round).where(Round.game_id == game.id)
        .order_by(Round.round_no.desc()).limit(1))
    if 마지막 is not None and 마지막.is_open:
        raise 충돌("open_round", "아직 끝나지 않은 라운드가 있다", game.id)

    회차 = 1 if 마지막 is None else 마지막.round_no + 1
    시드 = new_seed()
    처음 = play(시드, [])
    라운드 = Round(game_id=game.id, user_id=game.user_id, round_no=회차, seed=시드,
                  actions="", is_open=True, dealer_up=int(처음.dealer_up),
                  dealer_cards="", net=0.0)
    사용자번호, 닫았다 = game.user_id, isinstance(처음, Finished)
    try:
        session.add(라운드)
        session.flush()
        if 닫았다:
            # 왜: 플레이어나 딜러가 내추럴이면 결정이 하나도 없이 끝난다.
            닫기(game, 라운드, 처음)
        session.commit()
    except IntegrityError:
        # 왜 flush까지 감싸는가: 유일 인덱스는 INSERT를 보내는 flush에서 터진다.
        #   commit만 감싸면 확인과 쓰기 사이에 끼어든 요청이 500으로 나갔다(10/10).
        session.rollback()
        raise 열린라운드충돌(session, 사용자번호) from None
    if 닫았다:
        # 왜: net_result가 SQL 식이라 커밋 뒤 값이 만료돼 있다. 응답을 만들기 전에
        #   실제 값으로 되읽어 둔다.
        session.refresh(game)
    return 라운드


def 게임찾기(session: Session, game_id: int, user: User) -> Game:
    """내 게임만 찾는다. 남의 것이면 없는 것처럼 군다."""
    게임 = session.get(Game, game_id)
    if 게임 is None or 게임.user_id != user.id:
        # 왜 404인가: 403을 주면 "그 번호의 게임이 존재한다"를 알려 주는 것이고,
        #   번호를 훑어 남이 몇 판 했는지 셀 수 있다.
        raise HTTPException(status.HTTP_404_NOT_FOUND, "그런 게임이 없다")
    return 게임


def 규칙확인(game: Game) -> None:
    """게임을 시작한 규칙이 지금 규칙과 같은지 본다. 다르면 409."""
    # 왜: 규칙이 바뀐 배포 뒤 옛 게임을 재생하면 같은 시드가 다른 딜러 카드와 다른
    #   정산을 낸다(실측). 저장된 net과 어긋난 화면을 주느니 막는 편이 낫다.
    if game.rules_fp != RULES_V1.fingerprint():
        raise 충돌("rules_changed", "이 게임은 지금과 다른 규칙으로 시작했다", game.id)


def 게임끝남확인(game: Game) -> None:
    """끝낸 게임에는 더 둘 수 없다."""
    # 왜: 끝낸 뒤에도 딜이 되면 전적에 잡히지 않는 라운드가 생긴다.
    if game.ended_at is not None:
        raise 충돌("game_ended", "이미 끝난 게임이다", game.id)


def 열린라운드(session: Session, game: Game) -> Round | None:
    """이 게임에서 열려 있는 라운드."""
    return session.scalar(
        select(Round).where(Round.game_id == game.id, Round.is_open.is_(True))
        .order_by(Round.round_no.desc()))


def 마지막라운드(session: Session, game: Game) -> Round:
    """이 게임의 가장 최근 라운드. 하나도 없으면 404."""
    라운드 = session.scalar(
        select(Round).where(Round.game_id == game.id)
        .order_by(Round.round_no.desc()).limit(1))
    if 라운드 is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "라운드가 없다")
    return 라운드


def 상대(session: Session, game: Game) -> OpponentOut | None:
    """게임이 상대한 모델의 표시용 정보."""
    if game.opponent_model_id is None:
        return None
    행 = session.get(ModelRegistry, game.opponent_model_id)
    if 행 is None:
        return None
    return OpponentOut(id=행.id, name=행.name, family=행.family,
                       exact_ev=float(행.exact_ev))
