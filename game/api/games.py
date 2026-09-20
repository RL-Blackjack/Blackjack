"""이 파일은 게임 생성과 행동 처리, 결정 기록 엔드포인트를 제공한다.
입력: 로그인한 사용자의 행동 요청.
출력: 라운드의 지금 모습과 DP 평가, 그리고 decisions 표의 새 행.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Response, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from blackjack_rl.rules import RULES_V1
from blackjack_rl.state import legal_actions
from game.db import get_session
from game.deps import current_user
from game.engine import Finished, IllegalAction, Pending, play
from game.models import Decision, Game, User
from game.optimal import ev_loss, optimal_action
from game.play_schemas import (
    ActIn, ActOut, Feedback, GameOut, NewGameIn, OpponentOut, RoundOut)
from game.rounds import (
    게임끝남확인,
    게임찾기,
    규칙확인,
    닫기,
    마지막라운드,
    보기,
    사용자열린라운드,
    상대,
    새라운드,
    열린라운드,
    열린라운드충돌,
    충돌,
    행동목록,
)
from game.serving import model_action, store

router = APIRouter(prefix="/api", tags=["games"])

# 왜 위아래를 막는가: 파이썬 int는 무한정이라 거대한 번호가 그대로 DB 드라이버에
#   닿아 500이 났다. games.id는 32비트 정수 키이므로 그 범위 밖은 있을 수 없다.
GameId = Annotated[int, Path(ge=1, le=2**31 - 1)]


def _저장소(session: Session):
    """레지스트리가 바뀌었을 때만 모델을 다시 읽고 저장소를 돌려준다."""
    # 왜 ensure_fresh인가: 지문 조회(13행) 한 번이면 재학습 덮어쓰기와 서빙 끄기를
    #   잡아낸다. "비어 있으면 읽기"는 전부 실패했을 때 요청마다 파일을 다시 열었다.
    return store.ensure_fresh(session)


@router.get("/models", response_model=list[OpponentOut])
def list_models(session: Annotated[Session, Depends(get_session)]) -> list[OpponentOut]:
    """서빙 중인 상대 모델을 EV가 좋은 순으로 준다."""
    return [OpponentOut(id=m.id, name=m.name, family=m.family, exact_ev=m.exact_ev)
            for m in _저장소(session).serving()]


@router.post("/games", response_model=GameOut, status_code=status.HTTP_201_CREATED)
def create_game(몸체: NewGameIn,
                사용자: Annotated[User, Depends(current_user)],
                session: Annotated[Session, Depends(get_session)]) -> GameOut:
    """새 게임을 만들고 첫 라운드를 딜한다."""
    저장소 = _저장소(session)
    if 몸체.opponent_model_id is None:
        상대모델 = 저장소.default()
    else:
        # 왜 서빙 목록으로 판정하는가: 레지스트리 행은 남아 있어도 is_serving을 끈
        #   모델은 올라와 있지 않다. 행으로 판정하면 끈 모델로 게임이 시작됐다.
        상대모델 = 저장소.get(몸체.opponent_model_id)
        if 상대모델 is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "그런 모델이 없다")

    if 사용자열린라운드(session, 사용자.id) is not None:
        # 왜 INSERT 전인가: 여기서 막아야 빈 게임 행이 남지 않는다. 새 게임으로
        #   대기 라운드를 버리면 그 결정이 기록되지 않아 일치율이 지워진다.
        raise 열린라운드충돌(session, 사용자.id)

    게임 = Game(user_id=사용자.id,
               opponent_model_id=None if 상대모델 is None else 상대모델.id,
               # 왜 해시까지 박는가: 레지스트리 행은 같은 이름으로 덮어쓰인다.
               #   시작 시점의 파일 해시가 있어야 ai_action의 출처를 가려낸다.
               opponent_artifact_sha256=(None if 상대모델 is None
                                         else 상대모델.artifact_sha256),
               rules_fp=RULES_V1.fingerprint(),
               started_at=datetime.now(timezone.utc), net_result=0.0)
    session.add(게임)
    session.flush()
    라운드 = 새라운드(session, 게임)

    return GameOut(game_id=게임.id, opponent=상대(session, 게임),
                   round_no=라운드.round_no,
                   round=보기(play(라운드.seed, 행동목록(라운드)), 0))


@router.get("/games/{game_id}", response_model=GameOut)
def read_game(game_id: GameId,
              사용자: Annotated[User, Depends(current_user)],
              session: Annotated[Session, Depends(get_session)]) -> GameOut:
    """게임의 지금 모습. 상태는 언제나 시드로 다시 만든다."""
    게임 = 게임찾기(session, game_id, 사용자)
    규칙확인(게임)
    라운드 = 마지막라운드(session, 게임)
    행동 = 행동목록(라운드)
    return GameOut(game_id=게임.id, opponent=상대(session, 게임),
                   round_no=라운드.round_no,
                   round=보기(play(라운드.seed, 행동), len(행동)))


@router.post("/games/{game_id}/rounds", response_model=RoundOut,
             status_code=status.HTTP_201_CREATED)
def deal_round(game_id: GameId,
               사용자: Annotated[User, Depends(current_user)],
               session: Annotated[Session, Depends(get_session)]) -> RoundOut:
    """같은 게임에서 새 라운드를 딜한다."""
    게임 = 게임찾기(session, game_id, 사용자)
    규칙확인(게임)
    게임끝남확인(게임)
    # 왜 여기서 열림을 따로 안 보는가: 새라운드가 마지막 라운드 한 행으로 회차와
    #   열림을 함께 판정한다. 두 번 읽으면 그 사이가 또 하나의 경합 창이 된다.
    라운드 = 새라운드(session, 게임)
    return RoundOut(round_no=라운드.round_no, round=보기(play(라운드.seed, []), 0))


@router.post("/games/{game_id}/act", response_model=ActOut)
def act(game_id: GameId, 몸체: ActIn,
        사용자: Annotated[User, Depends(current_user)],
        session: Annotated[Session, Depends(get_session)]) -> ActOut:
    """행동 하나를 처리하고 DP 정답·EV 손실과 함께 기록한다."""
    게임 = 게임찾기(session, game_id, 사용자)
    규칙확인(게임)
    게임끝남확인(게임)
    라운드 = 열린라운드(session, 게임)
    if 라운드 is None:
        raise 충돌("no_open_round", "열려 있는 라운드가 없다", 게임.id)

    행동 = 행동목록(라운드)
    if 몸체.seq != len(행동):
        # 왜: 같은 요청이 두 번 오거나 순서가 뒤엉킨 것이다. 조용히 받아들이면
        #   결정이 두 번 기록돼 일치율이 망가진다.
        raise 충돌("stale_seq",
                 f"지금 기다리는 결정은 {len(행동)}번인데 {몸체.seq}번이 왔다", 게임.id)

    지금 = play(라운드.seed, 행동)
    if not isinstance(지금, Pending):
        raise 충돌("round_finished", "이 라운드는 이미 끝났다", 게임.id)

    키 = 지금.key
    합법 = legal_actions(키)
    if not 합법[몸체.action]:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"지금 고를 수 없는 행동이다: {몸체.action}")

    # 왜 깊이를 넘기는가: 키는 규칙상 깊이를 담지 않는다. 갈라진 손을 깊이 0 Q로
    #   재면 손실이 과대로 나온다(최대 0.111). 엔진이 센 실제 깊이로 조회한다.
    깊이 = 지금.split_depth
    정답 = optimal_action(키, 합법, depth=깊이)
    손실 = ev_loss(키, 몸체.action, depth=깊이)
    상대모델 = _저장소(session).get(게임.opponent_model_id or -1)
    ai = model_action(상대모델, 키, 합법) if 상대모델 is not None else None

    새행동 = [*행동, int(몸체.action)]
    try:
        다음 = play(라운드.seed, 새행동)
    except IllegalAction as e:
        session.rollback()
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e

    닫았다 = isinstance(다음, Finished)
    try:
        session.add(Decision(
            round_id=라운드.id, seq=len(행동),
            total=키.total, is_soft=키.is_soft, dealer_up=키.dealer_up,
            can_double=키.can_double, can_split=키.can_split,
            # 왜 키의 값인가: decisions의 6필드는 상태 키 그대로여야 (X, y)
            #   데이터셋이 바로 나온다. 실제 깊이는 DP 조회에만 쓴다.
            split_depth=키.split_depth,
            action_taken=int(몸체.action), dp_optimal_action=int(정답),
            dp_ev_loss=float(손실), ai_action=ai,
            created_at=datetime.now(timezone.utc)))
        라운드.actions = ",".join(str(a) for a in 새행동)
        if 닫았다:
            닫기(게임, 라운드, 다음)
        session.commit()
    except IntegrityError:
        # 왜: 같은 seq가 동시에 들어왔다. 파이썬 순번 확인만으로는 둘 다 통과하므로
        #   (round_id, seq) 유일 제약이 두 번째를 거부한다.
        session.rollback()
        raise 충돌("conflict", "같은 결정이 동시에 들어왔다", 게임.id) from None
    if 닫았다:
        # 왜: net_result가 SQL 식이라 커밋 뒤 만료돼 있다. 실제 값으로 되읽는다.
        session.refresh(게임)

    return ActOut(
        feedback=Feedback(dp_optimal_action=int(정답), dp_ev_loss=float(손실),
                          was_optimal=int(정답) == int(몸체.action), ai_action=ai),
        round=보기(다음, len(새행동)))


@router.post("/games/{game_id}/finish", status_code=status.HTTP_204_NO_CONTENT)
def finish_game(game_id: GameId,
                사용자: Annotated[User, Depends(current_user)],
                session: Annotated[Session, Depends(get_session)]) -> Response:
    """게임을 끝낸다. 열린 라운드가 남아 있으면 끝낼 수 없다."""
    # 왜 규칙 지문을 안 보는가: 끝내기는 라운드를 재생하지 않는다. 막으면 규칙이
    #   바뀐 옛 게임에 갇혀 새 게임도 시작하지 못한다.
    게임 = 게임찾기(session, game_id, 사용자)
    if 열린라운드(session, 게임) is not None:
        # 왜: 열린 라운드를 둔 채 끝내면 그 결정이 기록 없이 사라진다.
        raise 충돌("open_round", "아직 끝나지 않은 라운드가 있다", 게임.id)
    if 게임.ended_at is None:
        게임.ended_at = datetime.now(timezone.utc)
        session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
