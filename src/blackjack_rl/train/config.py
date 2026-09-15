"""이 파일은 학습 실험 한 번의 설정을 얼려 두고 12자 지문을 찍는다.
입력: CLI 플래그(argparse Namespace) 또는 직접 넘긴 필드 값들.
출력: 불변 ExperimentConfig, 설정 지문, JSON 덤프 문자열, 산출물 파일 이름 줄기."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from dataclasses import dataclass

from blackjack_rl.rules import (RULES_H17, RULES_NO_DAS, RULES_SHOE, RULES_V1,
                                RuleSet)

# 왜: dqn은 torch를 import해야 하는데, src/blackjack_rl 안에서 torch를 쓰는 파일은
#     agents/dqn.py 하나로 못박혀 있다. 이 목록에 dqn을 넣어 두면 runner가
#     '지원 안 함' 분기를 영원히 들고 다녀야 하므로 애초에 빼 둔다.
ALGOS: tuple[str, ...] = ("mc", "q", "doubleq", "sarsa", "esarsa")
START_DISTS: tuple[str, ...] = ("natural", "exploring")
STEPS: tuple[str, ...] = ("sample_average", "constant")
# 왜 s1 하나뿐인가: agents/tabular.ENCODERS와 같은 이유다. s2는 계획서 ④ W9의 몫이고,
#     지금 열어 두면 파싱만 되고 동작이 없는 플래그가 된다.
ENCODERS: tuple[str, ...] = ("s1",)

RULE_PRESETS: dict[str, RuleSet] = {
    "v1": RULES_V1,
    "h17": RULES_H17,
    "no_das": RULES_NO_DAS,
    "shoe": RULES_SHOE,
}

# 왜: 학습곡선의 x축은 np.geomspace(1000, n, frames)로 깔린다. 첫 지점이 1000이므로
#     설정 단계에서 n이 그보다 작으면 곡선이 거꾸로 간다. 상수를 여기 한 벌만 두고
#     snapshot.log_schedule이 이 값을 가져다 쓴다.
LOG_SCHEDULE_START: int = 1_000

# 왜: 추정이 아니라 실측이다. play_round + compute_transitions + N/Q 갱신을 모두 포함한
#     MC 학습 루프를 6만 에피소드 돌려 59,919 eps/s가 나왔다. --live는 이 값으로
#     '초'를 '에피소드 수'로 환산한다.
LIVE_EPS_PER_SEC: int = 60_000

# 왜: 헤드라인 롱런은 exploring starts 3,000만 에피소드다. 라이브 화면에
#     축소 배율을 그대로 띄우는 편이 과장보다 강하다(설계서 §5).
HEADLINE_EPISODES: int = 30_000_000


@dataclass(frozen=True)
class ExperimentConfig:
    """학습 한 번의 전체 설정. 파일명·meta·JSON 덤프가 전부 이 객체 하나를 본다."""

    name: str
    algo: str
    rules: RuleSet
    seed: int
    n_episodes: int
    start_dist: str = "natural"
    step: str = "sample_average"
    alpha: float = 0.02
    eps0: float = 0.25
    eps_final: float = 0.02
    eps_decay_at: int = 5_000_000
    encoder: str = "s1"
    n_frames: int = 200

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name이 비어 있다.")
        # 왜: 파일명이 '이름__규칙지문__설정지문__seedNN' 꼴이라 이름 안의 '__'는
        #     나중에 파일명을 되읽을 때 조각 개수를 망가뜨린다.
        if "__" in self.name:
            raise ValueError(f"name에 '__'를 쓸 수 없다: {self.name!r}")
        for 금지문자 in ("/", "\\", " "):
            if 금지문자 in self.name:
                raise ValueError(f"name에 {금지문자!r}를 쓸 수 없다: {self.name!r}")

        if self.algo not in ALGOS:
            raise ValueError(f"algo는 {ALGOS} 중 하나여야 한다: {self.algo!r}")
        if self.start_dist not in START_DISTS:
            raise ValueError(
                f"start_dist는 {START_DISTS} 중 하나여야 한다: {self.start_dist!r}")
        if self.step not in STEPS:
            raise ValueError(f"step은 {STEPS} 중 하나여야 한다: {self.step!r}")
        if self.encoder not in ENCODERS:
            raise ValueError(
                f"encoder는 {ENCODERS} 중 하나여야 한다: {self.encoder!r} "
                "(s2는 계획서 ④ W9에서 추가한다)")

        if not isinstance(self.rules, RuleSet):
            raise TypeError(f"rules는 RuleSet이어야 한다: {type(self.rules)}")
        if self.seed < 0:
            raise ValueError(f"seed는 0 이상이어야 한다: {self.seed}")
        if self.n_frames < 2:
            raise ValueError(f"n_frames는 2 이상이어야 한다: {self.n_frames}")

        최소 = LOG_SCHEDULE_START + self.n_frames
        if self.n_episodes < 최소:
            raise ValueError(
                f"n_episodes는 {최소} 이상이어야 한다(로그 일정 시작점 "
                f"{LOG_SCHEDULE_START} + 프레임 {self.n_frames}): {self.n_episodes}")

        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha는 0보다 크고 1 이하여야 한다: {self.alpha}")
        if not 0.0 <= self.eps_final <= self.eps0 <= 1.0:
            raise ValueError(
                f"0 <= eps_final <= eps0 <= 1 이어야 한다: "
                f"eps0={self.eps0}, eps_final={self.eps_final}")
        if self.eps_decay_at < 1:
            raise ValueError(f"eps_decay_at은 1 이상이어야 한다: {self.eps_decay_at}")

    def eps_at(self, episode: int) -> float:
        """episode번째 에피소드에서 쓸 탐험률. eps0에서 eps_final까지 직선으로 내린다."""
        # 왜: ε 스케줄이 두 군데 있으면 ev_behavior 곡선이 실제 행동정책과 어긋난다.
        #     TabularAgent.current_eps()가 self.t로 계산하는 식과 글자 그대로 같다.
        if episode <= 0:
            return float(self.eps0)
        if episode >= self.eps_decay_at:
            return float(self.eps_final)
        비율 = episode / self.eps_decay_at
        return float(self.eps0 + (self.eps_final - self.eps0) * 비율)

    def fingerprint(self) -> str:
        """설정 한 벌을 12자 문자열로 요약한다. 파일명과 npz meta에 박힌다."""
        조각들: list[str] = []
        for 필드 in dataclasses.fields(self):
            # 왜: name은 사람이 붙이는 꼬리표일 뿐이고 파일명 앞머리에 따로 들어간다.
            #     이름만 바꾼 두 실행이 서로 다른 실험으로 보이면 안 된다.
            if 필드.name == "name":
                continue
            값 = getattr(self, 필드.name)
            if isinstance(값, RuleSet):
                값 = 값.fingerprint()
            조각들.append(f"{필드.name}={값!r}")
        본문 = ";".join(조각들)
        # 왜: sha256은 실행할 때마다 값이 같다. 파이썬 내장 hash()는 실행마다 달라져
        #     파일명에 쓰면 어제 만든 산출물을 오늘 못 찾는다.
        return hashlib.sha256(본문.encode("utf-8")).hexdigest()[:12]

    def artifact_stem(self) -> str:
        """산출물 파일 이름의 확장자 앞부분."""
        return (f"{self.name}__{self.rules.fingerprint()}"
                f"__{self.fingerprint()}__seed{self.seed:02d}")

    def to_json(self) -> str:
        """산출물 옆에 그대로 떨어뜨릴 JSON 문자열. npz meta의 'config' 블록과 같다."""
        본문 = {
            "name": self.name,
            "algo": self.algo,
            "seed": self.seed,
            "n_episodes": self.n_episodes,
            "start_dist": self.start_dist,
            "step": self.step,
            "alpha": self.alpha,
            "eps0": self.eps0,
            "eps_final": self.eps_final,
            "eps_decay_at": self.eps_decay_at,
            "encoder": self.encoder,
            "n_frames": self.n_frames,
            "rules": dataclasses.asdict(self.rules),
            "rules_fp": self.rules.fingerprint(),
            "cfg_fp": self.fingerprint(),
            "artifact_stem": self.artifact_stem(),
        }
        return json.dumps(본문, ensure_ascii=False, indent=2, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    """학습 실행용 CLI 파서. 이 저장소에 학습 파서는 이것 하나뿐이다."""
    parser = argparse.ArgumentParser(description="블랙잭 RL 표 기반 학습 실행")
    parser.add_argument("--name", type=str, default="mc_natural",
                        help="산출물 파일 이름의 앞머리")
    parser.add_argument("--algo", type=str, default="mc", choices=list(ALGOS))
    parser.add_argument("--rules", type=str, default="v1",
                        choices=sorted(RULE_PRESETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=10_000_000)
    parser.add_argument("--start-dist", type=str, default="natural",
                        choices=list(START_DISTS))
    parser.add_argument("--step", type=str, default="sample_average",
                        choices=list(STEPS))
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--eps0", type=float, default=0.25)
    parser.add_argument("--eps-final", type=float, default=0.02)
    parser.add_argument("--eps-decay-at", type=int, default=5_000_000)
    parser.add_argument("--frames", type=int, default=200)
    # 왜 여기 있나: 아래 셋은 ExperimentConfig 필드가 아니다(설정 지문에 안 들어간다).
    #     산출물을 어디 둘지, 라이브로 볼지, 일치율을 몇 핸드로 잴지는 '보고 설정'이고
    #     같은 학습 결과를 만든다. 그래도 파서는 하나여야 하므로 여기 둔다.
    parser.add_argument("--artifacts", type=str, default="artifacts")
    parser.add_argument("--agreement-hands", type=int, default=200_000,
                        help="실시간 일치율 훅의 자연 빈도 표본 수(0이면 훅을 끈다)")
    parser.add_argument("--live", action="store_true",
                        help="로컬 전용 미니 실시간 학습. 스냅샷을 저장하지 않는다")
    parser.add_argument("--live-seconds", type=int, default=60)
    parser.add_argument("--live-every", type=int, default=3,
                        help="라이브 화면 갱신 간격(초)")
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """argparse 결과를 얼린 설정 객체로 바꾼다."""
    에피소드 = args.episodes
    프레임 = args.frames
    if args.live:
        에피소드 = LIVE_EPS_PER_SEC * args.live_seconds
        # 왜: 프레임 하나가 화면 한 번 갱신이다. 최소 2는 있어야 '변하는 것'이 보인다.
        프레임 = max(2, args.live_seconds // args.live_every)
    return ExperimentConfig(
        name=args.name,
        algo=args.algo,
        rules=RULE_PRESETS[args.rules],
        seed=args.seed,
        n_episodes=에피소드,
        start_dist=args.start_dist,
        step=args.step,
        alpha=args.alpha,
        eps0=args.eps0,
        eps_final=args.eps_final,
        eps_decay_at=args.eps_decay_at,
        n_frames=프레임,
    )
