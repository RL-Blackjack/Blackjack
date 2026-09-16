"""이 파일은 HTTP 라우터들을 담는 패키지를 만든다.
입력: 없음.
출력: 없음(네임스페이스).
"""

# 왜 이 순서인가: create_app()이 이 순서대로 붙인다. 계획서의 태스크 순서와 같아서
#   아직 만들지 않은 라우터는 자연히 건너뛰어진다.
ROUTER_MODULES: tuple[str, ...] = ("auth", "games", "me", "leaderboard")
