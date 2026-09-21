# 배포 안내

집 PC 한 대에서 Docker Compose 로 게임 서버·연구 대시보드·PostgreSQL 을 띄운다.
설계서 §8 의 구성 그대로다: Caddy(자동 HTTPS) → FastAPI / Streamlit → PostgreSQL(포트 비공개) + 매일 백업.

## PG로 테스트 돌리기

기본 테스트는 SQLite 로 돈다. 배포 전에 한 번은 실제 PostgreSQL 로 같은 테스트를 돌려
시각(TimeZone=Asia/Seoul)·부분 유일 인덱스·rowcount 가 PG 에서도 같은지 확인한다.

```bash
docker compose -f deploy/docker-compose.test.yml up -d
BJ_TEST_PG_URL="postgresql+psycopg://bj:bj@127.0.0.1:5433/bj_test" PYTHONIOENCODING=utf-8 \
  ./.venv/Scripts/python.exe -m pytest tests/test_game_api.py tests/test_game_rules.py \
  tests/test_game_round_lifecycle.py tests/test_game_stale_rounds.py tests/test_game_concurrency.py \
  tests/test_stats.py tests/test_leaderboard.py tests/test_leaderboard_rules.py \
  tests/test_auth_api.py tests/test_auth_tokens.py tests/test_auth_google.py \
  -q -W ignore -p no:cacheprovider
docker compose -f deploy/docker-compose.test.yml down
```

`BJ_TEST_PG_URL` 이 있으면 `tests/game_helpers.py` 의 스위치가 위 파일들의 DB 픽스처를
전부 그 PG 로 돌리고, 테스트마다 표를 지우고 새로 만든다. 실패하면 기준값을 바꾸지 말고
원인을 찾는다.

실행 기록: (아직 없음 — Docker Desktop 을 켠 뒤 위 명령을 돌리고 여기에 날짜·개수·시간을 적는다)
