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

## 배포 (집 PC, Docker Desktop)

```bash
cp deploy/.env.example deploy/.env   # 값 채우기. 도메인이 아직 없으면 PLAY_HOST=play.localhost, RESEARCH_HOST=research.localhost 로 두면 Caddy 가 자체 서명 인증서를 쓴다(둘이 같으면 사이트 중복으로 기동을 거부한다).
docker compose -f deploy/docker-compose.yml up -d --build
docker compose -f deploy/docker-compose.yml ps           # 5개 running, db healthy. backup 이 "Restarting" 이면 스크립트가 죽고 있는 것
curl -k https://play.localhost/api/health                # {"status":"ok","rules_fp":...,"models_loaded":13}
docker compose -f deploy/docker-compose.yml logs backup  # "backup: /backups/bj-....sql.gz" 한 줄, 그 뒤 오류 없음
ls -l deploy/backups/                                    # 완료 조건 1: bj-YYYYMMDD-HHMMSS.sql.gz 가 있고 크기 > 0
gunzip -t deploy/backups/bj-*.sql.gz                     # 완료 조건 2: 종료 코드 0(gzip 이 온전하다)
```

브라우저로 `https://play.localhost/` 가입 → 한 판. 첫 기동에서 `api` 가 `scripts/register_models.py` 로
모델 13개를 레지스트리에 올린다.

실행 기록: (아직 없음 — Docker Desktop 을 켠 뒤 위 절차를 한 번 돌리고 날짜·결과를 적는다)

## 복원 훈련 (설계서 §8.2 "한 번은 실제로 해본다")

```bash
docker compose -f deploy/docker-compose.yml stop api
docker compose -f deploy/docker-compose.yml exec -T db psql -U bj -d postgres -c "DROP DATABASE bj; CREATE DATABASE bj OWNER bj;"
docker compose -f deploy/docker-compose.yml exec -T db psql -U bj -d bj -c "GRANT USAGE, CREATE ON SCHEMA public TO bj_app;"
gunzip -c deploy/backups/bj-최신.sql.gz | docker compose -f deploy/docker-compose.yml exec -T db psql -U bj -d bj
docker compose -f deploy/docker-compose.yml start api
python scripts/preflight.py https://play.localhost   # 그리고 브라우저에서 옛 계정으로 로그인 → 전적이 그대로
```

복원 훈련 기록: (아직 없음 — 날짜 / 걸린 시간 / 막힌 곳)

## 발표 당일 점검표 (30분 전)
1. 전날: 윈도우 업데이트 일시 중지, 자동 재시작 끄기, 절전 끄기.
2. `docker compose -f deploy/docker-compose.yml ps` → 5개 running.
3. `python scripts/preflight.py https://PLAY_HOST` → 전부 OK.
4. 2차(외부 회선이 죽어도 시연): 발표 노트북의 hosts 파일(`C:\Windows\System32\drivers\etc\hosts`)에
   `127.0.0.1 PLAY_HOST` 와 `127.0.0.1 RESEARCH_HOST` 를 적어 두면, 인터넷 없이도 같은 주소·같은 인증서로
   Caddy 를 거쳐 접속된다(api 포트는 밖에 열지 않으므로 localhost:8000 은 닿지 않는다 — 그 주소를 쓰지 않는다).
   리허설에서 랜선·핫스팟을 끄고 `python scripts/preflight.py https://PLAY_HOST` 가 그대로 OK 인지 실제로 본다.
5. 3차: 슬라이드에 media/*.mp4 와 최종 전략표 PNG 가 들어 있는지.
6. 휴대폰 핫스팟으로 https://PLAY_HOST 접속해 가입 → 한 판 → 순위표까지 눌러 본다.
7. 리허설 3회 기록: 날짜 / 걸린 시간 / 막힌 곳 / 2차 경로 확인 여부.
