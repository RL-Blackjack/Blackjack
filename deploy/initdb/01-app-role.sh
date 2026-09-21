#!/bin/sh
# 앱 전용 계정. 슈퍼유저가 아니고, 표를 만들고 읽고 쓰는 권한만 가진다(설계서 §8.2).
# postgres 공식 이미지가 데이터 볼륨이 비어 있을 때 한 번만 실행한다.
set -eu
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<EOSQL
CREATE ROLE bj_app LOGIN PASSWORD '$APP_DB_PASSWORD' NOSUPERUSER NOCREATEDB NOCREATEROLE;
GRANT CONNECT ON DATABASE bj TO bj_app;
GRANT USAGE, CREATE ON SCHEMA public TO bj_app;
EOSQL
