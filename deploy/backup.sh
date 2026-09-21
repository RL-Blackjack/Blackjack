#!/bin/sh
# deploy/backup.sh — 매일 pg_dump 하나, 7일 지난 것은 지운다(설계서 §8.2).
# 왜 sleep 루프인가: cron 을 컨테이너에 넣는 것보다 읽기 쉽고, 시계가 어긋나도 하루 한 번은 돈다.
set -eu
mkdir -p /backups
while true; do
  # 왜 변수 이름이 영문인가: sh(dash)는 변수 이름에 한글을 허용하지 않는다. 한글이면
  #   명령어로 해석돼 127로 죽고, set -e 때문에 백업이 한 번도 안 만들어진다.
  FILE="/backups/bj-$(date +%Y%m%d-%H%M%S).sql.gz"
  pg_dump -h db -U bj -d bj | gzip > "$FILE"
  echo "backup: $FILE"
  find /backups -name 'bj-*.sql.gz' -mtime +7 -delete
  sleep 86400
done
