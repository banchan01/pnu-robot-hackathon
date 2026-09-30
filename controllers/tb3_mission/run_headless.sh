#!/bin/bash
# 헤드리스 자동 테스트: Webots를 fast 모드로 띄워 컨트롤러 로그를 파일로 남긴다.
# 사용법:  ./run_headless.sh [world 파일] [실행 시간(초, 실시간)] [로그 파일]
#   예)    ./run_headless.sh ../../worlds/apartment_debug.wbt 300 /tmp/tb3.log
# 환경 변수:
#   TB3_DEBUG_GT=1     Supervisor가 켜진 월드에서 ground truth 오차를 함께 기록
#   TB3_SAVE_EVERY=30  30초(시뮬레이션 시간)마다 output/에 지도 스냅샷 저장
#   TB3_TIME_LIMIT=600 제한 시간 덮어쓰기
set -u
WORLD=${1:-../../worlds/apartment_mission.wbt}
SECS=${2:-300}
LOG=${3:-/tmp/tb3_mission.log}
WEBOTS=${WEBOTS_BIN:-/Applications/Webots.app/Contents/MacOS/webots}

# 다른 사람이 띄운 Webots를 죽이지 않도록 이 스크립트가 띄운 인스턴스만 종료한다.
OUT=${TB3_OUTPUT_DIR:-/tmp/tb3_out_$$}
mkdir -p "$OUT"
TB3_HEADLESS=1 TB3_OUTPUT_DIR="$OUT" "$WEBOTS" --batch --mode=fast --no-rendering --stdout --stderr --minimize "$WORLD" > "$LOG" 2>&1 &
WPID=$!
echo "webots pid $WPID, output dir $OUT"
sleep "$SECS"
kill "$WPID" 2>/dev/null; sleep 2; kill -9 "$WPID" 2>/dev/null
echo "stopped after ${SECS}s" >> "$LOG"
echo "=== 컨트롤러 로그 (마지막 40줄) ==="
grep -E "^\[" "$LOG" | tail -40
echo "=== 오류 ==="
grep -nE "Traceback|Error" "$LOG" | head
