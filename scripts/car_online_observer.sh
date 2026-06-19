#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/cyk/codes/OpenAvatarChat}"
cd "$REPO_DIR"

SHA="$(git rev-parse --short HEAD)"
OUT_DIR="${OUT_DIR:-tests/results/$SHA}"
mkdir -p "$OUT_DIR"

usage() {
  cat <<'EOF'
Usage: car_online_observer.sh <command>

Commands:
  health             Save service/env/network/power snapshot
  start-logs         Start background journald/resource capture
  stop-logs          Stop captures started by start-logs
  status             Show capture PIDs, output files, and latest resource sample
  tail [N]           Show latest N lines from server/client/resource logs
  vad-row ARGS...    Append one row to vad_trials.md
  tracking-row ARGS  Append one row to tracking_trials.md

Examples:
  bash /tmp/car_online_observer.sh health
  bash /tmp/car_online_observer.sh start-logs
  bash /tmp/car_online_observer.sh tail 80
  bash /tmp/car_online_observer.sh vad-row 1 success normal "responded after 1s"
  bash /tmp/car_online_observer.sh tracking-row "0-20s" "lost/reacquire" 3 "7s" "TRACKING-IDLE" "static marker"
EOF
}

append_header_if_missing() {
  local file="$1"
  local header="$2"
  if [[ ! -s "$file" ]]; then
    printf '%s\n' "$header" > "$file"
  fi
}

run_health() {
  {
    echo "=== health $(date -Is) ==="
    echo "repo=$REPO_DIR"
    echo "sha=$SHA"
    git status --short --branch
    git log --oneline -n 5
    echo
    echo "=== services ==="
    systemctl --user status openavatarchat openavatarchat-client --no-pager || true
    echo
    echo "=== recent server log ==="
    journalctl --user -u openavatarchat -n 80 --no-pager || true
    echo
    echo "=== recent client log ==="
    journalctl --user -u openavatarchat-client -n 120 --no-pager || true
    echo
    echo "=== env/network/power ==="
    if grep -q '^DASHSCOPE_API_KEY=.' .env 2>/dev/null; then
      echo "KEY set in .env: True"
    else
      echo "KEY set in .env: False"
    fi
    .venv/bin/python -c 'import speexdsp; print("speexdsp OK:", speexdsp.__file__)' || true
    ping -c1 dashscope.aliyuncs.com || true
    vcgencmd measure_temp || true
    vcgencmd get_throttled || true
    echo
    echo "=== systemd exec ==="
    systemctl --user show openavatarchat -p ExecStart --no-pager || true
    systemctl --user show openavatarchat-client -p ExecStart --no-pager || true
  } | tee "$OUT_DIR/online_health.txt"
}

start_one() {
  local name="$1"
  local cmd="$2"
  local pid_file="$OUT_DIR/$name.pid"
  local log_file="$OUT_DIR/$name.log"

  if [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "$name already running: pid=$(cat "$pid_file")"
    return
  fi

  nohup bash -lc "$cmd" > "$log_file" 2>&1 &
  echo $! > "$pid_file"
  echo "$name started: pid=$(cat "$pid_file") log=$log_file"
}

start_logs() {
  start_one "server-live" "journalctl --user -fu openavatarchat --no-pager"
  start_one "client-live" "journalctl --user -fu openavatarchat-client --no-pager"
  start_one "resource-live" \
    "while true; do date -Is; vcgencmd measure_temp 2>/dev/null || true; vcgencmd get_throttled 2>/dev/null || true; ps -eo pid,psr,pcpu,pmem,comm,args --sort=-pcpu | head -20; sleep 2; done"
}

stop_logs() {
  for name in server-live client-live resource-live; do
    local pid_file="$OUT_DIR/$name.pid"
    if [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
      kill "$(cat "$pid_file")"
      echo "$name stopped: pid=$(cat "$pid_file")"
    else
      echo "$name not running"
    fi
  done
}

show_status() {
  echo "repo=$REPO_DIR"
  echo "sha=$SHA"
  echo "out=$OUT_DIR"
  for name in server-live client-live resource-live; do
    local pid_file="$OUT_DIR/$name.pid"
    if [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
      echo "$name: running pid=$(cat "$pid_file")"
    else
      echo "$name: not running"
    fi
  done
  ls -lh "$OUT_DIR"/*live.log "$OUT_DIR"/online_health.txt 2>/dev/null || true
  echo
  echo "latest resource sample:"
  tail -30 "$OUT_DIR/resource-live.log" 2>/dev/null || true
}

tail_logs() {
  local n="${1:-80}"
  echo "=== server-live tail ==="
  tail -n "$n" "$OUT_DIR/server-live.log" 2>/dev/null || true
  echo
  echo "=== client-live tail ==="
  tail -n "$n" "$OUT_DIR/client-live.log" 2>/dev/null || true
  echo
  echo "=== resource-live tail ==="
  tail -n "$n" "$OUT_DIR/resource-live.log" 2>/dev/null || true
}

vad_row() {
  local file="$OUT_DIR/vad_trials.md"
  append_header_if_missing "$file" '| trial | time | wake reply heard | user phrase accepted | delay | notes |
|---|---|---|---|---|---|'
  local trial="${1:-}"
  local accepted="${2:-}"
  local delay="${3:-}"
  local notes="${4:-}"
  printf '| %s | %s | %s | %s | %s | %s |\n' \
    "$trial" "$(date -Is)" "manual" "$accepted" "$delay" "$notes" >> "$file"
  tail -5 "$file"
}

tracking_row() {
  local file="$OUT_DIR/tracking_trials.md"
  append_header_if_missing "$file" '| window | tag visible pattern | Tag lost count | longest apparent loss | tracking state pattern | notes |
|---|---|---:|---:|---|---|'
  printf '| %s | %s | %s | %s | %s | %s |\n' \
    "${1:-}" "${2:-}" "${3:-}" "${4:-}" "${5:-}" "${6:-}" >> "$file"
  tail -5 "$file"
}

cmd="${1:-}"
shift || true
case "$cmd" in
  health) run_health ;;
  start-logs) start_logs ;;
  stop-logs) stop_logs ;;
  status) show_status ;;
  tail) tail_logs "${1:-80}" ;;
  vad-row) vad_row "$@" ;;
  tracking-row) tracking_row "$@" ;;
  ""|-h|--help|help) usage ;;
  *) echo "Unknown command: $cmd" >&2; usage >&2; exit 2 ;;
esac
