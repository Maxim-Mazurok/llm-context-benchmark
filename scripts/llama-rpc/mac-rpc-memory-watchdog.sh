#!/bin/sh
set -eu

pid_file=${1:-"$HOME/qwen-mac-rpc-50053.pid"}
log_file=${2:-"$HOME/qwen-mac-rpc-memory-watchdog.log"}
minimum_free_percent=${3:-10}
maximum_swap_mb=${4:-6144}
breaches=0

while :; do
    pid=$(cat "$pid_file" 2>/dev/null || true)
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        printf '%s rpc_exited\n' "$(date -u +%FT%TZ)" >> "$log_file"
        exit 0
    fi

    pressure=$(memory_pressure -Q 2>/dev/null || true)
    free_percent=$(printf '%s\n' "$pressure" | awk '/free percentage:/ {gsub(/%/, "", $NF); print int($NF)}')
    swap_line=$(sysctl vm.swapusage 2>/dev/null || true)
    swap_mb=$(printf '%s\n' "$swap_line" | awk '{for(i=1;i<=NF;i++) if($i=="used") {v=$(i+2); sub(/M$/, "", v); print int(v)}}')
    free_percent=${free_percent:-0}
    swap_mb=${swap_mb:-0}
    printf '%s pid=%s free_percent=%s swap_mb=%s\n' "$(date -u +%FT%TZ)" "$pid" "$free_percent" "$swap_mb" >> "$log_file"

    if [ "$free_percent" -lt "$minimum_free_percent" ] || [ "$swap_mb" -gt "$maximum_swap_mb" ]; then
        breaches=$((breaches + 1))
    else
        breaches=0
    fi
    if [ "$breaches" -ge 2 ]; then
        printf '%s terminating_rpc_for_memory_safety\n' "$(date -u +%FT%TZ)" >> "$log_file"
        kill "$pid" 2>/dev/null || true
        exit 0
    fi
    sleep 1
done
