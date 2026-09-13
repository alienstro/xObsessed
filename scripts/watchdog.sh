#!/usr/bin/env bash
# Stop the pod at the deadline. Never delete its disk automatically.
set -euo pipefail
cd "$(dirname "$0")/.."
UV="${UV:-/root/.local/bin/uv}"
DEADLINE_MINUTES="${1:-150}"
HEARTBEAT_GRACE="${2:-15}"
[[ "$DEADLINE_MINUTES" =~ ^[1-9][0-9]*$ ]]
[[ "$HEARTBEAT_GRACE" =~ ^[1-9][0-9]*$ ]]
STARTED_AT=$(date +%s)
mkdir -p out
touch out/HEARTBEAT
echo "The watchdog will stop the pod after $DEADLINE_MINUTES minutes."
while true; do
    NOW=$(date +%s)
    MODIFIED=$(stat -c %Y out/HEARTBEAT)
    if (( NOW - STARTED_AT >= DEADLINE_MINUTES * 60 || NOW - MODIFIED >= HEARTBEAT_GRACE * 60 )); then
        echo "The deadline or heartbeat limit expired. Request a pod stop."
        "$UV" run --env-file .env python scripts/runpod.py stop
        exit 0
    fi
    sleep 60
done
