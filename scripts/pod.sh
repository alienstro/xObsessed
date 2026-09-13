#!/usr/bin/env bash
# Operate one pod from this repository.
set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE_DIR="/workspace/xObsessed"
ENV_FILE="${ENV_FILE:-.env}"
UV="${UV:-/root/.local/bin/uv}"
if [ -f "$ENV_FILE" ]; then
    set -a
    # Read only a trusted local environment file.
    source "$ENV_FILE"
    set +a
fi

case "${1:-}" in
verify|pods|terminate)
    COMMAND="$1"
    shift
    case "$COMMAND" in
    verify) exec uv run --env-file "$ENV_FILE" python scripts/verify_upload.py "$@" ;;
    pods) exec uv run --env-file "$ENV_FILE" python scripts/runpod.py list "$@" ;;
    terminate)
        # Warning: this command deletes the pod disk after upload verification.
        exec uv run --env-file "$ENV_FILE" python scripts/runpod.py terminate "$@"
        ;;
    esac
    ;;
esac

POD_SSH="${POD_SSH:-}"
POD_SSH="${POD_SSH#ssh }"
if [ -z "$POD_SSH" ]; then
    echo "POD_SSH is empty. Set it in .env or export it." >&2
    exit 1
fi
read -r -a CONNECTION <<< "$POD_SSH"
HOST="${CONNECTION[0]}"
FLAGS=(-o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o StrictHostKeyChecking=yes)
EXTRA=("${CONNECTION[@]:1}")
remote() {
    ssh "${FLAGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"} "$HOST" "$@"
}
printf -v TRANSPORT '%q ' ssh "${FLAGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"}

case "${1:-}" in
check)
    remote "nvidia-smi && command -v tmux && command -v rsync && command -v cmake"
    ;;
sync)
    remote "mkdir -p '$REMOTE_DIR'"
    rsync -az --exclude '.git' --exclude '.venv' --exclude 'data' --exclude 'out' \
        --exclude 'llama.cpp' --exclude '__pycache__' --exclude '.env*' \
        -e "$TRANSPORT" ./ "$HOST:$REMOTE_DIR/"
    ;;
data)
    test -f data/raw.jsonl
    remote "mkdir -p '$REMOTE_DIR/data'"
    rsync -az -e "$TRANSPORT" data/raw.jsonl "$HOST:$REMOTE_DIR/data/"
    ;;
secrets)
    test -f "$ENV_FILE"
    remote "umask 077; mkdir -p '$REMOTE_DIR'; cat > '$REMOTE_DIR/.env'; chmod 600 '$REMOTE_DIR/.env'" < "$ENV_FILE"
    ;;
setup)
    remote "cd '$REMOTE_DIR' && '$UV' sync --locked && '$UV' run pytest -q"
    ;;
run)
    shift
    if [ "$#" -eq 0 ]; then
        echo "Supply a command." >&2
        exit 1
    fi
    if [[ "$1" == *.py ]]; then
        set -- python "$@"
    fi
    printf -v COMMAND '%q ' "$@"
    printf -v UV_COMMAND '%q ' "$UV" run --env-file .env
    printf -v JOB '%q ' bash -o pipefail -c "${UV_COMMAND}${COMMAND}2>&1 | tee out/job.log"
    printf -v START '%q ' tmux new-session -d -s xobsessed-job "$JOB"
    # Use one job session. A second command cannot replace an active job.
    remote "cd '$REMOTE_DIR' && mkdir -p out && $START"
    ;;
watch)
    remote "touch '$REMOTE_DIR/out/HEARTBEAT'; tr '\\r' '\\n' < '$REMOTE_DIR/out/job.log' | tail -n 40"
    ;;
heartbeat)
    remote "mkdir -p '$REMOTE_DIR/out'; touch '$REMOTE_DIR/out/HEARTBEAT'"
    ;;
sessions)
    remote "tmux list-sessions"
    ;;
manifest)
    mkdir -p out
    rsync -az -e "$TRANSPORT" "$HOST:$REMOTE_DIR/out/upload-manifest.json" out/
    ;;
*)
    echo "Usage: pod.sh {check|sync|data|secrets|setup|run|watch|heartbeat|sessions|manifest|verify|pods|terminate}" >&2
    exit 1
    ;;
esac
