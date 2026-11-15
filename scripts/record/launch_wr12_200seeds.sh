#!/usr/bin/env bash
# Start the WR12 recorder independently of the interactive terminal.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source scripts/tools/activate_sim_env.sh

RUN_ID="${RUN_ID:-wr12_200seeds_v14}"
RENDER_GPUS="${RENDER_GPUS:-4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7}"
if [[ ! "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "RUN_ID contains invalid characters." >&2
    exit 2
fi
mkdir -p outputs/record
supervisor_log="outputs/record/${RUN_ID}_supervisor.log"
if [[ -e "$supervisor_log" || -e "outputs/record/$RUN_ID" ]]; then
    echo "Run already exists: $RUN_ID" >&2
    exit 1
fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
resume_args=()
if [[ -n "${RESUME_FROM:-}" ]]; then
    resume_args=(--resume-from "$RESUME_FROM")
fi
setsid nohup "$ROBOCASA_PYTHON" -u scripts/record/record_wr12_200seeds.py \
    --run-id "$RUN_ID" --render-gpus "$RENDER_GPUS" \
    "${resume_args[@]}" \
    > "$supervisor_log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "outputs/record/${RUN_ID}_supervisor.pid"
printf 'WR12 recorder started: PID=%s, seeds=0-199, GPUs=%s\n' "$pid" "$RENDER_GPUS"
printf 'Supervisor log: %s\n' "$supervisor_log"
printf 'Run directory: outputs/record/%s\n' "$RUN_ID"
