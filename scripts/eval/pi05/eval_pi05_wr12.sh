#!/usr/bin/env bash
# Evaluate the WR12 PI0.5 checkpoint. Override GPU, RENDER_GPU, PORT, or RUN_ID.
# Run serve, wait for "server listening" in its log, then preflight and smoke.
# Use eval for the formal 50-seed suite after reviewing the smoke artifacts.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

CKPT_DIR="${CKPT_DIR:-${ROOT}/outputs/pi05-ckpts/wr12/29999}"
CONFIG_NAME="${CONFIG_NAME:-pi05_membench_wr12_3view_full_stride10_30k}"
PORT="${PORT:-59212}"
GPU="${GPU:-0}"
RENDER_GPU="${RENDER_GPU:-1}"
SUITE="${SUITE:-configs/eval/suites/wr12_v1.yaml}"
SMOKE_SUITE="${SMOKE_SUITE:-configs/eval/smoke/wr12_pi05_r50_seed1049_v1.yaml}"
CKPT_STEP="$(basename "${CKPT_DIR}")"
RUN_ID="${RUN_ID:-wr12-pi05-3view-full-stride10-s${CKPT_STEP}-r50-seedrng-v1}"
SMOKE_RUN_ID="${SMOKE_RUN_ID:-${RUN_ID}-smoke-1049}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
RETRY_ERRORS="${RETRY_ERRORS:-0}"
SEED_BY_EPISODE="${SEED_BY_EPISODE:-1}"
LOG_DIR="${OUTPUT_ROOT}/logs"
PID_FILE="${LOG_DIR}/serve_wr12_${PORT}.pid"
LOG_FILE="${LOG_DIR}/serve_wr12_${PORT}.log"

DEFAULT_OPENPI_PYTHON="${ROOT}/policy/openpi/.venv/bin/python"
if [[ ! -x "${DEFAULT_OPENPI_PYTHON}" && -x "${HOME}/openpi/.venv/bin/python" ]]; then
    DEFAULT_OPENPI_PYTHON="${HOME}/openpi/.venv/bin/python"
fi
OPENPI_PYTHON="${OPENPI_PYTHON:-${DEFAULT_OPENPI_PYTHON}}"

usage() {
    echo "Usage: $0 {serve|serve-fg|preflight|smoke|eval|smoke-summary|summary|status|stop}"
}

validate_common() {
    [[ -x "${OPENPI_PYTHON}" ]] || { echo "OpenPI environment is missing: ${OPENPI_PYTHON}" >&2; exit 1; }
    [[ -x "${ROOT}/.venv-robocasa/bin/python" ]] || { echo "RoboCasa environment is missing" >&2; exit 1; }
    [[ -d "${CKPT_DIR}/params" ]] || { echo "Checkpoint params are missing: ${CKPT_DIR}/params" >&2; exit 1; }
    [[ -f "${CKPT_DIR}/assets/wr12-final/norm_stats.json" ]] || {
        echo "Checkpoint normalization stats are missing: ${CKPT_DIR}/assets/wr12-final/norm_stats.json" >&2
        exit 1
    }
}

start_server() {
    local seed_args=()
    validate_common
    cd "${ROOT}"
    mkdir -p "${LOG_DIR}"
    if [[ -f "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
        echo "Server ${PORT} is already running (pid $(<"${PID_FILE}"))"
        return 0
    fi
    if [[ "${SEED_BY_EPISODE}" == "1" ]]; then
        seed_args=(--seed-by-episode)
    fi
    cd "${ROOT}"
    "${OPENPI_PYTHON}" -c 'import socket, sys; s=socket.socket(); s.bind(("127.0.0.1", int(sys.argv[1]))); s.close()' "${PORT}"
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
    CUDA_VISIBLE_DEVICES="${GPU}" nohup "${OPENPI_PYTHON}" scripts/serve/serve_openpi.py \
        --config "${CONFIG_NAME}" \
        --checkpoint "${CKPT_DIR}" \
        --image-size 224 \
        --state-schema membench.robot_state37.v2 \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --pytorch-device cuda:0 \
        "${seed_args[@]}" \
        >"${LOG_FILE}" 2>&1 </dev/null &
    echo $! >"${PID_FILE}"
    echo "Started server ${PORT} (pid $!, GPU ${GPU}, checkpoint ${CKPT_STEP})"
    echo "Log: ${LOG_FILE}"
}

start_server_fg() {
    local seed_args=()
    validate_common
    cd "${ROOT}"
    mkdir -p "${LOG_DIR}"
    echo "$$" >"${PID_FILE}"
    if [[ "${SEED_BY_EPISODE}" == "1" ]]; then
        seed_args=(--seed-by-episode)
    fi
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
    export CUDA_VISIBLE_DEVICES="${GPU}"
    exec "${OPENPI_PYTHON}" scripts/serve/serve_openpi.py \
        --config "${CONFIG_NAME}" \
        --checkpoint "${CKPT_DIR}" \
        --image-size 224 \
        --state-schema membench.robot_state37.v2 \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --pytorch-device cuda:0 \
        "${seed_args[@]}"
}

run_remote() {
    local suite="$1"
    local run_id="$2"
    local retry_args=()
    validate_common
    cd "${ROOT}"
    export MEMBENCH_RENDER_GPU="${RENDER_GPU}"
    source scripts/tools/activate_sim_env.sh
    if [[ "${RETRY_ERRORS}" == "1" ]]; then
        retry_args=(--retry-errors)
    fi
    "${ROBOCASA_PYTHON}" scripts/eval/run_remote.py \
        --endpoint "ws://${SERVER_HOST}:${PORT}" \
        --suite "${suite}" \
        --output-root "${OUTPUT_ROOT}" \
        --run-id "${run_id}" \
        "${retry_args[@]}"
}

run_preflight() {
    validate_common
    cd "${ROOT}"
    export MEMBENCH_RENDER_GPU="${RENDER_GPU}"
    source scripts/tools/activate_sim_env.sh
    "${ROBOCASA_PYTHON}" scripts/eval/run_remote.py \
        --endpoint "ws://${SERVER_HOST}:${PORT}" \
        --suite "${SUITE}" \
        --preflight-only
}

run_summary() {
    local run_id="$1"
    cd "${ROOT}"
    export MEMBENCH_RENDER_GPU="${RENDER_GPU}"
    source scripts/tools/activate_sim_env.sh
    "${ROBOCASA_PYTHON}" scripts/eval/summarize_run.py "${OUTPUT_ROOT}/${run_id}"
}

status() {
    if [[ -f "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
        echo "Server ${PORT}: running (pid $(<"${PID_FILE}"), checkpoint ${CKPT_STEP})"
    else
        echo "Server ${PORT}: not running"
    fi
}

stop_server() {
    if [[ -f "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
        kill "$(<"${PID_FILE}")"
        rm -f "${PID_FILE}"
        echo "Stopped server ${PORT}"
    else
        rm -f "${PID_FILE}" 2>/dev/null || true
        echo "Server ${PORT} is not running"
    fi
}

case "${1:-}" in
    serve) start_server ;;
    serve-fg) start_server_fg ;;
    preflight) run_preflight ;;
    smoke) run_remote "${SMOKE_SUITE}" "${SMOKE_RUN_ID}" ;;
    eval) run_remote "${SUITE}" "${RUN_ID}" ;;
    smoke-summary) run_summary "${SMOKE_RUN_ID}" ;;
    summary) run_summary "${RUN_ID}" ;;
    status) status ;;
    stop) stop_server ;;
    help|-h|--help) usage ;;
    *) usage >&2; exit 2 ;;
esac
