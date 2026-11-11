#!/usr/bin/env bash
# Evaluate the WA-05 three-view pi0.5 checkpoint with 40-action replanning.
#
#   ./scripts/eval/pi05/eval_pi05_wa05_3view.sh serve-fg
#   ./scripts/eval/pi05/eval_pi05_wa05_3view.sh preflight
#   ./scripts/eval/pi05/eval_pi05_wa05_3view.sh smoke
#   ./scripts/eval/pi05/eval_pi05_wa05_3view.sh smoke-summary
#   ./scripts/eval/pi05/eval_pi05_wa05_3view.sh eval
#   ./scripts/eval/pi05/eval_pi05_wa05_3view.sh summary
#   ./scripts/eval/pi05/eval_pi05_wa05_3view.sh status
#   ./scripts/eval/pi05/eval_pi05_wa05_3view.sh stop

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

CKPT_DIR="${CKPT_DIR:-${ROOT}/outputs/pi05-ckpts/wa05/20000}"
CONFIG_NAME="${CONFIG_NAME:-pi05_membench_wa05_3view_full_stride10_10k}"
PORT="${PORT:-59205}"
GPU="${GPU:-4}"
RENDER_GPU="${RENDER_GPU:-5}"
SUITE="${SUITE:-configs/eval/legacy_suites/wa05_pi05_r40_t2000_prefix_v2.yaml}"
SMOKE_SUITE="${SMOKE_SUITE:-configs/eval/smoke/wa05_pi05_r40_t2000_seed1049_prefix_v2.yaml}"
CKPT_STEP="$(basename "${CKPT_DIR}")"
RUN_ID="${RUN_ID:-wa05-pi05-3view-full-stride10-s${CKPT_STEP}-r40-t2000-seedrng-prefix-v2}"
SMOKE_RUN_ID="${SMOKE_RUN_ID:-${RUN_ID}-smoke-1049}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_INFERENCE_TIMEOUT_S="${SERVER_INFERENCE_TIMEOUT_S:-300}"
RETRY_ERRORS="${RETRY_ERRORS:-1}"
SEED_BY_EPISODE="${SEED_BY_EPISODE:-1}"
LOG_DIR="${OUTPUT_ROOT}/logs"
PID_FILE="${LOG_DIR}/serve_wa05_3view_${PORT}.pid"
LOG_FILE="${LOG_DIR}/serve_wa05_3view_${PORT}.log"

DEFAULT_OPENPI_PYTHON="${ROOT}/policy/openpi/.venv/bin/python"
if [[ ! -x "${DEFAULT_OPENPI_PYTHON}" && -x "${HOME}/openpi/.venv/bin/python" ]]; then
    DEFAULT_OPENPI_PYTHON="${HOME}/openpi/.venv/bin/python"
fi
OPENPI_PYTHON="${OPENPI_PYTHON:-${DEFAULT_OPENPI_PYTHON}}"

usage() {
    sed -n '2,11p' "${BASH_SOURCE[0]}" | sed -n 's/^# \{0,1\}//p'
}

validate_common() {
    [[ -x "${OPENPI_PYTHON}" ]] || { echo "OpenPI environment is missing: ${OPENPI_PYTHON}" >&2; exit 1; }
    [[ -x "${ROOT}/.venv-robocasa/bin/python" ]] || { echo "RoboCasa environment is missing" >&2; exit 1; }
    [[ -d "${CKPT_DIR}/params" ]] || { echo "Checkpoint params are missing: ${CKPT_DIR}/params" >&2; exit 1; }
    [[ -f "${CKPT_DIR}/assets/wa05-final/norm_stats.json" ]] || {
        echo "Checkpoint normalization stats are missing: ${CKPT_DIR}/assets/wa05-final/norm_stats.json" >&2
        exit 1
    }
    [[ -f "${ROOT}/${SUITE}" || -f "${SUITE}" ]] || { echo "Evaluation suite is missing: ${SUITE}" >&2; exit 1; }
    [[ -f "${ROOT}/${SMOKE_SUITE}" || -f "${SMOKE_SUITE}" ]] || { echo "Smoke suite is missing: ${SMOKE_SUITE}" >&2; exit 1; }
}

server_args() {
    SERVER_ARGS=(
        scripts/serve/serve_openpi.py
        --config "${CONFIG_NAME}"
        --checkpoint "${CKPT_DIR}"
        --image-size 224
        --state-schema membench.robot_state37.v2
        --inference-timeout-s "${SERVER_INFERENCE_TIMEOUT_S}"
        --host 0.0.0.0
        --port "${PORT}"
        --pytorch-device cuda:0
    )
    [[ "${SEED_BY_EPISODE}" == "1" ]] && SERVER_ARGS+=(--seed-by-episode)
}

start_server() {
    validate_common
    mkdir -p "${LOG_DIR}"
    if [[ -f "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
        echo "Server ${PORT} is already running (pid $(<"${PID_FILE}"))"
        return 0
    fi
    server_args
    cd "${ROOT}"
    export OPENPI_PRETRAIN_MODEL_DIR="${OPENPI_PRETRAIN_MODEL_DIR:-${HOME}/openpi/pretrained}"
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
    CUDA_VISIBLE_DEVICES="${GPU}" nohup "${OPENPI_PYTHON}" "${SERVER_ARGS[@]}" \
        >"${LOG_FILE}" 2>&1 </dev/null &
    echo $! >"${PID_FILE}"
    echo "Started server ${PORT} (pid $!, GPU ${GPU}, checkpoint ${CKPT_STEP})"
    echo "Log: ${LOG_FILE}"
}

start_server_fg() {
    validate_common
    mkdir -p "${LOG_DIR}"
    echo "$$" >"${PID_FILE}"
    server_args
    cd "${ROOT}"
    export OPENPI_PRETRAIN_MODEL_DIR="${OPENPI_PRETRAIN_MODEL_DIR:-${HOME}/openpi/pretrained}"
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
    export CUDA_VISIBLE_DEVICES="${GPU}"
    exec "${OPENPI_PYTHON}" "${SERVER_ARGS[@]}"
}

activate_sim() {
    export MEMBENCH_RENDER_GPU="${RENDER_GPU}"
    # shellcheck disable=SC1091
    source "${ROOT}/scripts/tools/activate_sim_env.sh"
}

run_preflight() {
    validate_common
    cd "${ROOT}"
    activate_sim
    "${ROBOCASA_PYTHON}" scripts/eval/run_remote.py \
        --endpoint "ws://${SERVER_HOST}:${PORT}" \
        --suite "${SUITE}" \
        --preflight-only
}

run_eval() {
    local suite="$1"
    local run_id="$2"
    local retry_args=()
    validate_common
    [[ "${RETRY_ERRORS}" == "1" ]] && retry_args=(--retry-errors)
    cd "${ROOT}"
    activate_sim
    "${ROBOCASA_PYTHON}" scripts/eval/run_remote.py \
        --endpoint "ws://${SERVER_HOST}:${PORT}" \
        --suite "${suite}" \
        --output-root "${OUTPUT_ROOT}" \
        --run-id "${run_id}" \
        "${retry_args[@]}"
}

run_summary() {
    local run_id="$1"
    cd "${ROOT}"
    activate_sim
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
    smoke) run_eval "${SMOKE_SUITE}" "${SMOKE_RUN_ID}" ;;
    smoke-summary) run_summary "${SMOKE_RUN_ID}" ;;
    eval) run_eval "${SUITE}" "${RUN_ID}" ;;
    summary) run_summary "${RUN_ID}" ;;
    status) status ;;
    stop) stop_server ;;
    help|-h|--help) usage ;;
    *) usage >&2; exit 2 ;;
esac
