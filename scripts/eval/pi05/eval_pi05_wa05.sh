#!/usr/bin/env bash
# Evaluate one pi0.5 LoRA checkpoint on the formal 50-seed WA-05 suite.
#
#   GPU=2 PORT=55005 CKPT_STEP=59999 ./scripts/eval/pi05/eval_pi05_wa05.sh serve
#   PORT=55005 ./scripts/eval/pi05/eval_pi05_wa05.sh preflight
#   PORT=55005 ./scripts/eval/pi05/eval_pi05_wa05.sh eval
#   PORT=55005 ./scripts/eval/pi05/eval_pi05_wa05.sh summary
#   PORT=55005 ./scripts/eval/pi05/eval_pi05_wa05.sh stage-eval
#   ./scripts/eval/pi05/eval_pi05_wa05.sh stage-summary
#   PORT=55005 ./scripts/eval/pi05/eval_pi05_wa05.sh stop
#   GPU=2 PORT=55005 CKPT_STEP=59999 ./scripts/eval/pi05/eval_pi05_wa05.sh full-stage

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

CKPT_ROOT="${CKPT_ROOT:-/data/shared_workspace/chensihang/ckpts/membench/gr00t/wa05}"
CKPT_STEP="${CKPT_STEP:-59999}"
CKPT_DIR="${CKPT_DIR:-${CKPT_ROOT}/${CKPT_STEP}}"
CONFIG_NAME="${CONFIG_NAME:-pi05_membench_wa05_lora}"
PORT="${PORT:-55005}"
GPU="${GPU:-2}"
RENDER_GPU="${RENDER_GPU:-${GPU}}"
# Stage diagnostics retain the legacy policy-controlled first stage.
case "${1:-}" in
    *stage*|prefixes|prefix-summary)
        SUITE="${SUITE:-configs/eval/legacy_suites/wa05_pi05_r20_v1.yaml}"
        RUN_ID="${RUN_ID:-wa05-pi05-${CKPT_STEP}-50seeds-r20}"
        ;;
esac
SUITE="${SUITE:-configs/eval/legacy_suites/wa05_pi05_r20_prefix_v2.yaml}"
RUN_ID="${RUN_ID:-wa05-pi05-${CKPT_STEP}-50seeds-r20-prefix-v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_INFERENCE_TIMEOUT_S="${SERVER_INFERENCE_TIMEOUT_S:-300}"
RETRY_ERRORS="${RETRY_ERRORS:-0}"
STAGE_SUITE="${STAGE_SUITE:-configs/eval/stages/wa05_pi05_r20_formal_v1.yaml}"
PREFIX_ROOT="${PREFIX_ROOT:-outputs/stage_prefixes/wa05_formal_v1}"
STAGE_OUTPUT_ROOT="${STAGE_OUTPUT_ROOT:-outputs/stage_eval}"
STAGE_RUN_ID="${STAGE_RUN_ID:-wa05-pi05-${CKPT_STEP}-stage-50seeds-r20-formal-v1}"
STAGE_RECORD_VIDEO="${STAGE_RECORD_VIDEO:-0}"
MAIN_ROLLOUT_RUN="${MAIN_ROLLOUT_RUN:-${OUTPUT_ROOT}/${RUN_ID}}"
LOG_DIR="${OUTPUT_ROOT}/logs"

OPENPI_PYTHON="${ROOT}/policy/openpi/.venv/bin/python"
ROBOCASA_PYTHON="${ROOT}/.venv-robocasa/bin/python"
MPLIB_PYTHON="${ROOT}/.venv-mplib/bin/python"
MANISKILL_PANDA_DIR="${ROOT}/.venv-robocasa/lib/python3.11/site-packages/mani_skill/assets/robots/panda"
PID_FILE="${LOG_DIR}/serve_wa05_${PORT}.pid"
LOG_FILE="${LOG_DIR}/serve_wa05_${PORT}.log"

usage() {
    sed -n '2,10p' "${BASH_SOURCE[0]}" | sed -n 's/^# \{0,1\}//p'
}

validate_common() {
    [[ -x "${OPENPI_PYTHON}" ]] || { echo "OpenPI environment is missing: ${OPENPI_PYTHON}" >&2; exit 1; }
    [[ -x "${ROBOCASA_PYTHON}" ]] || { echo "RoboCasa environment is missing: ${ROBOCASA_PYTHON}" >&2; exit 1; }
    [[ -f "${ROOT}/${SUITE}" || -f "${SUITE}" ]] || { echo "Evaluation suite is missing: ${SUITE}" >&2; exit 1; }
}

start_server() {
    validate_common
    [[ -d "${CKPT_DIR}/params" ]] || { echo "Checkpoint params are missing: ${CKPT_DIR}/params" >&2; exit 1; }
    [[ -f "${CKPT_DIR}/assets/wa05_200seeds_v061/norm_stats.json" ]] || {
        echo "Checkpoint normalization stats are missing: ${CKPT_DIR}" >&2
        exit 1
    }
    mkdir -p "${LOG_DIR}"
    if [[ -f "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
        echo "Server ${PORT} is already running (pid $(<"${PID_FILE}"))"
        return 0
    fi
    cd "${ROOT}"
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
    CUDA_VISIBLE_DEVICES="${GPU}" \
        nohup "${OPENPI_PYTHON}" scripts/serve/serve_openpi.py \
            --config "${CONFIG_NAME}" \
            --checkpoint "${CKPT_DIR}" \
            --image-size 224 \
            --state-schema membench.robot_state37.v2 \
            --inference-timeout-s "${SERVER_INFERENCE_TIMEOUT_S}" \
            --host 0.0.0.0 \
            --port "${PORT}" \
            >"${LOG_FILE}" 2>&1 &
    echo $! >"${PID_FILE}"
    echo "Started server ${PORT} (pid $!, GPU ${GPU}, checkpoint ${CKPT_STEP})"
    echo "Log: ${LOG_FILE}"
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

run_preflight() {
    validate_common
    cd "${ROOT}"
    export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
    export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
    "${ROBOCASA_PYTHON}" scripts/eval/run_remote.py \
        --endpoint "ws://${SERVER_HOST}:${PORT}" \
        --suite "${SUITE}" \
        --preflight-only
}

run_eval() {
    local retry_args=()
    validate_common
    [[ -x "${MPLIB_PYTHON}" ]] || { echo "MPlib environment is missing: ${MPLIB_PYTHON}" >&2; exit 1; }
    [[ -d "${MANISKILL_PANDA_DIR}" ]] || { echo "ManiSkill Panda assets are missing: ${MANISKILL_PANDA_DIR}" >&2; exit 1; }
    [[ "${RETRY_ERRORS}" == "1" ]] && retry_args=(--retry-errors)
    cd "${ROOT}"
    export PYTHONPATH=.
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    export MUJOCO_EGL_DEVICE_ID="${RENDER_GPU}"
    export MEMBENCH_MPLIB_PYTHON="${MEMBENCH_MPLIB_PYTHON:-${MPLIB_PYTHON}}"
    export MEMBENCH_MANISKILL_PANDA_DIR="${MEMBENCH_MANISKILL_PANDA_DIR:-${MANISKILL_PANDA_DIR}}"
    export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
    export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
    "${ROBOCASA_PYTHON}" scripts/eval/run_remote.py \
        --endpoint "ws://${SERVER_HOST}:${PORT}" \
        --suite "${SUITE}" \
        --output-root "${OUTPUT_ROOT}" \
        --run-id "${RUN_ID}" \
        "${retry_args[@]}"
}

run_summary() {
    validate_common
    cd "${ROOT}"
    "${ROBOCASA_PYTHON}" scripts/eval/summarize_run.py "${OUTPUT_ROOT}/${RUN_ID}"
}

run_stage_eval() {
    local video_args=()

    validate_common
    [[ -f "${ROOT}/${STAGE_SUITE}" || -f "${STAGE_SUITE}" ]] || {
        echo "Stage suite is missing: ${STAGE_SUITE}" >&2
        exit 1
    }
    [[ -d "${PREFIX_ROOT}" ]] || {
        echo "Expert prefix root is missing: ${PREFIX_ROOT}" >&2
        exit 1
    }
    [[ -f "${MAIN_ROLLOUT_RUN}/summary.json" ]] || {
        echo "Completed main rollout is required: ${MAIN_ROLLOUT_RUN}" >&2
        exit 1
    }
    if [[ "${STAGE_RECORD_VIDEO}" == "1" ]]; then
        video_args=(--record-video)
    elif [[ "${STAGE_RECORD_VIDEO}" == "0" ]]; then
        video_args=(--no-record-video)
    else
        echo "STAGE_RECORD_VIDEO must be 0 or 1" >&2
        exit 2
    fi
    cd "${ROOT}"
    export PYTHONPATH=.
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    export MUJOCO_EGL_DEVICE_ID="${RENDER_GPU}"
    export MEMBENCH_MPLIB_PYTHON="${MEMBENCH_MPLIB_PYTHON:-${MPLIB_PYTHON}}"
    export MEMBENCH_MANISKILL_PANDA_DIR="${MEMBENCH_MANISKILL_PANDA_DIR:-${MANISKILL_PANDA_DIR}}"
    export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
    export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
    "${ROBOCASA_PYTHON}" scripts/eval/run_stage_remote.py \
        --endpoint "ws://${SERVER_HOST}:${PORT}" \
        --stage-suite "${STAGE_SUITE}" \
        --prefix-root "${PREFIX_ROOT}" \
        --main-rollout-run "${MAIN_ROLLOUT_RUN}" \
        --output-root "${STAGE_OUTPUT_ROOT}" \
        --run-id "${STAGE_RUN_ID}" \
        "${video_args[@]}"
}

run_stage_summary() {
    validate_common
    cd "${ROOT}"
    "${ROBOCASA_PYTHON}" scripts/eval/summarize_stage_run.py \
        "${STAGE_OUTPUT_ROOT}/${STAGE_RUN_ID}"
}

status() {
    if [[ -f "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
        echo "Server ${PORT}: running (pid $(<"${PID_FILE}"), checkpoint ${CKPT_STEP})"
    else
        echo "Server ${PORT}: not running"
    fi
}

wait_for_preflight() {
    local attempt
    for attempt in {1..10}; do
        if run_preflight; then
            return
        fi
        if [[ ! -f "${PID_FILE}" ]] || ! kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
            echo "Policy server exited before preflight completed; see ${LOG_FILE}" >&2
            return 1
        fi
        echo "Policy server is still loading (preflight attempt ${attempt}/10)"
        sleep 5
    done
    echo "Policy server did not become ready; see ${LOG_FILE}" >&2
    return 1
}

run_full() {
    start_server
    trap stop_server EXIT
    wait_for_preflight
    run_eval
}

run_full_stage() {
    start_server
    trap stop_server EXIT
    wait_for_preflight
    run_eval
    run_stage_eval
}

case "${1:-}" in
    serve) start_server ;;
    preflight) run_preflight ;;
    eval) run_eval ;;
    summary) run_summary ;;
    stage-eval) run_stage_eval ;;
    stage-summary) run_stage_summary ;;
    stop) stop_server ;;
    status) status ;;
    full) run_full ;;
    full-stage) run_full_stage ;;
    help|-h|--help) usage ;;
    *)
        echo "Unknown command: ${1:-}" >&2
        usage >&2
        exit 2
        ;;
esac
