#!/usr/bin/env bash
# Evaluate one GR00T N1.7 LoRA checkpoint on the 50-seed WA-05 suite.
#
#   GPU=3 PORT=57020 CKPT_STEP=070000 ./scripts/eval/gr00t/eval_gr00t_wa05.sh start
#   PORT=57020 CKPT_STEP=070000 ./scripts/eval/gr00t/eval_gr00t_wa05.sh preflight
#   PORT=57020 CKPT_STEP=070000 ./scripts/eval/gr00t/eval_gr00t_wa05.sh eval
#   CKPT_STEP=070000 ./scripts/eval/gr00t/eval_gr00t_wa05.sh summary
#   ./scripts/eval/gr00t/eval_gr00t_wa05.sh stage-prefix
#   PORT=57020 CKPT_STEP=070000 ./scripts/eval/gr00t/eval_gr00t_wa05.sh stage-eval
#   CKPT_STEP=070000 ./scripts/eval/gr00t/eval_gr00t_wa05.sh stage-summary
#   PORT=57020 ./scripts/eval/gr00t/eval_gr00t_wa05.sh stop

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CKPT_ROOT="${ROOT}/ckpts/gr00t/wa05"

CKPT_STEP="${CKPT_STEP:-100000}"
CKPT_DIR="${CKPT_DIR:-${CKPT_ROOT}/${CKPT_STEP}/pretrained_model}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/data/shared_workspace/chensihang/pretrained/GR00T-N1.7-3B}"
VLM_MODEL_DIR="${VLM_MODEL_DIR:-/data/shared_workspace/chensihang/pretrained/Cosmos-Reason2-2B}"
# Stage diagnostics retain the legacy policy-controlled first stage.
case "${1:-}" in
    *stage*|prefixes|prefix-summary)
        SUITE="${SUITE:-configs/eval/legacy_suites/wa05_gr00t_r40_v1.yaml}"
        RUN_ID="${RUN_ID:-wa05-gr00t-${CKPT_STEP}-50seeds-r40}"
        ;;
esac
SUITE="${SUITE:-configs/eval/legacy_suites/wa05_gr00t_r40_prefix_v2.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_INFERENCE_TIMEOUT_S="${SERVER_INFERENCE_TIMEOUT_S:-300}"
PORT="${PORT:-57020}"
GPU="${GPU:-3}"
RENDER_GPU="${RENDER_GPU:-${GPU}}"
RETRY_ERRORS="${RETRY_ERRORS:-0}"

RUN_ID="${RUN_ID:-wa05-gr00t-${CKPT_STEP}-50seeds-r40-prefix-v2}"
STAGE_SUITE="${STAGE_SUITE:-configs/eval/stages/wa05_gr00t_r40_v1.yaml}"
PREFIX_ROOT="${PREFIX_ROOT:-}"
STAGE_OUTPUT_ROOT="${STAGE_OUTPUT_ROOT:-outputs/stage_eval}"
STAGE_RUN_ID="${STAGE_RUN_ID:-wa05-gr00t-${CKPT_STEP}-stage-r40}"
STAGE_RECORD_VIDEO="${STAGE_RECORD_VIDEO:-}"
MAIN_ROLLOUT_RUN="${MAIN_ROLLOUT_RUN:-${OUTPUT_ROOT}/${RUN_ID}}"
LOG_DIR="${OUTPUT_ROOT}/logs"
PID_FILE="${LOG_DIR}/serve_gr00t_wa05_${PORT}.pid"
LOG_FILE="${LOG_DIR}/serve_gr00t_wa05_${PORT}.log"

GROOT_PYTHON="${GROOT_PYTHON:-${ROOT}/.venv-gr00t/bin/python}"
ROBOCASA_PYTHON="${ROBOCASA_PYTHON:-${ROOT}/.venv-robocasa/bin/python}"
MPLIB_PYTHON="${MPLIB_PYTHON:-${ROOT}/.venv-mplib/bin/python}"
MANISKILL_PANDA_DIR="${MANISKILL_PANDA_DIR:-${ROOT}/.venv-robocasa/lib/python3.11/site-packages/mani_skill/assets/robots/panda}"

SERVER_ARGS=(
    scripts/serve/serve_gr00t.py
    --checkpoint "${CKPT_DIR}"
    --base-model-dir "${BASE_MODEL_DIR}"
    --vlm-model-dir "${VLM_MODEL_DIR}"
    --image-size 224
    --state-schema membench.robot_state37.v2
    --inference-timeout-s "${SERVER_INFERENCE_TIMEOUT_S}"
    --host 0.0.0.0
    --port "${PORT}"
)

usage() {
    sed -n '2,9p' "${BASH_SOURCE[0]}" | sed -n 's/^# \{0,1\}//p'
}

validate_common() {
    [[ -x "${ROBOCASA_PYTHON}" ]] || {
        echo "RoboCasa environment is missing: ${ROBOCASA_PYTHON}" >&2
        exit 1
    }
    [[ -f "${ROOT}/${SUITE}" || -f "${SUITE}" ]] || {
        echo "Evaluation suite is missing: ${SUITE}" >&2
        exit 1
    }
}

validate_server() {
    validate_common
    [[ -x "${GROOT_PYTHON}" ]] || {
        echo "GR00T environment is missing: ${GROOT_PYTHON}" >&2
        exit 1
    }
    [[ -f "${CKPT_DIR}/train_config.json" ]] || {
        echo "GR00T checkpoint is missing: ${CKPT_DIR}" >&2
        exit 1
    }
    [[ -f "${BASE_MODEL_DIR}/config.json" ]] || {
        echo "GR00T base model is missing: ${BASE_MODEL_DIR}" >&2
        exit 1
    }
    [[ -f "${VLM_MODEL_DIR}/tokenizer.json" ]] || {
        echo "Cosmos tokenizer is missing: ${VLM_MODEL_DIR}" >&2
        exit 1
    }
}

serve_foreground() {
    validate_server
    mkdir -p "${LOG_DIR}"
    echo $$ >"${PID_FILE}"
    cd "${ROOT}"
    export CUDA_VISIBLE_DEVICES="${GPU}"
    exec "${GROOT_PYTHON}" "${SERVER_ARGS[@]}"
}

start_server() {
    validate_server
    mkdir -p "${LOG_DIR}"
    if [[ -f "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
        echo "GR00T server ${PORT} is already running (pid $(<"${PID_FILE}"))"
        return 0
    fi
    cd "${ROOT}"
    CUDA_VISIBLE_DEVICES="${GPU}" nohup "${GROOT_PYTHON}" "${SERVER_ARGS[@]}" \
        >"${LOG_FILE}" 2>&1 &
    echo $! >"${PID_FILE}"
    echo "Started GR00T server ${PORT} (pid $!, GPU ${GPU}, checkpoint ${CKPT_STEP})"
    echo "Log: ${LOG_FILE}"
}

stop_server() {
    if [[ -f "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
        kill "$(<"${PID_FILE}")"
        rm -f "${PID_FILE}"
        echo "Stopped GR00T server ${PORT}"
    else
        rm -f "${PID_FILE}" 2>/dev/null || true
        echo "GR00T server ${PORT} is not running"
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
    [[ -x "${MPLIB_PYTHON}" ]] || {
        echo "MPlib environment is missing: ${MPLIB_PYTHON}" >&2
        exit 1
    }
    [[ -d "${MANISKILL_PANDA_DIR}" ]] || {
        echo "ManiSkill Panda assets are missing: ${MANISKILL_PANDA_DIR}" >&2
        exit 1
    }
    if [[ "${RETRY_ERRORS}" == "1" ]]; then
        retry_args=(--retry-errors)
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

run_stage_prefix() {
    local prefix_args=()

    validate_common
    if [[ -n "${PREFIX_ROOT}" ]]; then
        prefix_args=(--output-root "${PREFIX_ROOT}")
    fi
    cd "${ROOT}"
    export PYTHONPATH=.
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    export MUJOCO_EGL_DEVICE_ID="${RENDER_GPU}"
    export MEMBENCH_MPLIB_PYTHON="${MEMBENCH_MPLIB_PYTHON:-${MPLIB_PYTHON}}"
    export MEMBENCH_MANISKILL_PANDA_DIR="${MEMBENCH_MANISKILL_PANDA_DIR:-${MANISKILL_PANDA_DIR}}"
    "${ROBOCASA_PYTHON}" scripts/eval/build_wa05_expert_prefixes.py \
        --stage-suite "${STAGE_SUITE}" \
        "${prefix_args[@]}"
}

run_stage_eval() {
    local prefix_args=()
    local video_args=()

    validate_common
    if [[ -n "${PREFIX_ROOT}" ]]; then
        prefix_args=(--prefix-root "${PREFIX_ROOT}")
    fi
    if [[ "${STAGE_RECORD_VIDEO}" == "0" ]]; then
        video_args=(--no-record-video)
    elif [[ "${STAGE_RECORD_VIDEO}" == "1" ]]; then
        video_args=(--record-video)
    elif [[ -n "${STAGE_RECORD_VIDEO}" ]]; then
        echo "STAGE_RECORD_VIDEO must be 0, 1, or empty" >&2
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
        --main-rollout-run "${MAIN_ROLLOUT_RUN}" \
        --output-root "${STAGE_OUTPUT_ROOT}" \
        --run-id "${STAGE_RUN_ID}" \
        "${prefix_args[@]}" \
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
        echo "GR00T server ${PORT}: running (pid $(<"${PID_FILE}"), checkpoint ${CKPT_STEP})"
    else
        echo "GR00T server ${PORT}: not running"
    fi
}

case "${1:-}" in
    start) start_server ;;
    serve-fg|serve) serve_foreground ;;
    preflight) run_preflight ;;
    eval) run_eval ;;
    summary) run_summary ;;
    stage-prefix) run_stage_prefix ;;
    stage-eval) run_stage_eval ;;
    stage-summary) run_stage_summary ;;
    stop) stop_server ;;
    status) status ;;
    help|-h|--help) usage ;;
    *)
        echo "Unknown command: ${1:-}" >&2
        usage >&2
        exit 2
        ;;
esac
