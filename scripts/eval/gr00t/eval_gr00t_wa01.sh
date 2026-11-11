#!/usr/bin/env bash
# Evaluate GR00T N1.7 WA-01 checkpoints on the shared 50-seed r20 protocol.
#
#   GPU=0 PORT=57101 CKPT_STEP=100000 ./scripts/eval/gr00t/eval_gr00t_wa01.sh start
#   PORT=57101 CKPT_STEP=100000 ./scripts/eval/gr00t/eval_gr00t_wa01.sh preflight
#   PORT=57101 CKPT_STEP=100000 ./scripts/eval/gr00t/eval_gr00t_wa01.sh eval
#   ./scripts/eval/gr00t/eval_gr00t_wa01.sh prefixes
#   ./scripts/eval/gr00t/eval_gr00t_wa01.sh prefix-summary
#   PORT=57101 CKPT_STEP=100000 ./scripts/eval/gr00t/eval_gr00t_wa01.sh stage-eval
#   CKPT_STEP=100000 ./scripts/eval/gr00t/eval_gr00t_wa01.sh stage-summary
#   PORT=57101 ./scripts/eval/gr00t/eval_gr00t_wa01.sh stop
#   GPU=0 PORT=57101 CKPT_STEP=100000 ./scripts/eval/gr00t/eval_gr00t_wa01.sh full-main
#   GPU=0 PORT=57101 CKPT_STEP=100000 ./scripts/eval/gr00t/eval_gr00t_wa01.sh full-stage

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CKPT_ROOT="${CKPT_ROOT:-/data/chensihang/ckpts/gr00t/wa01}"

CKPT_STEP="${CKPT_STEP:-100000}"
CKPT_DIR="${CKPT_DIR:-${CKPT_ROOT}/${CKPT_STEP}/pretrained_model}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/data/shared_workspace/chensihang/pretrained/GR00T-N1.7-3B}"
VLM_MODEL_DIR="${VLM_MODEL_DIR:-/data/shared_workspace/chensihang/pretrained/Cosmos-Reason2-2B}"
# Stage diagnostics retain the legacy policy-controlled first stage.
case "${1:-}" in
    *stage*|prefixes|prefix-summary)
        SUITE="${SUITE:-configs/eval/legacy_suites/wa01_gr00t_r20_v1.yaml}"
        RUN_ID="${RUN_ID:-wa01-gr00t-${CKPT_STEP}-50seeds-r20}"
        ;;
esac
SUITE="${SUITE:-configs/eval/legacy_suites/wa01_gr00t_r20_prefix_v2.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_INFERENCE_TIMEOUT_S="${SERVER_INFERENCE_TIMEOUT_S:-300}"
PORT="${PORT:-57101}"
GPU="${GPU:-0}"
RENDER_GPU="${RENDER_GPU:-${GPU}}"
RETRY_ERRORS="${RETRY_ERRORS:-0}"

RUN_ID="${RUN_ID:-wa01-gr00t-${CKPT_STEP}-50seeds-r20-prefix-v2}"
STAGE_SUITE="${STAGE_SUITE:-configs/eval/stages/wa01_gr00t_r20_pilot_v1.yaml}"
PREFIX_ROOT="${PREFIX_ROOT:-outputs/stage_prefixes/wa01_gr00t_r20_pilot_v1}"
STAGE_OUTPUT_ROOT="${STAGE_OUTPUT_ROOT:-outputs/stage_eval}"
STAGE_RUN_ID="${STAGE_RUN_ID:-wa01-gr00t-${CKPT_STEP}-stage-50seeds-r20-pilot-v1}"
STAGE_RECORD_VIDEO="${STAGE_RECORD_VIDEO:-0}"
MAIN_ROLLOUT_RUN="${MAIN_ROLLOUT_RUN:-${OUTPUT_ROOT}/${RUN_ID}}"
LOG_DIR="${OUTPUT_ROOT}/logs"
PID_FILE="${LOG_DIR}/serve_gr00t_wa01_${PORT}.pid"
LOG_FILE="${LOG_DIR}/serve_gr00t_wa01_${PORT}.log"

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
    sed -n '2,11p' "${BASH_SOURCE[0]}" | sed -n 's/^# \{0,1\}//p'
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

validate_runtime() {
    validate_common
    [[ -x "${MPLIB_PYTHON}" ]] || {
        echo "MPlib environment is missing: ${MPLIB_PYTHON}" >&2
        exit 1
    }
    [[ -d "${MANISKILL_PANDA_DIR}" ]] || {
        echo "ManiSkill Panda assets are missing: ${MANISKILL_PANDA_DIR}" >&2
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

configure_runtime() {
    validate_runtime
    export PYTHONPATH=.
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    export MUJOCO_EGL_DEVICE_ID="${RENDER_GPU}"
    export MEMBENCH_MPLIB_PYTHON="${MEMBENCH_MPLIB_PYTHON:-${MPLIB_PYTHON}}"
    export MEMBENCH_MANISKILL_PANDA_DIR="${MEMBENCH_MANISKILL_PANDA_DIR:-${MANISKILL_PANDA_DIR}}"
    export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
    export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
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
    [[ "${RETRY_ERRORS}" == "1" ]] && retry_args=(--retry-errors)
    cd "${ROOT}"
    configure_runtime
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

run_prefixes() {
    cd "${ROOT}"
    configure_runtime
    "${ROBOCASA_PYTHON}" scripts/eval/build_expert_prefixes.py \
        --stage-suite "${STAGE_SUITE}" \
        --output-root "${PREFIX_ROOT}" \
        "${@}"
}

run_prefix_summary() {
    validate_common
    cd "${ROOT}"
    "${ROBOCASA_PYTHON}" scripts/eval/summarize_expert_prefixes.py \
        --stage-suite "${STAGE_SUITE}" \
        --prefix-root "${PREFIX_ROOT}" \
        --output "${PREFIX_ROOT}/summary.json" \
        "${@}"
}

run_stage_preflight() {
    validate_common
    cd "${ROOT}"
    "${ROBOCASA_PYTHON}" scripts/eval/run_stage_remote.py \
        --endpoint "ws://${SERVER_HOST}:${PORT}" \
        --stage-suite "${STAGE_SUITE}" \
        --preflight-only
}

run_stage_eval() {
    local video_args=()
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
    configure_runtime
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
        echo "GR00T server ${PORT}: running (pid $(<"${PID_FILE}"), checkpoint ${CKPT_STEP})"
    else
        echo "GR00T server ${PORT}: not running"
    fi
}

wait_for_preflight() {
    for attempt in {1..30}; do
        if run_preflight; then
            return 0
        fi
        if [[ ! -f "${PID_FILE}" ]] || ! kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
            echo "GR00T server exited before preflight completed; see ${LOG_FILE}" >&2
            return 1
        fi
        echo "GR00T server is still loading (attempt ${attempt}/30)"
        sleep 10
    done
    echo "GR00T server did not become ready; see ${LOG_FILE}" >&2
    return 1
}

run_full_main() {
    start_server
    trap stop_server EXIT
    wait_for_preflight
    run_eval
    run_summary
}

run_full_stage() {
    start_server
    trap stop_server EXIT
    wait_for_preflight
    run_stage_eval
    run_stage_summary
}

case "${1:-}" in
    start) start_server ;;
    serve-fg|serve) serve_foreground ;;
    preflight) run_preflight ;;
    eval) run_eval ;;
    summary) run_summary ;;
    prefixes) shift; run_prefixes "${@}" ;;
    prefix-summary) shift; run_prefix_summary "${@}" ;;
    stage-preflight) run_stage_preflight ;;
    stage-eval) run_stage_eval ;;
    stage-summary) run_stage_summary ;;
    full-main) run_full_main ;;
    full-stage) run_full_stage ;;
    stop) stop_server ;;
    status) status ;;
    help|-h|--help) usage ;;
    *)
        echo "Unknown command: ${1:-}" >&2
        usage >&2
        exit 2
        ;;
esac
