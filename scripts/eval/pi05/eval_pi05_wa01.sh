#!/usr/bin/env bash
#
# 评测 pi05 LoRA checkpoint（WA-01）的可复用入口。
#
# 用法（各子命令在对应终端执行）：
#   ./scripts/eval/pi05/eval_pi05_wa01.sh serve      终端1：启动 OpenPI 策略服务（常驻）
#   ./scripts/eval/pi05/eval_pi05_wa01.sh preflight  终端2：协议 preflight 检查
#   ./scripts/eval/pi05/eval_pi05_wa01.sh eval       终端2：闭环评测
#   ./scripts/eval/pi05/eval_pi05_wa01.sh summary    任意终端：重建评测汇总
#   ./scripts/eval/pi05/eval_pi05_wa01.sh prefixes   生成 WA-01 专家阶段轨迹
#   ./scripts/eval/pi05/eval_pi05_wa01.sh prefix-summary  检查 prefix 并计算预算
#   ./scripts/eval/pi05/eval_pi05_wa01.sh stage-preflight  阶段协议检查
#   ./scripts/eval/pi05/eval_pi05_wa01.sh stage-eval       阶段评测
#   ./scripts/eval/pi05/eval_pi05_wa01.sh stage-summary    阶段汇总
#
# WA-01 纯低层、特权子任务切换 pilot：
#   CKPT_DIR=/data/chensihang/ckpts/memer/pi05-2view/15000 \
#   CONFIG_NAME=memer_multi_15 PORT=55070 \
#   ./scripts/eval/pi05/eval_pi05_wa01.sh serve
#   SUITE=configs/eval/legacy_suites/wa01_pi05_oracle_subtasks_v1.yaml \
#   PORT=55070 RUN_ID=wa01-pi05-oracle-15000-pilot \
#   ./scripts/eval/pi05/eval_pi05_wa01.sh eval
#
# 可通过环境变量覆盖（保持默认即可复用）：
#   CKPT_DIR      checkpoint 目录，默认 ckpts/pi05/.../45000
#   CONFIG_NAME   OpenPI 配置名，默认 pi05_membench_wa01_lora
#   SUITE         评测 suite，默认 configs/eval/suites/wa01_v2.yaml
#   RUN_ID        评测运行 ID，默认 wa01-pi05-lora-45000-prefix-v2
#   OUTPUT_ROOT   评测输出根目录，默认 outputs/eval
#   SERVER_HOST   客户端连接的服务端地址，默认 127.0.0.1（同机）
#   PORT          服务端端口，默认 8000
#   RETRY_ERRORS  设为 1 时续跑 status=error 的 episode（断网中断后恢复用），默认 0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

CKPT_DIR="${CKPT_DIR:-${ROOT}/ckpts/pi05/pi05_membench_wa01_lora/pi05_lora_membench_wa01/45000}"
CONFIG_NAME="${CONFIG_NAME:-pi05_membench_wa01_lora}"
# Stage diagnostics retain the legacy policy-controlled first stage.
case "${1:-}" in
    *stage*|prefixes|prefix-summary)
        SUITE="${SUITE:-configs/eval/suites/wa01_v1.yaml}"
        RUN_ID="${RUN_ID:-wa01-pi05-lora-45000}"
        ;;
esac
SUITE="${SUITE:-configs/eval/suites/wa01_v2.yaml}"
RUN_ID="${RUN_ID:-wa01-pi05-lora-45000-prefix-v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
RETRY_ERRORS="${RETRY_ERRORS:-0}"
STAGE_SUITE="${STAGE_SUITE:-configs/eval/stages/wa01_pi05_r50_pilot_v1.yaml}"
PREFIX_ROOT="${PREFIX_ROOT:-outputs/stage_prefixes/wa01_pilot_v1}"
STAGE_OUTPUT_ROOT="${STAGE_OUTPUT_ROOT:-outputs/stage_eval}"
STAGE_RUN_ID="${STAGE_RUN_ID:-${RUN_ID}-stage-pilot-v1}"
MAIN_ROLLOUT_RUN="${MAIN_ROLLOUT_RUN:-${OUTPUT_ROOT}/${RUN_ID}}"
STAGE_RECORD_VIDEO="${STAGE_RECORD_VIDEO:-0}"

OPENPI_PYTHON="${ROOT}/policy/openpi/.venv/bin/python"
ROBOCASA_PYTHON="${ROOT}/.venv-robocasa/bin/python"
MPLIB_PYTHON="${ROOT}/.venv-mplib/bin/python"
MANISKILL_PANDA_DIR="${ROOT}/.venv-robocasa/lib/python3.11/site-packages/mani_skill/assets/robots/panda"

usage() {
    sed -n '2,40p' "${BASH_SOURCE[0]}" | grep -E '^#' | sed 's/^# \{0,1\}//'
    exit 0
}

stage_video_args() {
    if [[ "${STAGE_RECORD_VIDEO}" == "1" ]]; then
        printf '%s\n' --record-video
    elif [[ "${STAGE_RECORD_VIDEO}" == "0" ]]; then
        printf '%s\n' --no-record-video
    else
        echo "STAGE_RECORD_VIDEO must be 0 or 1" >&2
        return 2
    fi
}

validate_runtime() {
    [[ -x "${ROBOCASA_PYTHON}" ]] || {
        echo "RoboCasa .venv 不存在: ${ROBOCASA_PYTHON}" >&2
        return 1
    }
    [[ -x "${MPLIB_PYTHON}" ]] || {
        echo "MPlib .venv 不存在: ${MPLIB_PYTHON}" >&2
        return 1
    }
    [[ -d "${MANISKILL_PANDA_DIR}" ]] || {
        echo "ManiSkill Panda assets 不存在: ${MANISKILL_PANDA_DIR}" >&2
        return 1
    }
}

configure_runtime() {
    validate_runtime
    export PYTHONPATH=.
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    export MEMBENCH_MPLIB_PYTHON="${MEMBENCH_MPLIB_PYTHON:-${MPLIB_PYTHON}}"
    export MEMBENCH_MANISKILL_PANDA_DIR="${MEMBENCH_MANISKILL_PANDA_DIR:-${MANISKILL_PANDA_DIR}}"
}

case "${1:-}" in
    serve)
        [[ -x "${OPENPI_PYTHON}" ]] || { echo "OpenPI .venv 不存在: ${OPENPI_PYTHON}" >&2; exit 1; }
        [[ -d "${CKPT_DIR}/params" ]] || { echo "checkpoint 不存在: ${CKPT_DIR}" >&2; exit 1; }
        cd "${ROOT}"
        export OPENPI_PRETRAIN_MODEL_DIR="${OPENPI_PRETRAIN_MODEL_DIR:-${HOME}/openpi/pretrained}"
        export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
        exec "${OPENPI_PYTHON}" scripts/serve/serve_openpi.py \
            --config "${CONFIG_NAME}" \
            --checkpoint "${CKPT_DIR}" \
            --image-size 224 \
            --state-schema membench.robot_state37.v2 \
            --host 0.0.0.0 \
            --port "${PORT}"
        ;;
    preflight)
        validate_runtime
        cd "${ROOT}"
        "${ROBOCASA_PYTHON}" scripts/eval/run_remote.py \
            --endpoint "ws://${SERVER_HOST}:${PORT}" \
            --suite "${SUITE}" \
            --preflight-only
        ;;
    eval)
        cd "${ROOT}"
        configure_runtime
        retry_args=()
        if [[ "${RETRY_ERRORS}" == "1" ]]; then
            retry_args=(--retry-errors)
        fi
        "${ROBOCASA_PYTHON}" scripts/eval/run_remote.py \
            --endpoint "ws://${SERVER_HOST}:${PORT}" \
            --suite "${SUITE}" \
            --output-root "${OUTPUT_ROOT}" \
            --run-id "${RUN_ID}" \
            "${retry_args[@]}"
        ;;
    summary)
        validate_runtime
        cd "${ROOT}"
        "${ROBOCASA_PYTHON}" scripts/eval/summarize_run.py "${OUTPUT_ROOT}/${RUN_ID}"
        ;;
    prefixes)
        cd "${ROOT}"
        configure_runtime
        "${ROBOCASA_PYTHON}" scripts/eval/build_expert_prefixes.py \
            --stage-suite "${STAGE_SUITE}" \
            --output-root "${PREFIX_ROOT}" \
            "${@:2}"
        ;;
    prefix-summary)
        validate_runtime
        cd "${ROOT}"
        "${ROBOCASA_PYTHON}" scripts/eval/summarize_expert_prefixes.py \
            --stage-suite "${STAGE_SUITE}" \
            --prefix-root "${PREFIX_ROOT}" \
            --output "${PREFIX_ROOT}/summary.json" \
            "${@:2}"
        ;;
    stage-preflight)
        validate_runtime
        cd "${ROOT}"
        "${ROBOCASA_PYTHON}" scripts/eval/run_stage_remote.py \
            --endpoint "ws://${SERVER_HOST}:${PORT}" \
            --stage-suite "${STAGE_SUITE}" \
            --preflight-only
        ;;
    stage-eval)
        [[ -d "${PREFIX_ROOT}" ]] || {
            echo "专家 prefix 不存在: ${PREFIX_ROOT}" >&2
            exit 1
        }
        [[ -f "${MAIN_ROLLOUT_RUN}/summary.json" ]] || {
            echo "需要同 seed 的完整主表结果: ${MAIN_ROLLOUT_RUN}" >&2
            exit 1
        }
        video_arg="$(stage_video_args)"
        cd "${ROOT}"
        configure_runtime
        "${ROBOCASA_PYTHON}" scripts/eval/run_stage_remote.py \
            --endpoint "ws://${SERVER_HOST}:${PORT}" \
            --stage-suite "${STAGE_SUITE}" \
            --prefix-root "${PREFIX_ROOT}" \
            --main-rollout-run "${MAIN_ROLLOUT_RUN}" \
            --output-root "${STAGE_OUTPUT_ROOT}" \
            --run-id "${STAGE_RUN_ID}" \
            "${video_arg}"
        ;;
    stage-summary)
        validate_runtime
        cd "${ROOT}"
        "${ROBOCASA_PYTHON}" scripts/eval/summarize_stage_run.py \
            "${STAGE_OUTPUT_ROOT}/${STAGE_RUN_ID}"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "未知子命令: ${1:-}" >&2
        usage
        ;;
esac
