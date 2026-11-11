#!/usr/bin/env bash
# Run the WA-01 MemER + OpenPI composite rollout.
#
#   ./scripts/eval/memer/eval_memer_wa01.sh serve
#   ./scripts/eval/memer/eval_memer_wa01.sh preflight
#   ./scripts/eval/memer/eval_memer_wa01.sh eval
#   ./scripts/eval/memer/eval_memer_wa01.sh summary

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
QWEN_CHECKPOINT="${QWEN_CHECKPOINT:-${ROOT}/ckpts/memer-qwenvlm/wa01-50-stride10-v2-qwen3vl-4b}"
OPENPI_CHECKPOINT="${OPENPI_CHECKPOINT:-${ROOT}/ckpts/memer-pi05/wa01/20000}"
OPENPI_CONFIG="${OPENPI_CONFIG:-memer_wa01}"
OPENPI_NORM_ASSET_ID="${OPENPI_NORM_ASSET_ID:-wa01_200seeds_v061_subtasks_v2}"
SUITE="${SUITE:-configs/eval/suites/wa01_v2.yaml}"
RUN_ID="${RUN_ID:-wa01-memer-full-20000-seed1050-prefix-v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

OPENPI_PYTHON="${OPENPI_PYTHON:-${ROOT}/policy/openpi/.venv/bin/python}"
ROBOCASA_RUN=(conda run -n robocasa)

case "${1:-}" in
  serve)
    [[ -x "${OPENPI_PYTHON}" ]] || { echo "Python not found: ${OPENPI_PYTHON}" >&2; exit 1; }
    [[ -d "${OPENPI_CHECKPOINT}/params" ]] || { echo "OpenPI checkpoint not found: ${OPENPI_CHECKPOINT}" >&2; exit 1; }
    cd "${ROOT}"
    # The composite service loads Qwen with PyTorch and OpenPI with JAX in one
    # process. Disable JAX's default 75% per-GPU preallocation so the two models
    # can coexist when CUDA_VISIBLE_DEVICES exposes the selected GPUs.
    export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
    export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
    exec "${OPENPI_PYTHON}" scripts/serve/serve_memer_wa01.py \
      --qwen-checkpoint "${QWEN_CHECKPOINT}" \
      --openpi-checkpoint "${OPENPI_CHECKPOINT}" \
      --openpi-config "${OPENPI_CONFIG}" \
      --openpi-norm-asset-id "${OPENPI_NORM_ASSET_ID}" \
      --host 0.0.0.0 --port "${PORT}" \
      --state-schema membench.robot_state37.v2 --image-size 224
    ;;
  preflight)
    cd "${ROOT}"
    "${ROBOCASA_RUN[@]}" python scripts/eval/run_remote.py \
      --endpoint "ws://${SERVER_HOST}:${PORT}" --suite "${SUITE}" --preflight-only
    ;;
  eval)
    cd "${ROOT}"
    export PYTHONPATH=.
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    "${ROBOCASA_RUN[@]}" python scripts/eval/run_remote.py \
      --endpoint "ws://${SERVER_HOST}:${PORT}" --suite "${SUITE}" \
      --output-root "${OUTPUT_ROOT}" --run-id "${RUN_ID}"
    ;;
  summary)
    cd "${ROOT}"
    "${ROBOCASA_RUN[@]}" python scripts/eval/summarize_run.py "${OUTPUT_ROOT}/${RUN_ID}"
    ;;
  *)
    echo "usage: $0 {serve|preflight|eval|summary}" >&2
    exit 2
    ;;
esac
