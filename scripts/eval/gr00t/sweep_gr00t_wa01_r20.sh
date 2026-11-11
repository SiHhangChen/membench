#!/usr/bin/env bash
# Sequentially evaluate WA-01 GR00T checkpoints with one shared GPU and prefix set.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EVAL_SCRIPT="${ROOT}/scripts/eval/gr00t/eval_gr00t_wa01.sh"
GPU="${GPU:-0}"
RENDER_GPU="${RENDER_GPU:-${GPU}}"
PORT="${PORT:-57101}"
# This campaign includes first-stage policy diagnostics and retains protocol v1.
export SUITE="${SUITE:-configs/eval/legacy_suites/wa01_gr00t_r20_v1.yaml}"

if [[ "$#" -gt 0 ]]; then
    CHECKPOINTS=("${@}")
else
    CHECKPOINTS=(100000 070000 080000 090000)
fi

cd "${ROOT}"

if [[ "${SKIP_PREFIXES:-0}" != "1" ]]; then
    GPU="${GPU}" RENDER_GPU="${RENDER_GPU}" "${EVAL_SCRIPT}" prefixes
    "${EVAL_SCRIPT}" prefix-summary
fi

for step in "${CHECKPOINTS[@]}"; do
    export CKPT_STEP="${step}"
    export GPU RENDER_GPU PORT
    "${EVAL_SCRIPT}" start
    trap '"${EVAL_SCRIPT}" stop' EXIT
    for attempt in {1..30}; do
        if "${EVAL_SCRIPT}" preflight; then
            break
        fi
        if [[ "${attempt}" == "30" ]]; then
            echo "GR00T server did not become ready for checkpoint ${step}" >&2
            exit 1
        fi
        sleep 10
    done
    RUN_ID="${RUN_ID:-wa01-gr00t-${step}-50seeds-r20}" "${EVAL_SCRIPT}" eval
    "${EVAL_SCRIPT}" stage-eval
    "${EVAL_SCRIPT}" stage-summary
    "${EVAL_SCRIPT}" stop
    trap - EXIT
done
