#!/usr/bin/env bash
# Local release15 step-200000 WA01 r40 evaluation, with episode-seeded sampling.
set -euo pipefail

_wa01_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${_wa01_root}"
_wa01_command="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
_wa01_port="${PORT:-57101}"
_wa01_output="${OUTPUT_ROOT:-${_wa01_root}/outputs/eval}"
_wa01_formal_id="${RUN_ID:-release15-200000-wa01-r40-seedrng-prefix-v2}"
_wa01_smoke_id="${SMOKE_RUN_ID:-release15-200000-wa01-r40-seedrng-smoke-1049-prefix-v2}"
_wa01_suite="${_wa01_root}/configs/eval/legacy_suites/wa01_gr00t_r40_prefix_v2.yaml"

case "${_wa01_command}" in
    serve)
        # Do not inherit simulation/older LeRobot source overrides in this process.
        export PYTHONPATH="${_wa01_root}"
        export CUDA_VISIBLE_DEVICES="${POLICY_GPU:-0}"
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        exec "${_wa01_root}/.venv-gr00t/bin/python" -u scripts/serve/serve_gr00t.py \
            --checkpoint "${_wa01_root}/outputs/gr00t_n1_7_release15_v1_lora/checkpoints/200000/pretrained_model" \
            --base-model-dir /home/chensihang/pretrained_models/GR00T-N1.7-3B \
            --vlm-model-dir /home/chensihang/pretrained_models/Cosmos-Reason2-2B \
            --image-size 224 --state-schema membench.robot_state37.v2 \
            --pytorch-device cuda:0 --host 127.0.0.1 --port "${_wa01_port}" \
            --inference-timeout-s 30 --seed-by-episode "${@}"
        ;;
    preflight|smoke|eval|summary|smoke-summary)
        source scripts/tools/activate_sim_env.sh
        _wa01_run_id="${_wa01_formal_id}"
        if [[ "${_wa01_command}" == smoke* ]]; then
            _wa01_suite="${_wa01_root}/configs/eval/smoke/wa01_gr00t_r40_seed1049_prefix_v2.yaml"
            _wa01_run_id="${_wa01_smoke_id}"
        fi
        if [[ "${_wa01_command}" == *summary ]]; then
            exec "${ROBOCASA_PYTHON}" scripts/eval/summarize_run.py \
                "${_wa01_output}/${_wa01_run_id}" "${@}"
        fi
        _wa01_args=(
            --endpoint "ws://127.0.0.1:${_wa01_port}"
            --suite "${_wa01_suite}"
        )
        if [[ "${_wa01_command}" == preflight ]]; then
            _wa01_args+=(--preflight-only)
        else
            _wa01_args+=(--output-root "${_wa01_output}" --run-id "${_wa01_run_id}")
            _wa01_args+=(--resume --no-retry-errors)
        fi
        exec "${ROBOCASA_PYTHON}" -u scripts/eval/run_remote.py "${_wa01_args[@]}" "${@}"
        ;;
    help|-h|--help)
        echo "Terminal 1: bash scripts/eval/gr00t/eval_release15_wa01_local.sh serve"
        echo "Terminal 2: bash scripts/eval/gr00t/eval_release15_wa01_local.sh preflight"
        echo "Then:       bash scripts/eval/gr00t/eval_release15_wa01_local.sh smoke"
        echo "After checking smoke artifacts: ... eval (50 formal seeds)"
        echo "Summaries:  ... smoke-summary | summary"
        echo "Overrides: PORT, POLICY_GPU, MEMBENCH_RENDER_GPU, OUTPUT_ROOT, RUN_ID, SMOKE_RUN_ID"
        ;;
    *)
        echo "Unknown command: ${_wa01_command}" >&2
        exit 2
        ;;
esac
