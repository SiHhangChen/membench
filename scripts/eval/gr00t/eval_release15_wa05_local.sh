#!/usr/bin/env bash
# Local release15 step-200000 WA05 r40 evaluation, with episode-seeded sampling.
set -euo pipefail

_wa05_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${_wa05_root}"
_wa05_command="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
_wa05_port="${PORT:-57101}"
_wa05_output="${OUTPUT_ROOT:-${_wa05_root}/outputs/eval}"
_wa05_formal_id="${RUN_ID:-release15-200000-wa05-r40-seedrng-prefix-v2}"
_wa05_smoke_id="${SMOKE_RUN_ID:-release15-200000-wa05-r40-seedrng-smoke-1049-prefix-v2}"
_wa05_suite="${_wa05_root}/configs/eval/legacy_suites/wa05_gr00t_r40_prefix_v2.yaml"

case "${_wa05_command}" in
    serve)
        # The joint model is task-neutral; reuse it for release15 task clients.
        export PYTHONPATH="${_wa05_root}"
        export CUDA_VISIBLE_DEVICES="${POLICY_GPU:-0}"
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        exec "${_wa05_root}/.venv-gr00t/bin/python" -u scripts/serve/serve_gr00t.py \
            --checkpoint "${_wa05_root}/outputs/gr00t_n1_7_release15_v1_lora/checkpoints/200000/pretrained_model" \
            --base-model-dir /home/chensihang/pretrained_models/GR00T-N1.7-3B \
            --vlm-model-dir /home/chensihang/pretrained_models/Cosmos-Reason2-2B \
            --image-size 224 --state-schema membench.robot_state37.v2 \
            --pytorch-device cuda:0 --host 127.0.0.1 --port "${_wa05_port}" \
            --inference-timeout-s 30 --seed-by-episode "${@}"
        ;;
    preflight|smoke|eval|summary|smoke-summary)
        source scripts/tools/activate_sim_env.sh
        _wa05_run_id="${_wa05_formal_id}"
        if [[ "${_wa05_command}" == smoke* ]]; then
            _wa05_suite="${_wa05_root}/configs/eval/smoke/wa05_gr00t_r40_seed1049_prefix_v2.yaml"
            _wa05_run_id="${_wa05_smoke_id}"
        fi
        if [[ "${_wa05_command}" == *summary ]]; then
            exec "${ROBOCASA_PYTHON}" scripts/eval/summarize_run.py \
                "${_wa05_output}/${_wa05_run_id}" "${@}"
        fi
        _wa05_args=(
            --endpoint "ws://127.0.0.1:${_wa05_port}"
            --suite "${_wa05_suite}"
        )
        if [[ "${_wa05_command}" == preflight ]]; then
            _wa05_args+=(--preflight-only)
        else
            _wa05_args+=(--output-root "${_wa05_output}" --run-id "${_wa05_run_id}")
            _wa05_args+=(--resume --no-retry-errors)
        fi
        exec "${ROBOCASA_PYTHON}" -u scripts/eval/run_remote.py "${_wa05_args[@]}" "${@}"
        ;;
    help|-h|--help)
        echo "Terminal 1: bash scripts/eval/gr00t/eval_release15_wa05_local.sh serve"
        echo "Reuse the existing release15 service if it is already on this port."
        echo "Terminal 2: bash scripts/eval/gr00t/eval_release15_wa05_local.sh preflight"
        echo "Then:       bash scripts/eval/gr00t/eval_release15_wa05_local.sh smoke"
        echo "After checking smoke artifacts: ... eval (50 formal seeds)"
        echo "Summaries:  ... smoke-summary | summary"
        echo "Overrides: PORT, POLICY_GPU, MEMBENCH_RENDER_GPU, OUTPUT_ROOT, RUN_ID, SMOKE_RUN_ID"
        ;;
    *)
        echo "Unknown command: ${_wa05_command}" >&2
        exit 2
        ;;
esac
