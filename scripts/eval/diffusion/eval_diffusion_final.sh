#!/usr/bin/env bash
set -euo pipefail

command="${1:?Usage: eval_diffusion_final.sh serve|preflight|smoke|smoke-summary|eval|summary TASK [STEP] [GPU]}"
task="${2:?Specify one of: tm01 tm02 ts02 ts03 wa01 wa05 wr01 wr08 wr11 wx01}"
step="${3:-40000}"
gpu="${4:-0}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$root"

case "$task" in
    tm01|tm02|ts02|ts03|wa01|wa05|wr01|wr08|wr11|wx01) ;;
    *) echo "Unsupported diffusion task: $task" >&2; exit 2 ;;
esac

checkpoint="${CHECKPOINT:-$root/outputs/train/diffusion_final_aligned_v1/$task/checkpoints/$(printf '%06d' "$step")/pretrained_model}"
suite="$root/configs/eval/diffusion/${task}_r1_v1.yaml"
smoke_suite="$root/configs/eval/diffusion/${task}_r1_seed1049_v1.yaml"
if [[ "$task" == "wa01" || "$task" == "wa05" ]]; then
    suite="$root/configs/eval/diffusion/${task}_r1_prefix_v2.yaml"
    smoke_suite="$root/configs/eval/diffusion/${task}_r1_seed1049_prefix_v2.yaml"
fi
port="${PORT:-57103}"
protocol="v1"
if [[ "$task" == wa01 || "$task" == wa05 ]]; then protocol="prefix-v2"; fi
run_id="${RUN_ID:-diffusion-${task}-${step}-r1-seedrng-${protocol}}"
smoke_run_id="diffusion-${task}-${step}-r1-seedrng-smoke-1049-${protocol}"

if [[ ! -d "$checkpoint" ]]; then
    echo "Checkpoint not found: $checkpoint" >&2
    exit 2
fi

case "$command" in
    serve)
        export CUDA_VISIBLE_DEVICES="$gpu"
        export PYTHONPATH="$root/policy/lerobot/src:$root${PYTHONPATH:+:$PYTHONPATH}"
        exec /home/chensihang/miniconda3/envs/membench/bin/python -u scripts/serve/serve_lerobot.py \
            --checkpoint "$checkpoint" --device cuda --seed-by-episode \
            --host 127.0.0.1 --port "$port" --inference-timeout-s 60
        ;;
    preflight|smoke|eval|smoke-summary|summary)
        export MEMBENCH_RENDER_GPU="${RENDER_GPU:-0}"
        source scripts/tools/activate_sim_env.sh
        if [[ "$command" == "summary" ]]; then
            exec "$ROBOCASA_PYTHON" scripts/eval/summarize_run.py "$root/outputs/eval/$run_id"
        elif [[ "$command" == "smoke-summary" ]]; then
            exec "$ROBOCASA_PYTHON" scripts/eval/summarize_run.py "$root/outputs/eval/$smoke_run_id"
        fi
        selected_suite="$suite"
        selected_run_id="$run_id"
        if [[ "$command" == "smoke" ]]; then
            if [[ ! -f "$smoke_suite" ]]; then
                echo "Smoke suite is not defined for $task." >&2
                exit 2
            fi
            selected_suite="$smoke_suite"
            selected_run_id="$smoke_run_id"
        fi
        args=(--endpoint "ws://127.0.0.1:$port" --suite "$selected_suite")
        if [[ "$command" == "preflight" ]]; then
            args+=(--preflight-only)
        else
            args+=(--output-root "$root/outputs/eval" --run-id "$selected_run_id" --resume --no-retry-errors)
        fi
        exec "$ROBOCASA_PYTHON" -u scripts/eval/run_remote.py "${args[@]}"
        ;;
    *) echo "Unknown command: $command" >&2; exit 2 ;;
esac
