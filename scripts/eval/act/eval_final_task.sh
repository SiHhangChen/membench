#!/usr/bin/env bash
set -euo pipefail

command="${1:?Usage: eval_final_task.sh serve|preflight|eval|summary TASK [STEP] [GPU]}"
task="${2:?Specify task id, e.g. wa05}"
step="${3:-40000}"
gpu="${4:-0}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$root"
checkpoint="${CHECKPOINT:-$root/outputs/train/act/final_bs48_stride10_100k_torchcodec/$task/checkpoints/$(printf '%06d' "$step")/pretrained_model}"
suite="$root/configs/eval/suites/${task}_v1.yaml"
if [[ "$task" == wa01 || "$task" == wa05 ]]; then
    suite="$root/configs/eval/suites/${task}_v2.yaml"
elif [[ "$task" == wr11 ]]; then
    suite="$root/configs/eval/suites/wr11_v3.yaml"
fi
port="${PORT:-57102}"
run_suffix=""
if [[ "$task" == wa01 || "$task" == wa05 ]]; then run_suffix="-prefix-v2"; fi
run_id="${RUN_ID:-act-${task}-${step}${run_suffix}}"

case "$command" in
    serve)
        /home/chensihang/miniconda3/envs/membench/bin/python scripts/eval/act/prepare_checkpoint.py --task "$task" --step "$step" --checkpoint "$checkpoint"
        export CUDA_VISIBLE_DEVICES="$gpu"
        export PYTHONPATH="$root/policy/lerobot/src:$root${PYTHONPATH:+:$PYTHONPATH}"
        exec /home/chensihang/miniconda3/envs/membench/bin/python -u scripts/serve/serve_lerobot.py \
            --checkpoint "$checkpoint" --device cuda --host 127.0.0.1 --port "$port"
        ;;
    preflight|eval|summary)
        source scripts/tools/activate_sim_env.sh
        if [[ "$command" == summary ]]; then
            exec "$ROBOCASA_PYTHON" scripts/eval/summarize_run.py "$root/outputs/eval/$run_id"
        fi
        args=(--endpoint "ws://127.0.0.1:$port" --suite "$suite")
        if [[ "$command" == preflight ]]; then
            args+=(--preflight-only)
        else
            args+=(--output-root "$root/outputs/eval" --run-id "$run_id" --resume --no-retry-errors)
        fi
        exec "$ROBOCASA_PYTHON" -u scripts/eval/run_remote.py "${args[@]}"
        ;;
    *) echo "Unknown command: $command" >&2; exit 2 ;;
esac
