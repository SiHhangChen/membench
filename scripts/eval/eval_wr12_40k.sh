#!/usr/bin/env bash
# Evaluate one trained WR12 checkpoint, optionally rendering on another GPU.
set -euo pipefail
policy="${1:?Usage: eval_wr12_40k.sh act|diffusion GPU PORT RUN_ID}"
gpu="${2:?Specify GPU}"
port="${3:?Specify port}"
run_id="${4:?Specify run ID}"
case "$policy" in act|diffusion) ;; *) exit 2 ;; esac
[[ "$gpu" =~ ^[0-9]+$ && "$port" =~ ^[0-9]+$ ]]
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
checkpoint="$root/outputs/train/${policy}_wr12_40k_v1/wr12/checkpoints/040000/pretrained_model"
suite="$root/configs/eval/suites/wr12_v1.yaml"
if [[ "$policy" == diffusion ]]; then suite="$root/configs/eval/diffusion/wr12_r1_v1.yaml"; fi
suite="${EVAL_SUITE:-$suite}"
log_dir="$root/outputs/eval/logs/$run_id"
mkdir -p "$log_dir"
export OMP_NUM_THREADS="${POLICY_CPU_THREADS:-1}" MKL_NUM_THREADS="${POLICY_CPU_THREADS:-1}" OPENBLAS_NUM_THREADS="${POLICY_CPU_THREADS:-1}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libffi.so.7${LD_PRELOAD:+ $LD_PRELOAD}"
policy_python="${LEROBOT_PYTHON:-/home/chensihang/miniconda3/envs/membench/bin/python}"
"$policy_python" -c 'import socket, sys; s=socket.socket(); s.bind(("127.0.0.1", int(sys.argv[1]))); s.close()' "$port"
server_pid=""
cleanup() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$root/policy/lerobot/src:$root${PYTHONPATH:+:$PYTHONPATH}" \
    "$policy_python" -u scripts/serve/serve_lerobot.py --checkpoint "$checkpoint" \
    --device "${POLICY_DEVICE:-cuda}" --host 127.0.0.1 --port "$port" --seed-by-episode \
    --inference-timeout-s "${INFERENCE_TIMEOUT_S:-60}" >"$log_dir/server.log" 2>&1 &
server_pid=$!
ready=false
for ((attempt=0; attempt<180; attempt++)); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        tail -n 60 "$log_dir/server.log" >&2
        exit 1
    fi
    if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then ready=true; break; fi
    sleep 1
done
[[ "$ready" == true ]] || { echo "Policy startup timed out" >&2; exit 1; }
export MEMBENCH_RENDER_GPU="${RENDER_GPU:-$gpu}"
source scripts/tools/activate_sim_env.sh
args=(--endpoint "ws://127.0.0.1:$port" --suite "$suite")
"$ROBOCASA_PYTHON" scripts/eval/run_remote.py "${args[@]}" --preflight-only >"$log_dir/preflight.json" 2>&1
echo "Preflight passed; starting $run_id GPU=$gpu PORT=$port"
set +e
"$ROBOCASA_PYTHON" -u scripts/eval/run_remote.py "${args[@]}" \
    --output-root "$root/outputs/eval" --run-id "$run_id" --resume --no-retry-errors >"$log_dir/eval.log" 2>&1
eval_code=$?
"$ROBOCASA_PYTHON" scripts/eval/summarize_run.py "$root/outputs/eval/$run_id" >"$log_dir/summary.log" 2>&1
summary_code=$?
set -e
if [[ "$eval_code" != 0 ]]; then exit "$eval_code"; fi
exit "$summary_code"
