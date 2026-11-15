#!/usr/bin/env bash
# WR12: same ACT/DP presets as TS02/WR11, independent single-GPU runs.
set -euo pipefail
policy="${1:?Usage: bash scripts/train/train_wr12_40k.sh act|diffusion GPU}"
gpu="${2:?Specify a physical GPU index}"
case "$policy" in act|diffusion) ;; *) echo "Invalid policy: $policy" >&2; exit 2 ;; esac
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "GPU must be a numeric index" >&2; exit 2; }
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
python_bin="${LEROBOT_PYTHON:-/home/chensihang/miniconda3/envs/membench/bin/python}"
config="$root/configs/train/wr12_40k_v1/${policy}_wr12.json"
[[ -f "$config" && -f data/final/wr12-final/meta/info.json ]]
if [[ -e "outputs/train/${policy}_wr12_40k_v1/wr12" ]]; then
    echo "Run directory exists; resume using its checkpoint config." >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$root/policy/lerobot/src:$root${PYTHONPATH:+:$PYTHONPATH}"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libffi.so.7${LD_PRELOAD:+ $LD_PRELOAD}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ACCELERATE_MIXED_PRECISION=no
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE="$root/outputs/train/cache/huggingface/datasets"
mkdir -p "$HF_DATASETS_CACHE"
"$python_bin" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "Requested CUDA device unavailable"; print("GPU:", torch.cuda.get_device_name(0), flush=True)'
exec "$python_bin" -u -m lerobot.scripts.lerobot_train --config_path="$config"
