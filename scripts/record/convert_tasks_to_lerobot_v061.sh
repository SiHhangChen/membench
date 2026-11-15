#!/bin/bash
# ============================================================
# 通用: 原生 MemBench episode → LeRobot 0.6.1 数据集 (参数化版)
# ============================================================
# 用法:
#   conda activate lerobot061
#   cd /home/user/chensihang/membench
#   bash scripts/record/convert_tasks_to_lerobot_v061.sh TS01 [200]
#   bash scripts/record/convert_tasks_to_lerobot_v061.sh TS03 [200]
#   bash scripts/record/convert_tasks_to_lerobot_v061.sh WA-02 [200]
#   bash scripts/record/convert_tasks_to_lerobot_v061.sh WR-12 200 \
#     outputs/datasets/wr12_200success_v14_complete data/final/wr12-final wr12-final
#   Optional arguments: INPUT_SHARD OUTPUT_ROOT REPO_ID.
#   PYTHON and MEMBENCH_LEROBOT_SRC may override the active interpreter/source.
#
# 前置:
#   - conda 环境 lerobot061 (Python 3.12 + lerobot 0.6.1)
#   - 原生 episode 目录结构: outputs/datasets/<TASK>/train/<TASK_ID>/seed_*_attempt_*
#
# 注意:
#   - 参数是 catalog 中的 task_id (可能带连字符, 如 WA-02); 数据集目录与输出
#     命名使用去连字符形式 (如 outputs/datasets/WA02, wa02_200seeds_v061)
#   - 转换器 create 模式要求输出根目录不存在 (不要预先创建)
#   - 默认校验 episode 数 == 200, 不符则中止并打印 attempt 分布
# ============================================================

set -euo pipefail

TASK_UPPER="${1:?用法: bash $0 TASK_ID [EXPECTED_EPISODES], 如 TS01 或 WA-02}"
EXPECTED_EPISODES="${2:-200}"
TASK_LOWER=$(echo "${TASK_UPPER//-/}" | tr 'A-Z' 'a-z')

SHARD="${3:-outputs/datasets/${TASK_UPPER//-/}}"
TASK_DIR="$SHARD/train/${TASK_UPPER}"
OUTPUT_ROOT="${4:-lerobot_data/${TASK_LOWER}_${EXPECTED_EPISODES}seeds_v061}"
REPO_ID="${5:-local/${TASK_LOWER}_${EXPECTED_EPISODES}seeds_v061}"
PYTHON="${PYTHON:-python}"

if [[ ! -d "$TASK_DIR" ]]; then
  echo "错误: 任务目录不存在: $TASK_DIR"
  echo "提示: 参数应为 catalog 中的 task_id (可能带连字符), 如 WA-02;"
  echo "      数据集目录为 outputs/datasets/WA02 (去连字符大写)"
  exit 1
fi

# ==================== 收集 episode ====================
episode_dirs=()
while IFS= read -r -d '' d; do
  episode_dirs+=("$d")
done < <(find -L "$TASK_DIR" -mindepth 1 -maxdepth 1 -type d -name 'seed_*_attempt_*' -print0 | sort -z)

# ==================== 数量校验 ====================
if (( ${#episode_dirs[@]} != EXPECTED_EPISODES )); then
  echo "错误: 预期 ${EXPECTED_EPISODES} 个 episode, 实际找到 ${#episode_dirs[@]}"
  echo "先诊断 attempt 分布:"
  echo "  find $TASK_DIR -mindepth 1 -maxdepth 1 -type d | sed 's/.*_attempt_/attempt=/' | sort | uniq -c"
  exit 1
fi

echo "找到 ${#episode_dirs[@]} 个 ${TASK_UPPER} episode, 开始转换……"

# ==================== 输出目录预检 ====================
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "错误: 输出目录已存在, 请删除后重试: $OUTPUT_ROOT"
  exit 1
fi

# ==================== 转换 ====================
episode_args=()
for episode_dir in "${episode_dirs[@]}"; do
  episode_args+=(--episode-dir "$episode_dir")
done

# 默认仅转换 RGB; 需要深度图时设置 DEPTH=1
DEPTH="${DEPTH:-0}"
depth_arg=()
if (( DEPTH )); then
  depth_arg=(--depth)
fi

if [[ -z "${MEMBENCH_LEROBOT_SRC:-}" ]]; then
  if [[ -d "$PWD/policy/lerobot-0.6.1/src/lerobot" ]]; then
    MEMBENCH_LEROBOT_SRC="$PWD/policy/lerobot-0.6.1/src"
  else
    MEMBENCH_LEROBOT_SRC="$("$PYTHON" -c 'import pathlib, lerobot; print(pathlib.Path(lerobot.__file__).resolve().parent.parent)')"
  fi
fi
export MEMBENCH_LEROBOT_SRC

"$PYTHON" scripts/record/convert_bench_episodes_to_lerobot_v061.py \
  "${episode_args[@]}" \
  --output-root "$OUTPUT_ROOT" \
  --repo-id "$REPO_ID" \
  --fps 20 \
  --robot-type mobile_manipulator \
  --write-mode create \
  "${depth_arg[@]}"

echo "完成: $OUTPUT_ROOT"
