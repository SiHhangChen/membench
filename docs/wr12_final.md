# WR12 LeRobot 数据转换

使用与 WR11 相同的 LeRobot 0.6.1 转换器生成 v3.0 数据，输出
`data/final/wr12-final`，repo ID 为 `wr12-final`。

输入为 `outputs/datasets/wr12_200success_v14_complete`，任务版本 `1.4.0`，
观察 1 秒后自动封口。选取 seeds 0–69、71–200 的 200 条成功轨迹，
左右目标各 100 条，共 131,328 帧。seed 200 为补录，seed 70 的原失败录像保留。

格式与已有 final 任务对齐：20 FPS、37 维状态、13 维动作，三路 224×224 RGB，
不导出深度；保留相机内外参和成功/终止标记。每个 episode 一份数据 Parquet、
一份 episode 元数据 Parquet、三段视频，`meta/seeds.jsonl` 保留真实 seed 到
0–199 episode index 的映射。

转换命令：

```bash
PYTHON="$PWD/.venv-gr00t/bin/python" \
  bash scripts/record/convert_tasks_to_lerobot_v061.sh WR-12 200 \
  outputs/datasets/wr12_200success_v14_complete \
  data/final/wr12-final wr12-final
```

转换在临时目录完成，内部校验通过才安装正式目录；拒绝覆盖已有输出。
本次按 TM05 已验证的方式使用 8 路分片转换，每片 25 条，随后按真实 seed 合并、
重建统计并校验；底层仍为上述通用转换器。执行入口为
`outputs/conversion/wr12-final/run_conversion.py`。
本次工作流与日志位于 `outputs/conversion/wr12-final/`，`summary.json` 记录真实结果。
工作流使用工作区缓存，避免默认 Hugging Face 缓存目录的写入权限问题。

独立校验沿用 WR11：

```bash
.venv-gr00t/bin/python scripts/train/validate_lerobot_v3_dataset.py \
  data/final/wr12-final --repo-id wr12-final --expected-episodes 200 \
  --expected-fps 20 --expected-state-dim 37 --expected-action-dim 13 \
  --expected-cameras 3 --action-horizon 100

.venv-gr00t/bin/python scripts/record/audit_final_dataset.py \
  --dataset data/final/wr12-final \
  --source outputs/datasets/wr12_200success_v14_complete \
  --expected-episodes 200 --write-provenance \
  --output-report outputs/conversion/wr12-final/native_audit.json
```

独立校验覆盖全部 Parquet、600 个视频容器、实际 LeRobot 三路视频读取与
100 步动作读取；原生对照逐值检查全部动作、状态、相机矩阵、终止与成功标记、语言指令。
通过后 `meta/membench_provenance.json` 保存原始数据来源、seed、attempt、版本和 identity。

## 完成结果

200 条、131,328 帧已转换并通过全部独立校验；数据入口为 `data/final/wr12-final`。
全部 600 段视频的容器帧数、尺寸和 FPS 检查通过；LeRobot 三路视频读取、
100 步动作序列读取通过；全部原生数值列与语言指令逐值一致。

转换进程在完成写入和内部校验后出现与 WR11 相同的原生库退出清理异常（exit -6）。
独立校验发现 seed 185 腕部视频少一帧（558/559）；从完整原始视频重新编码后，
确认 559 帧并重新通过全量校验。修复记录 `video_repair.json`，初始文件与日志保留。
最终结果见 `outputs/conversion/wr12-final/summary.json`、`validate_repaired.log` 与
`native_audit.json`。此任务仅转换与校验，未执行训练或上传。

## 原始 source ZIP

按 TM05 的完整原始数据包结构导出，入口为 `outputs/datasets/wr12-source.zip`，
实际文件为 `outputs/datasets/source-zips/wr12-source.zip`，解压根目录为 `wr12-source/`。

- `train/WR-12/`：200 条成功原始轨迹，每条含 `episode.json`、`planner.json`、
  `events.jsonl`、`trajectory.h5`、三路原始 RGB 视频。
- `manifest.json`、`final_seeds.jsonl`：可移植的相对路径，以及真实 seed 与 final episode index 映射。
- `recording_provenance/`：原始 episode 元数据、录制时源码和任务配置快照、
  原始 200-seed 批次与 seed 200 补录/最终选择清单、转换和导出脚本。
- `validation/`：原生与 final 精确对照、转换汇总、LeRobot 读取结果、seed 185 视频修复说明。
- `README.md`、`package_info.json`、`recording_summary.json`、`file_inventory.json`：
  包说明、最终成功汇总及每个 payload 文件的大小和 SHA256。

HDF5、原始视频、planner/events 原样保存；只规范导出 episode 元数据中的文件路径，
原始 episode.json 另存。seed 70 的失败轨迹和转换缓存不进入训练 source 包；
历史录制清单中保留其失败记录。WR12 尚未制作阶段边界或语义标注，包内明确说明。
外部仿真资产与 Python 环境为仓库依赖，不随包重复分发。

打包脚本：`scripts/record/package_wr12_source.py`。
完成报告：`outputs/conversion/wr12-final/source_zip_report.json`。
打包验证包括 ZIP CRC、每个归档条目的 SHA256、原始文件未改动和录制快照哈希检查。

## ModelScope 上传

按用户指定上传到 `ouhebai/wr07-final`，根目录文件为 `wr12-final.zip` 与 `wr12-source.zip`。
两者均完成上传，远端大小和 SHA256 与本地校验包一致。
LeRobot ZIP：1,308,612,274 bytes；source ZIP：277,496,390 bytes。
上传脚本为 `scripts/record/upload_wr12_archives_modelscope.py`，报告为
`outputs/conversion/wr12-final/modelscope_upload/upload_report.json`。
