# WR-11：跨区域按样取物

机器人先在餐台观察样品牛奶盒，保持朝向平移到岛台另一端的取货区，从三盒候选中首次选中相同包装的一盒，再运回样品旁边的空托盘。样品始终留在原处，场景不关闭柜门、不隐藏物体；横向移动使样品离开相机视野。

使用 `milk_1` 的固定网格和碰撞几何，保留包装白色图案，颜色池为红、橙、黄、绿、青、蓝、紫、黑八种外观。样品和候选使用独立纹理，匹配依据为包装颜色。每次放置三盒不同色候选，其中恰好一盒与样品同色。

## 当前房间

任务版本为 `3.2.0`，采样 profile 为 `WR-11/sampling-v2`。使用布局 **33**、风格 **34**、场景种子 **0**：米白色橱柜、灰色瓷砖地板，以及带水槽的长岛台。样品和送回托盘位于岛台左端，三个候选位于同一岛台右端；两个导航站位横向相距约 2.3 m，朝向相同。机器人向右平移取货，再向左平移送回。

托盘高度根据各自台面和托盘底部计算，站位、物体位置和评测区域均已重新校准。此前布局 32 和转身版本的录制保留用于归档，不能当作当前版本的数据；回放会检查版本及 episode identity。当前评测使用 `configs/eval/suites/wr11_v3.yaml`。

## 记忆与随机化

记忆决策发生在取货台的第一次抓取：当前图像包含候选，却不包含餐台上的样品。候选随机流从八色池无放回地抽取三色并安排到左、中、右，形成 336 种等概率排列；独立的请求随机流在三色中等概率选择样品。因此样品每色边际概率为 1/8，给定候选排列时每盒成为目标的概率为 1/3，共有 1008 种样品与排列组合。样品和候选颜色因必须存在匹配而有关联；物体姿态与站位使用独立随机流，不依据正确候选生成。

样品和三盒候选各自在托盘中心半径 2.5 cm 的圆内按面积均匀采样，角度在 −180° 至 180° 内均匀采样。两个站位各自在半径 5 cm 的圆内按面积均匀采样，保持共同朝向。所有采样在 episode 解析时确定，初始、观察和返回复用同一个样品站位。

参数、坐标、采样约束和分布图见 [正式采样规则](../../../docs/wr11_sampling.md)。物体半径由 `resources/sampling/regions.json` 管理，机器人半径由 `resources/calibration/stations.json` 管理；重复声明半径或扩大到未校准范围会报错。

固定 prompt 不包含目标颜色、候选位置或正确物体 ID。私有 episode facts 和诊断可以保存这些信息；策略输入继续使用统一的三路 RGB、37 维机器人状态和固定 prompt。当前仅支持 `memory` variant。

当前整体指令：`Observe the color of the reference carton, then fetch a carton of the same color and place it in the empty tray.`
即观察目标参考盒的颜色，再拿取同色盒子放入空置盘。

## 物理判定

- 在餐台区域，至少一路相机能以无遮挡射线看到样品，持续三个连续控制步，记为 `sample_observed`。
- 到达取货台且样品的全部可见几何完整离开三路相机视锥，记为 `stock_reached`。未知相机或可见性数据不能满足这个条件。
- 抓取要求夹爪闭合、真实接触/持有且抬高至少 2 cm；滑动或掉落不能构成选择。第一次抓起错误候选立即以 `wrong_first_pick` 结束，换货不能恢复。正确选择时仍必须位于取货台，且样品不可见。
- 正确盒子必须从夹爪释放，并与空托盘真实接触，持续三个连续控制步才完成送回。仅靠接近托盘、持有盒子悬停或未抓取就接触托盘不能成功。
- 样品偏离评测起点超过 6 cm，或正确选择后又抓起另一个候选，均结束本次任务。
- 同一控制步的重复读取不累计持续时间，跳步会重置持续计数。成功和失败判定冻结，策略运行时收到 `stop_execution` 后拒绝继续动作。

录制与策略使用同一套物理评测，不依赖专家的颜色记忆回调。`WR-11/report-v1` 的公共阶段为 `sample_observed → stock_reached → first_pick_correct → item_delivered → success`，另保留首次选择是否正确及其时间。

专家先比较完整抓取链的可达性，优先选择沿盒子原生 Y 轴的相对 0°/180° 接近方向，然后比较路径长度。空手和携带时均保持出发朝向，直接平移至目标区域；不安排转身阶段。携带完成后检查真实持有和抬升证据。策略执行不受这些专家动作选择约束。运行中不写物体自由关节，不使用抓取附着或搬运辅助。

专家抓取 profile `WR-11/grasp-v2` 保留原有 6 cm 预抓取、12 cm 抬升路径；仅当四个方向的完整路径都不可达时，增加 4 cm 预抓取、8 cm 抬升的后备路径，并重新验证预抓取、抓取、抬升三个阶段。此规则适用于所有 seed，不改颜色、物体姿态、机器人站位或物理成功标准。选用的 profile、高度和方向写入 `planner.json`。

## 运行

在仓库根目录激活仿真环境：

```bash
source scripts/tools/activate_sim_env.sh
python scripts/record_entrypoint.py --task-id WR11 --seeds 0 \
  --dataset-id wr11_smoke --output-root outputs/datasets --no-record-depth
```

包含每阶段截图和可见性证据的验证入口：

```bash
.venv-robocasa/bin/python scripts/planning/validate_wr11.py \
  --seed 0 --output outputs/diagnostics/wr11/new_seed0
```

该入口通过生产 `RecordingRunner` 保存 `episode/` 下的原生动作、三路视频和录制结果，并在 `validation.json` 中保存物理评测与视野证据。添加 `--scene-only` 可检查两个站位的画面；此模式通过重设机器人位置检查视野，只属于场景诊断，不能作为成功轨迹或数据集。

正式评测配置为 `configs/eval/suites/wr11_v3.yaml`。录制通过后仍需用 `scripts/eval/replay_recorded_actions.py` 在全新策略运行时回放动作；专家录制成功本身不代表动作回放成功。当前房间的验证结果见 `docs/wr11_validation.md`。

200 seeds 的 final 数据选择、LeRobot 转换、校验及训练/评测入口见
[WR-11 final 数据说明](../../../docs/wr11_final.md)。

单个验证 episode 的回放也可以使用：

```bash
.venv-robocasa/bin/python scripts/planning/validate_wr11.py --seed 0 \
  --replay-episode outputs/diagnostics/wr11/new_seed0/episode \
  --output outputs/diagnostics/wr11/new_seed0_replay
```

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-robocasa/bin/python -m pytest -q \
  tests/test_wr11_task.py tests/test_wr11_evaluation.py \
  tests/simulator/test_wr11_visibility.py
```
