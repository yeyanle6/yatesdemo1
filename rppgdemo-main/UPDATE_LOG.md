# rPPG 项目更新记录（截至 2026-03-04）

本文件用于记录本次迭代中每一步“改了什么 / 如何改 / 结果如何”，便于复现与审计。

## 2026-04-21 迭代补充（MAE<3 目标）

- 修改点 1（发布门控，可配置）：
  - 文件：`main.py`, `evaluate_dataset.py`
  - 新增参数：
    - `publish_min_freq_conf_for_output`
    - `publish_min_sqi_for_output`
  - 新增 CLI：
    - `--publish-min-freq-conf`
    - `--publish-min-sqi`
  - 行为：仅在满足门控阈值时才写入 `hr_published`（用于 `--use-published` 评估口径）。

- 修改点 2（Adaptive ROI 保护门）：
  - 文件：`main.py`, `evaluate_dataset.py`
  - `AdaptiveROIController.update(...)` 增加 `sqi` 输入；
  - `_lock_detected()` 增加 `sqi_med <= adaptive_roi_max_sqi_for_lock` 约束，降低误触发概率。

- 修改点 3（文档）：
  - 文件：`README.md`
  - 新增“高精度门控模式”复现命令与覆盖率权衡说明。

- 实测结果（opencv, timestamp, 35 视频全量）：
  - baseline（best 口径）：`HR_MAE=4.229`, `n=4032`
  - 高精度门控（`--use-published --publish-min-freq-conf 0.9`）：
    - `HR_MAE=2.685`, `HR_RMSE=3.469`, `HR_corr=0.964`, `n=2232`
  - 结论：达成 `MAE<3`，但有效点数下降（覆盖率约 `55.4%`）。

## 2026-04-22 迭代补充（双通道导出）

- 修改点 1（离线评估双通道导出）：
  - 文件：`evaluate_dataset.py`
  - 新增 CLI：`--export-dual-channel`
  - 行为：一次运行同时导出 `best` 与 `published` 两套 detail/summary CSV。

- 修改点 2（输出字段标识）：
  - 文件：`evaluate_dataset.py`
  - detail/summary 新增 `export_channel` 字段，用于区分 `best` 与 `published`。

- 验证结果（diag_3）：
  - 同一次运行已生成：
    - `rppg_ecg_comparison_best_opencv_timestamp.csv`
    - `rppg_ecg_summary_best_opencv_timestamp.csv`
    - `rppg_ecg_comparison_published_opencv_timestamp.csv`
    - `rppg_ecg_summary_published_opencv_timestamp.csv`

## 1. 目标与评估口径

- 目标：提升视频端 rPPG 心率估计精度（相对 ECG）。
- 数据：`/Users/liangwenwang/Downloads/Code/demo1/data/001,002,003`
- 评估脚本：`/Users/liangwenwang/Downloads/Code/demo1/rpi_rppg/evaluate_dataset.py`
- 核心指标：`MAE / RMSE / MAPE / corr`
- 当前最终（全量，opencv）：`MAE=6.9403, RMSE=12.3881, MAPE=7.4435%, corr=0.8196`

## 2. 分阶段更新时间线

### 阶段 A：自动闭环框架搭建

- 更新内容：
  - 新增自动优化脚本：`auto_optimize.py`
  - 支持循环执行：调参 -> 评估 -> 分析 -> 下一轮决策
  - 输出 `history.jsonl / decisions.jsonl / REPORT.md / best_config.json`
- 如何更新：
  - 在现有 `evaluate_dataset.py` 基础上复用评估流程，增加 objective 与决策逻辑。
- 结果：
  - 运行目录：`results/auto_opt/20260304_100937`
  - 最优（iter04）：`MAE=10.9809, RMSE=16.5340`

### 阶段 B：频率时序融合 + 频率置信门控

- 更新内容：
  - 新增 `harmonic_temporal_fusion`（候选簇 + 连续性约束）。
  - 新增 `frequency_confidence` 并进入质量门控。
  - 线上/离线统一接入 `freq_conf`。
- 如何更新：
  - 文件：`main.py`, `evaluate_dataset.py`, `auto_optimize.py`
  - 在融合后增加 `combined_quality = 0.7*SQI + 0.3*FCONF`。
- 结果：
  - 运行目录：`results/auto_opt/20260304_105139`
  - 最优（iter04）：`MAE=9.9378, RMSE=16.0002`

### 阶段 C：生理与信号细节修正

- 更新内容：
  - 心率生理上限：`clamp_max_bpm 120 -> 160`。
  - 修复子谐波惩罚阈值不可触发问题（0.5f 分支）。
  - `pNN50` 改为 `pNN50(exp)`，增加最小窗口/可靠性约束。
- 如何更新：
  - 文件：`main.py`, `evaluate_dataset.py`, `README.md`
  - CSV 增加 `pnn50_reliable` 字段。
- 结果：
  - 运行目录：`results/auto_opt/20260304_123510`
  - 最优（iter01）：`MAE=9.5868, RMSE=15.3999`

### 阶段 D：双域门控（频域 HR + 时域 PPI）增强

- 更新内容：
  - 在 `apply_ppi_assist` 中加入“规则门控 + 强纠偏”策略。
  - 在高置信触发条件下更强地向 `PPI_HR` 拉拢。
- 如何更新：
  - 文件：`main.py`, `evaluate_dataset.py`
  - 保持历史跃迁限制，避免实时抖动。
- 结果：
  - 运行目录：`results/auto_opt/20260304_130134`
  - 最优（iter01）：`MAE=8.9050, RMSE=14.4716`

### 阶段 E：时频参数再优化

- 更新内容：
  - 自动探索得到更优时频参数：
  - `welch_overlap=0.8`, `pos_window_sec=1.8`。
- 如何更新：
  - 闭环运行：`results/auto_opt/20260304_132939`
  - 最优参数回写 `Config` 默认值。
- 结果：
  - 最优（iter03）：`MAE=8.3425, RMSE=14.1640`
  - 相比阶段 C（9.5868）提升：`12.98%`

### 阶段 F：ROI 模板系统与对照实验

- 更新内容：
  - 新增 ROI 模板系统：
  - `classic4 / hybrid7 / cheek_forehead6 / whole_face`
  - 新增 ROI 形变参数：`roi_scale_x / roi_scale_y / roi_shift_y`
  - 三种提取器统一走同一 ROI 构造逻辑（mediapipe solutions/tasks/opencv）。
- 如何更新：
  - 文件：`main.py`, `evaluate_dataset.py`, `auto_optimize.py`
  - 额外生成对照表：`roi_preset_benchmark_20260304_1437.csv`
- 结果（模板对照）：
  - `classic4`: `MAE=8.3425`
  - `hybrid7`: `MAE=7.6491`（最佳）
  - `whole_face`: `MAE=7.7263`
  - `cheek_forehead6`: `MAE=8.6606`

### 阶段 G：ROI 尺度网格搜索（hybrid7）

- 更新内容：
  - 对 `hybrid7` 做尺度与垂直位移搜索（x/y 缩放、上移偏移）。
- 如何更新：
  - 离线网格组合试验后固定最优：
  - `roi_scale_x=1.1`, `roi_scale_y=1.1`, `roi_shift_y=0.0`
  - 回写默认参数到 `Config`。
- 结果（最终）：
  - `MAE=6.9403, RMSE=12.3881, MAPE=7.4435%, corr=0.8196`
  - 结果文件：
  - `rppg_ecg_comparison_best_opencv_roi_hybrid7_large_final_20260304_152726.csv`
  - `rppg_ecg_summary_best_opencv_roi_hybrid7_large_final_20260304_152726.csv`

## 3. 当前默认配置（代码已生效）

来自 `Config()`（`main.py`）：

- `low_hz=0.6`
- `high_hz=4.75`
- `welch_seg_sec=4.5`
- `welch_overlap=0.8`
- `pos_window_sec=1.8`
- `buffer_sec=11.0`
- `clamp_max_bpm=160.0`
- `enable_ppi_assist=True`
- `roi_preset='hybrid7'`
- `roi_scale_x=1.1`
- `roi_scale_y=1.1`
- `roi_shift_y=0.0`

## 4. 增益汇总（关键基线对比）

- 相比阶段 A 最优（`MAE=10.9809`）：
  - 当前 `MAE=6.9403`，提升 `36.80%`
- 相比阶段 C（`MAE=9.5868`）：
  - 当前提升 `27.61%`
- 相比阶段 E（`MAE=8.3425`）：
  - 当前提升 `16.81%`

## 5. 可视化与实时能力新增

- 离线处理过程视频：
  - 脚本：`visualize_pipeline.py`
  - 示例输出：`results/vis_1-1_demo.mp4`
- 实时 Qt 界面：
  - 脚本：`realtime_qt.py`
  - 支持摄像头实时 ROI + 指标 + 曲线监控

## 6. 参考方法（论文与开源）

- CHROM: de Haan & Jeanne, 2013  
  https://pubmed.ncbi.nlm.nih.gov/23744659/
- POS: Wang et al., 2017  
  https://pubmed.ncbi.nlm.nih.gov/28113245/
- ROI 影响研究：  
  https://pubmed.ncbi.nlm.nih.gov/34077596/
- pyVHR:  
  https://github.com/phuselab/pyVHR
- rPPG-Toolbox:  
  https://github.com/ubicomplab/rPPG-Toolbox

## 7. 复现命令（当前最优）

```bash
cd /Users/liangwenwang/Downloads/Code/demo1/rpi_rppg
python evaluate_dataset.py \
  --data-dir /Users/liangwenwang/Downloads/Code/demo1/data \
  --out-dir /Users/liangwenwang/Downloads/Code/demo1/rpi_rppg/results \
  --roi-mode opencv \
  --roi-preset hybrid7 \
  --roi-scale-x 1.1 \
  --roi-scale-y 1.1
```
