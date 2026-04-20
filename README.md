# Raspberry Pi rPPG (from your iOS pipeline)

这个目录是你 iOS rPPG 项目在树莓派上的 Python 迁移版本，包含：
- D455 / 普通摄像头采集
- 4 ROI（forehead / glabella / right_malar / left_malar）
- GREEN + CHROM + POS
- harmonic-aware 融合 + 生理约束 + tail-stable best HR
- PPI 峰值检测 + `pNN50`
- SQI 三态机（GOOD/MARGINAL/BAD）+ 发布节流（GOOD/MARGINAL 1Hz, BAD 0.5Hz）

## 1) 安装

```bash
cd /Users/liangwenwang/Downloads/Code/demo1/rpi_rppg
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果你在 Raspberry Pi OS 上使用 D455，需要先安装 librealsense（`pyrealsense2` 依赖它）。

更推荐按环境分开：
- Desktop 验证：`requirements-desktop.txt`
- Raspberry Pi 部署：`requirements-rpi.txt`
- 详细步骤见 [ENVIRONMENTS.md](/Users/liangwenwang/Downloads/Code/demo1/rpi_rppg/ENVIRONMENTS.md)

## 2) 运行

自动选择相机（优先 D455，失败回退 webcam）：

```bash
python main.py --source auto --fps 60 --roi-mode auto --show
```

推荐显式指定当前最佳 ROI 方案：

```bash
python main.py --source auto --fps 60 --roi-mode opencv --roi-preset hybrid7 --roi-scale-x 1.1 --roi-scale-y 1.1 --show
```

强制 D455：

```bash
python main.py --source realsense --fps 60 --roi-mode mediapipe --show
```

强制普通摄像头：

```bash
python main.py --source webcam --roi-mode opencv --show
```

`--roi-mode` 说明：
- `mediapipe`: 使用 Face Mesh（精度更高，速度更慢）
- `opencv`: 不依赖 mediapipe，使用 OpenCV 人脸框推导 ROI（树莓派更稳妥）
- `auto`: 优先 mediapipe，失败自动回退 opencv

`--roi-preset` 说明：
- `hybrid7`（当前默认，综合精度最好）
- `classic4`（iOS 对齐模板）
- `cheek_forehead6`
- `whole_face`

Python 3.12 注意：
- 某些 `mediapipe` 发行版是 `tasks-only`（没有 `mp.solutions`）。
- 这时 `--roi-mode mediapipe` 需要额外传模型路径：

```bash
python main.py \
  --source webcam \
  --roi-mode mediapipe \
  --mp-face-detector-model /path/to/face_detection_short_range.tflite
```

如果不传模型，程序会在 `auto` 模式下自动回退 `opencv`。

终端每秒输出：
- `HR`: 融合后的 best HR（BPM）
- `PPI_HR`: 来自峰间期平均的 HR（BPM）
- `pNN50(exp)`: 实验性 rPPG 衍生比例（0~1）和百分比

## 3) 与 iOS 口径对齐（当前版本）

已对齐的关键点：
- bandpass: `0.6-4.75 Hz`（当前自动优化后默认）
- Welch: `4.5s`, `80% overlap`, `nfft=2048`（当前自动优化后默认）
- POS window: `1.8s`（当前自动优化后默认）
- ROI 像素过滤: `55~220`, 饱和 `<245`, valid ratio `>=60%`
- 融合: `45~170 BPM` 入池，谐波双峰判定 `gap>=15`, `ratio 1.6~2.4`
- Welch 候选评分: `physioScore * power` + 谐波证据重排
- 生理约束: `clamp 40~160` + 历史偏差限制 + 变化率限制 + 首次先验
- Best HR: 尾部稳定段 + 15% trim median
- 峰值检测/缺峰插值/PPI 异常值修正：按 iOS 同逻辑移植
- `pNN50(exp)`: `|NN_i - NN_{i-1}| > 50 ms` 的比例（仅在较长窗口下输出）

## 4) 注意

- `pNN50(exp)` 需要更长稳定窗口（当前实现至少约 20 秒）才有统计稳定性，不能直接等价 ECG-HRV 指标。
- 如果你只能跑 30fps，HR 可以工作，但 `pNN50` 时间分辨率会明显下降；建议 D455 用 `640x480@60fps`。
- 仍无法做到 100% 位级一致的部分：ROI 几何来源（iOS Vision vs Python MediaPipe）和相机 ISP/曝光链路差异。

## 5) 离线评估（视频 vs ECG）

脚本：`/Users/liangwenwang/Downloads/Code/demo1/rpi_rppg/evaluate_dataset.py`

一次跑两种 ROI（推荐）：

```bash
python evaluate_dataset.py \
  --data-dir /Users/liangwenwang/Downloads/Code/demo1/data \
  --out-dir /Users/liangwenwang/Downloads/Code/demo1/rpi_rppg/results \
  --roi-mode both \
  --mp-face-detector-model /path/to/face_detection_short_range.tflite
```

只跑一种：

```bash
python evaluate_dataset.py --roi-mode mediapipe
python evaluate_dataset.py --roi-mode opencv
```

输出文件名会带上 ROI 模式后缀，例如：
- `rppg_ecg_comparison_best_opencv.csv`
- `rppg_ecg_summary_best_opencv.csv`
- `rppg_ecg_comparison_best_mediapipe.csv`
- `rppg_ecg_summary_best_mediapipe.csv`

## 6) Qt 结果分析界面

程序：`/Users/liangwenwang/Downloads/Code/demo1/rpi_rppg/result_analyzer_qt.py`

安装 GUI 依赖：

```bash
cd /Users/liangwenwang/Downloads/Code/demo1/rpi_rppg
pip install -r requirements-gui.txt
```

启动（默认读取 `rpi_rppg/results`）：

```bash
python result_analyzer_qt.py
```

或指定结果目录：

```bash
python result_analyzer_qt.py /Users/liangwenwang/Downloads/Code/demo1/rpi_rppg/results
```

功能：
- 选择 summary CSV（自动匹配对应 detail CSV）
- 按 `group`、`ROI used`、关键词筛选样本
- 选择样本后显示 `ECG HR vs rPPG HR` 时序曲线和散点图
- 显示该样本 `MAE/RMSE/MAPE/Corr` 与状态分布
- 导出当前筛选后的 summary CSV

## 7) 自动化迭代优化（闭环）

脚本：`/Users/liangwenwang/Downloads/Code/demo1/rpi_rppg/auto_optimize.py`

能力：
- 每轮自动执行：调参 -> 运行评估 -> 分析误差 -> 生成下一轮参数
- 记录每次改动、指标、决策理由
- 输出最佳参数和迭代报告

全量模式（较慢）：

```bash
python auto_optimize.py \
  --data-dir /Users/liangwenwang/Downloads/Code/demo1/data \
  --out-dir /Users/liangwenwang/Downloads/Code/demo1/rpi_rppg/results/auto_opt \
  --roi-mode opencv \
  --iterations 4
```

快速模式（先验证流程）：

```bash
python auto_optimize.py \
  --roi-mode opencv \
  --iterations 2 \
  --groups 003 \
  --max-samples-per-group 2
```

每次运行会在 `results/auto_opt/<timestamp>/` 生成：
- `iter_XX_detail.csv`
- `iter_XX_summary.json`
- `history.jsonl`
- `decisions.jsonl`
- `best_config.json`
- `REPORT.md`

## 8) 处理过程可视化视频

脚本：`/Users/liangwenwang/Downloads/Code/demo1/rpi_rppg/visualize_pipeline.py`

功能：
- 原视频逐帧叠加 ROI 框（名称+权重）
- 右侧面板显示实时 HR / PPI_HR / SQI / FCONF / state
- 叠加绿色通道信号趋势与 HR 趋势曲线
- 可选叠加 ECG HR（提供 `--ecg-csv`）

示例：

```bash
python visualize_pipeline.py \
  --input-video /Users/liangwenwang/Downloads/Code/demo1/data/001/video/1-1.MOV \
  --ecg-csv /Users/liangwenwang/Downloads/Code/demo1/data/001/csvdata/1-1.csv \
  --output-video /Users/liangwenwang/Downloads/Code/demo1/rpi_rppg/results/vis_1-1.mp4 \
  --roi-mode opencv \
  --roi-preset hybrid7 \
  --roi-scale-x 1.1 \
  --roi-scale-y 1.1
```

## 9) 实时 Qt 监测界面（摄像头）

脚本：`/Users/liangwenwang/Downloads/Code/demo1/rpi_rppg/realtime_qt.py`

功能：
- 实时摄像头画面 + ROI 框可视化
- 实时指标：HR / PPI_HR / pNN50(exp) / SQI / FCONF / SNR / 状态
- 实时曲线：混合 Green 信号与 HR 趋势

启动：

```bash
python realtime_qt.py \
  --source auto \
  --fps 60 \
  --roi-mode opencv \
  --roi-preset hybrid7 \
  --roi-scale-x 1.1 \
  --roi-scale-y 1.1
```
