# rPPG 4p_v2 数据真实性二次审计（2026-04-23）

## 审计目标
对 `/Users/liangwenwang/Downloads/rppg_精度向上開発_報告書_4p_v2.html` 中关键声明进行独立复算核验，确认是否与原始评估 CSV 一致。

## 输入数据（只读）
- `/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/data2_as_data1_eval_20260423_rerun/rppg_ecg_summary_best_opencv_timestamp.csv`
- `/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/data2_as_data1_eval_20260423_rerun/rppg_ecg_summary_published_opencv_timestamp.csv`
- `/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/data2_as_data1_eval_20260423_rerun/rppg_ecg_comparison_best_opencv_timestamp.csv`
- `/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/data2_as_data1_eval_20260423_rerun/rppg_ecg_comparison_published_opencv_timestamp.csv`
- `/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/visualizations/fc_threshold_report.html`

## 审计方法（独立复算）
1. 从 4p_v2 报告中提取声明值（KPI、设备级、Data1 参考值、9/11 说明）。
2. 使用逐秒 `comparison_*.csv` 直接重算 MAE/RMSE/MAPE/Accuracy/n/corr/bias（不信任现成 summary 结论）。
3. 将重算结果与 4p_v2 声明做逐项容差比对。
4. 做第二层一致性验证：
   - `comparison_*.csv` 聚合结果 vs `summary_*.csv` 加权聚合结果。

## 核心结果
- 报告口径（Data2: g102+g103）：
  - best: n=842, MAE=3.7680, RMSE=5.6823, MAPE=5.2875%
  - published: n=599, MAE=2.3530, RMSE=3.3087, MAPE=3.2810%, Accuracy=96.7190%
  - coverage=71.1401%
- 全量口径（Data2: g101+g102+g103）：
  - best: n=1211, MAE=7.7989, RMSE=12.2093, MAPE=10.8421%
  - published: n=673, MAE=3.0484, RMSE=4.5973, MAPE=4.2148%, Accuracy=95.7852%
  - coverage=55.5740%

## 审计结论
- 4p_v2 报告的关键声明与原始数据复算一致。
- 校验通过：**39 / 39**（包含“声明核验 + 源数据链路一致性核验”）。
- 未发现伪造或与原始 CSV 不一致的情况。

## 逐项检查结果
- CSV 明细：`/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/audits/report_4p_v2_truth_checks_20260423.csv`

## 可视化审计页面
- `/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/visualizations/report_4p_v2_truth_audit.html`

## 复现命令
```bash
/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/.venv-desktop/bin/python \
  /Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/scripts/audit_report_4p_v2.py
```
