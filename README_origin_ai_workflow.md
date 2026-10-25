# Origin / Adobe Illustrator 科研绘图数据包

这个包的目标是把模型输出转换成适合 Origin、Adobe Illustrator、PPT 后期处理的标准数据文件。

## 1. 先生成 Origin/AI 数据包

把本文件夹内的 `prepare_origin_ai_data.py` 放到项目根目录：

```text
C:\Users\19735\Documents\Codex\2026-07-04\n-s
```

然后运行：

```cmd
python -m py_compile prepare_origin_ai_data.py

python prepare_origin_ai_data.py ^
  --strict-dirs outputs_v3_probe_sumfocus_s42 outputs_v3_probe_sumfocus_s7 outputs_v3_probe_sumfocus_s123 outputs_v3_probe_sumfocus_s2026 outputs_v3_probe_sumfocus_s3407 outputs_v3_probe_clsboost_s42 outputs_v3_probe_clsboost_s7 outputs_v3_probe_clsboost_s123 outputs_v3_probe_clsboost_s2026 outputs_v3_probe_clsboost_s3407 outputs_v3_dualhead_sumfocus_s42 outputs_v3_dualhead_sumfocus_s7 outputs_v3_dualhead_sumfocus_s123 outputs_v3_dualhead_sumfocus_s2026 outputs_v3_dualhead_sumfocus_s3407 ^
  --strict-top-k 7 ^
  --strict-weighting softmax ^
  --strict-temp 0.05 ^
  --update12-dirs outputs_v3_dualhead_update12h_s42 outputs_v3_dualhead_update12h_s7 outputs_v3_dualhead_update12h_s123 outputs_v3_dualhead_update12h_s2026 outputs_v3_dualhead_update12h_s3407 ^
  --update12-weighting equal ^
  --split stratified_random_9_1 ^
  --mode ral_val ^
  --out-dir origin_data
```

## 2. Origin 中怎么使用

### 分类指标柱状图

导入：

```text
origin_data/01_classification_metrics_long.csv
```

建议作图方式：

```text
Grouped Column / Grouped Bar
X = Metric
Y = Score
Group = Experiment
```

### 回归 MAE 对比柱状图

导入：

```text
origin_data/02_regression_mae_long.csv
```

建议作图方式：

```text
Grouped Column / Grouped Bar
X = Target
Y = MAE
Group = Experiment
```

### 高风险 ROC 曲线

导入：

```text
origin_data/04_highrisk_roc_curve.csv
```

建议作图方式：

```text
Line plot
X = FPR
Y = TPR
Group = Experiment
```

### 回归散点图

导入，例如：

```text
origin_data/09_scatter_peak_log.csv
origin_data/10_scatter_sum_log.csv
origin_data/11_scatter_mean_log.csv
origin_data/12_scatter_count_high_log.csv
```

建议作图方式：

```text
Scatter
X = true
Y1 = strict_pred
Y2 = update12_pred
添加 y = x 参考线
```

### 残差分布图

导入：

```text
origin_data/13_residual_peak_log.csv
origin_data/14_residual_sum_log.csv
origin_data/15_residual_mean_log.csv
origin_data/16_residual_count_high_log.csv
```

建议作图方式：

```text
Histogram
Data = strict_residual / update12_residual
添加 x = 0 参考线
```

## 3. Adobe Illustrator 中怎么使用

推荐流程：

```text
Origin 绘图 → 导出 SVG/PDF → Illustrator 打开 → 统一字体、线宽、子图编号 → 导出最终 PDF/TIFF
```

AI 中建议统一：

```text
字体：Times New Roman 或 Arial
坐标轴字号：8–9 pt
标题字号：9–10 pt
线宽：0.75–1.25 pt
子图编号：(a), (b), (c)
输出：PDF/SVG/TIFF 600 dpi
```

## 4. PPTX 形状版怎么用

混淆矩阵、流程图、模型结构图、简单柱状图适合 PPTX 原生形状版。
ROC、散点图、残差图不建议完全转为 PPT 形状，因为数据点太多，PPT 会卡。

## 5. 推荐最终保留文件

```text
origin_data/       # 所有作图数据
origin_project/    # Origin 工程文件，手动保存
ai_figures/        # AI 精修版
final_figures/     # 投稿用 PDF/SVG/TIFF
ppt_editable/      # PPT 形状版工作稿
```
