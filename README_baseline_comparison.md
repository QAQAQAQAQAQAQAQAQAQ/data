# 基线模型对比实验包

这个包用于补齐论文中最重要的“与其他模型对比”实验。

## 1. 文件

```text
run_v3_baseline_comparison.py
```

## 2. 严格预测任务：传统机器学习基线

把脚本放到项目根目录：

```text
C:\Users\19735\Documents\Codex\2026-07-04\n-s
```

运行：

```cmd
python -m py_compile run_v3_baseline_comparison.py

python run_v3_baseline_comparison.py ^
  --data camvnet_event_multimodal_v3.pt ^
  --split stratified_random_9_1 ^
  --feature-set ral_stat ^
  --out-dir baseline_results_strict ^
  --models logistic linear_svm rbf_svm knn rf extratrees gradboost histgb mlp ^
  --seed 42
```

输出：

```text
baseline_results_strict\baseline_metrics.csv
baseline_results_strict\baseline_metrics_markdown.txt
baseline_results_strict\baseline_predictions_*.csv
```

把终端里的：

```text
=== Baseline comparison summary ===
```

发给 ChatGPT 分析。

## 3. 可选：轻量深度学习基线

```cmd
python run_v3_baseline_comparison.py ^
  --data camvnet_event_multimodal_v3.pt ^
  --split stratified_random_9_1 ^
  --feature-set ral_stat ^
  --out-dir baseline_results_strict_dl ^
  --models torch_mlp lstm gru tcn transformer ^
  --epochs 120 ^
  --patience 20 ^
  --batch-size 64 ^
  --seed 42
```

## 4. Update-12h 动态预警基线

```cmd
python run_v3_baseline_comparison.py ^
  --data camvnet_event_multimodal_v3_update_12h.pt ^
  --split stratified_random_9_1 ^
  --feature-set ral_stat ^
  --out-dir baseline_results_update12h ^
  --models logistic linear_svm rbf_svm knn rf extratrees gradboost histgb mlp ^
  --seed 42
```

## 5. 论文中怎么用

正式论文主对比应该是：

```text
Proposed strict model vs. RF / ExtraTrees / SVM / MLP / LSTM / GRU / TCN / Transformer
```

Update-12h 是额外的动态更新实验，不要和严格预测主实验混为一谈。
