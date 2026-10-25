@echo off
REM Run this file from:
REM C:\Users\19735\Documents\Codex\2026-07-04\n-s

python -m py_compile prepare_origin_ai_data.py
if errorlevel 1 pause

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

echo.
echo Done. Check origin_data folder.
pause
