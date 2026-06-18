# Step 1 实验口径审计结论

## 已确认一致的 test 组

### opencood/logs/test_opv2v_where2comm_markov_fp32
- samples: 2170
- split: opv2v_data_dumping/test
- epoch: 20
- core_method: point_pillar_where2comm_arce
- score_threshold: 0.2
- where2comm_threshold: 0.01
- AP fixed_compact: 0.823 / 0.809 / 0.655
- AP non-compact or other run: 0.647 / 0.639 / 0.501

### opencood/logs/main_opv2v_where2comm_grace_full
- samples: 2170
- split: opv2v_data_dumping/test
- epoch: 20
- core_method: point_pillar_where2comm_arce
- score_threshold: 0.2
- where2comm_threshold: 0.01
- AP channel-aware heuristic: 0.799 / 0.787 / 0.623
- AP learned QRA: 0.792 / 0.778 / 0.610
- AP learned QRA + comp + FEC + cache: 0.792 / 0.777 / 0.609

## validate 组，仅可单独参考，不能直接和 test 组比较

### opencood/logs/main_opv2v_where2comm_channel
- samples: 1980
- split: opv2v_data_dumping/validate
- AP compact: 0.650 / 0.630 / 0.490

### opencood/logs/point_pillar_where2comm_2026_06_09_12_48_25
- samples: 1980
- split: opv2v_data_dumping/validate
- core_method: point_pillar_where2comm
- AP clean: 0.850 / 0.830 / 0.690

## 当前 Step 1 结论

1. 不能把 clean Where2Comm validate AP=0.850/0.830/0.690 与 GRACE test AP=0.792/0.778/0.610 直接比较。
2. test_opv2v_where2comm_markov_fp32 与 main_opv2v_where2comm_grace_full 在 samples、split、epoch、score_threshold、where2comm_threshold 上已经对齐。
3. 但 test_opv2v_where2comm_markov_fp32 同目录下两个 AP 差异极大：0.647/0.639/0.501 vs 0.823/0.809/0.655，需要继续确认 runtime 参数。
4. Step 1 不能完全通过，直到明确 fixed_compact 与 non-compact 的实际配置差异。
