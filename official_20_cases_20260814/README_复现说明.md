# WMA 官方 20 Case 正式归档

- 执行日期：2026-08-14
- 环境：AutoDL，NVIDIA GeForce RTX 4090 48GB vGPU，Ubuntu 22.04，PyTorch 2.1.2 / CUDA 11.8
- 范围：5 个官方场景 × 4 个 Case，共 20/20 完成
- 平均 PSNR：40.550516 dB
- PSNR >= 25：19/20
- 最佳：`unitree_z1_stackbox/case3`，49.562369 dB
- 最低：`unitree_z1_dual_arm_stackbox_v2/case1`，24.848944 dB
- 平均 wall-clock：950.35 s/Case

## 证据索引

- `wma_20case_summary.csv`：统一结果表。
- `results/wma/<scenario>/<case>/`：每个 Case 的日志、计时、GPU 监控、视频探测和 PSNR。
- `videos_all_20/`：20 个官方配置输出视频。
- `results/wma_repro_seed123/`：最低 PSNR Case 使用官方 seed=123 的独立复跑，结果完全一致。
- `results/wma_optimized_seed124/`、`results/wma_optimized_seed122/`：仅改变随机种子的单变量探索，PSNR 更低，未替换正式结果。

所有正式视频均为 512×320、8 fps、176 帧、22 秒。唯一低于 25 dB 的 Case 为 `unitree_z1_dual_arm_stackbox_v2/case1`；重复运行和两个预先限定的 seed 对照均未改善，因此如实保留官方 seed=123 正式结果，不修改输入、评价脚本或视频。
