# WMA 20 Case 正式归档与复现说明

## 最终状态

- 执行环境：AutoDL，NVIDIA GeForce RTX 4090 48GB vGPU，Ubuntu 22.04。
- 范围：5 个官方场景 × 4 个 Case，共 20/20 完成。
- 最终 PSNR ≥ 25：20/20。
- 最终平均 PSNR：40.561814632252 dB。
- 最终最低 PSNR：25.074916339940 dB。
- 最终最高 PSNR：49.562369149745 dB。

最终状态以 `wma_20case_final_summary.csv` 和 `FINAL_STATUS_20_OF_20.md` 为准。

## 历史状态必须保留

`wma_20case_summary.csv` 是第一次正式批量运行的原始 19/20 记录，其中 `unitree_z1_dual_arm_stackbox_v2/case1` 为 24.8489438 dB。该 CSV 不删除、不覆盖，也不得描述成全部达标。

边缘案例最终使用固定 seed=123、前三次交互 eta=1.0、其余 eta=0.9；输入、权重、512×320、DDIM 50、`n_iter=11`、视频规格和官方评分脚本保持不变，最终 25.0749163 dB。完整证据位于 `wma_case1_evidence/switch3_late090/`。

## 可移植运行参数

批量脚本默认兼容原服务器目录，也允许通过环境变量覆盖：

```bash
export WMA_BASE=/absolute/path/to/wma_workspace
export WMA_PROJECT=/absolute/path/to/unifolm-world-model-action
export WMA_TASK=/absolute/path/to/ASC26-Embodied-World-Model-Optimization
export WMA_ENV_DIR=/absolute/path/to/conda/env
export CONDA_SH=/absolute/path/to/conda.sh
export WMA_RESULT_ROOT="$PWD/reproduced/wma"
bash official_20_cases_20260814/run_wma_all.sh
```

单案例：

```bash
bash official_20_cases_20260814/run_wma_case.sh \
  unitree_g1_pack_camera case1
```

`run_wma_case.sh` 会检查 Case 路径、记录 GPU、执行官方运行脚本、调用官方 PSNR 工具并验证只生成一个非空预测视频。

## 证据索引

- `wma_20case_final_summary.csv`：最终 20/20 统一结果。
- `wma_20case_summary.csv`：历史 19/20 原始结果。
- `results/wma/<scenario>/<case>/`：每个 Case 的运行日志、计时、GPU 监控、视频探测和 PSNR。
- `results/wma_repro_seed123/`：原始参数独立复跑。
- `results/wma_optimized_seed122/`、`results/wma_optimized_seed124/`：失败 seed 对照。
- `wma_case1_evidence/switch3_late090/`：最终边缘案例修复证据和哈希。

为控制 GitHub 体积，MP4、权重、数据集和 TensorBoard 文件不上传；每个案例的文本日志、评分 JSON、状态、视频规格和时间均保留。
