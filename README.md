# ASC Embodied World Model 性能与质量优化记录

- 学生：孙逸腾（240810010427）
- 题目：Embodied World Model / WMA
- 赛题仓库：https://github.com/ASC-Competition/ASC26-Embodied-World-Model-Optimization
- 项目仓库：https://github.com/unitreerobotics/unifolm-world-model-action
- 最终验收：20/20 案例完成，20/20 的 PSNR ≥ 25 dB
- 最低 PSNR：25.0749163 dB

本仓库是轻量证据仓库。模型权重、数据集、完整 MP4 和 TensorBoard 大文件不上传；逐案例命令、日志、GPU 记录、PSNR、视频规格、状态和哈希均保留。

## 一分钟验收

```bash
python tools/verify_evidence.py
```

校验器仅使用 Python 标准库，检查最终 CSV 是否包含 20 个完成案例、是否全部 `PSNR >= 25`，并核对最低值。它不会启动 GPU 推理。

## 实验环境

- Ubuntu Linux 5.15
- Python 3.10.18
- PyTorch 2.3.1+cu121
- NVIDIA GeForce RTX 4090 24 GB
- 驱动 595.71.05；运行时报告 CUDA 13.2，PyTorch 构建 CUDA 12.1

完整采集结果见 [`environment.txt`](environment.txt)。依赖安装、权重和数据准备应按两个官方仓库执行，本仓库不以不完整 `requirements.txt` 替代官方环境。

## Baseline 与性能实验

固定 Case 2、输入、权重、分辨率 `512×320`、DDIM 50、`n_iter=11`，只改变工程实现：

| 阶段 | 实际时间 s | 相对 Baseline | PSNR dB | 说明 |
|---|---:|---:|---:|---|
| Baseline | 665.247 | 1.000× | 24.9963 | 性能分析用历史 Case 2 运行，不作为最终 20 案例达标证明 |
| optimized v1 | 629.126 | 1.057× | 保持同一输出验证流程 | 第一轮减少非必要保存 |
| optimized v2 | 613.167 | **1.085×** | 24.9963 | 禁止中间产物和最终 TensorBoard，时间下降 7.8285% |

原始数据见 [`comparison_v2.json`](comparison_v2.json)、[`baseline/`](baseline/) 和 [`optimized_v2/`](optimized_v2/)。性能实验中的 24.9963 dB 必须与后续正式 20 案例验收区分，不能写成 ≥25。

## 三类优化尝试

1. 减少逐轮中间产物保存，降低视频/图像 I/O。
2. 禁止最终 TensorBoard 写入，减少重复序列化和磁盘同步。
3. 对唯一边缘案例使用受控 eta 调度，在保持输入、权重、分辨率、DDIM 50、`n_iter=11` 和官方评分脚本不变的前提下修复质量门槛。

代码差异分别见 [`wma_single_optimization.patch`](wma_single_optimization.patch)、[`wma_second_optimization.patch`](wma_second_optimization.patch) 和最终边缘案例脚本 [`world_model_interaction_eta_schedule.py`](official_20_cases_20260814/wma_case1_evidence/switch3_late090/world_model_interaction_eta_schedule.py)。

## 最终 20 案例验收

- [`wma_20case_final_summary.csv`](official_20_cases_20260814/wma_20case_final_summary.csv)：最终 20 行汇总。
- [`FINAL_STATUS_20_OF_20.md`](official_20_cases_20260814/FINAL_STATUS_20_OF_20.md)：历史 19/20 与最终 20/20 的边界说明。
- [`results/wma/`](official_20_cases_20260814/results/wma/)：每个案例的 `output.log`、`gpu_usage.csv`、`psnr_result.json`、状态、时间和视频探测信息。
- [`switch3_late090/`](official_20_cases_20260814/wma_case1_evidence/switch3_late090/)：最后一个边缘案例的配置、补丁、评分和 SHA-256。

最终统计：平均 40.561815 dB，最低 25.0749163 dB，最高 49.5623691 dB。历史原始参数 24.8489438 dB 和失败 seed 对照仍原样保留，没有覆盖或伪装。

## 复现命令

完整参数以各结果目录中的 `command.txt`、`output.log` 和配置为准。代表性命令形式如下：

```bash
python scripts/evaluation/world_model_interaction.py \
  --seed 123 --ckpt_path ckpts/unifolm_wma_dual.ckpt \
  --config configs/inference/world_model_interaction.yaml \
  --bs 1 --height 320 --width 512 --ddim_steps 50 \
  --prompt_dir <scenario>/case2/world_model_interaction_prompts \
  --dataset <scenario> --video_length 16 --frame_stride 6 \
  --n_action_steps 16 --exe_steps 16 --n_iter 11 \
  --timestep_spacing uniform_trailing --guidance_rescale 0.7 \
  --perframe_ae --savedir <output_dir>
```

运行官方评分脚本后，以生成的 `psnr_result.json` 为质量依据。严禁只凭视频观感判断是否达标。

批量复现所需的可配置环境变量和安全检查见 [`README_复现说明.md`](official_20_cases_20260814/README_复现说明.md)。
