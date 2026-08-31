# ASC Embodied World Model（WMA）实验记录

姓名：孙逸腾  
学号：240810010427

本仓库保存 WMA Baseline、优化补丁、20 个正式案例的日志、指标和资源记录。

## 结果摘要

- 完成案例：20/20。
- 平均 PSNR：40.550516 dB。
- 基本验收要求：每个正式案例 PSNR ≥ 25 dB。
- 实际达标：19/20；因此本题已完成全量运行，但**尚未通过全案例基本验收**。
- 未达标案例：`unitree_z1_dual_arm_stackbox_v2/case1`，PSNR = 24.8489438 dB，比阈值低 0.1510562 dB。
- 同配置、同 seed=123 复跑得到相同结果；seed=122/124 分别为 18.9310/19.0734 dB，未选择性隐去失败数据。
- 峰值显存：19206 MiB。

## 目录

- `official_20_cases_20260814/`：正式案例脚本、状态、PSNR、时间和 GPU 使用记录。
- `comparison*.json`：Baseline 与优化对比。
- `wma_*_optimization.patch`：代码优化补丁。
- `代表帧/`：少量可视化证据。

为控制仓库体积，MP4 输出、TensorBoard 文件、模型权重和数据集未上传；其生成方法、摘要和逐案例指标均保留。


