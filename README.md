# ASC Embodied World Model（WMA）实验记录

姓名：孙逸腾  
学号：240810010427

本仓库保存 WMA Baseline、优化补丁、20 个正式案例的日志、指标和资源记录。

## 结果摘要

- 完成案例：20/20。
- 平均 PSNR：40.550516 dB。
- 19/20 案例达到 25 dB 阈值。
- 峰值显存：19206 MiB。

## 目录

- `official_20_cases_20260814/`：正式案例脚本、状态、PSNR、时间和 GPU 使用记录。
- `comparison*.json`：Baseline 与优化对比。
- `wma_*_optimization.patch`：代码优化补丁。
- `代表帧/`：少量可视化证据。

为控制仓库体积，MP4 输出、TensorBoard 文件、模型权重和数据集未上传；其生成方法、摘要和逐案例指标均保留。

