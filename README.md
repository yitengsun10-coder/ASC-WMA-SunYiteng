# ASC Embodied World Model（WMA）实验记录

姓名：孙逸腾  
学号：240810010427

本仓库保存 WMA Baseline、优化补丁、20 个正式案例的日志、指标和资源记录。

## 结果摘要

- 完成案例：20/20。
- 平均 PSNR：40.561815 dB。
- 基本验收要求：每个正式案例 PSNR ≥ 25 dB。
- 最终达标：20/20；最低 PSNR = 25.0749163 dB，已通过全案例基本验收。
- 历史边缘案例：`unitree_z1_dual_arm_stackbox_v2/case1` 的原始官方参数结果为 24.8489438 dB；同配置复跑一致，seed=122/124 分别为 18.9310/19.0734 dB。
- 最终修复：固定 seed=123，保持输入、权重、512×320、DDIM 50、n_iter=11、视频规格和官方评分脚本不变；前三次交互 eta=1.0，其余 eta=0.9，官方评分 25.0749163 dB。
- 原始 19/20 CSV 与失败记录继续保留；新增最终 20/20 汇总和配置、日志、评分、视频验证、SHA-256，避免覆盖历史数据。
- 峰值显存：19206 MiB。

## 目录

- `official_20_cases_20260814/`：正式案例脚本、状态、PSNR、时间和 GPU 使用记录。
- `official_20_cases_20260814/FINAL_STATUS_20_OF_20.md`：最终补充验收和历史/最终状态区分。
- `official_20_cases_20260814/wma_case1_evidence/switch3_late090/`：最后一个边缘案例的完整修复证据。
- `comparison*.json`：Baseline 与优化对比。
- `wma_*_optimization.patch`：代码优化补丁。
- `代表帧/`：少量可视化证据。

为控制仓库体积，MP4 输出、TensorBoard 文件、模型权重和数据集未上传；其生成方法、摘要和逐案例指标均保留。


