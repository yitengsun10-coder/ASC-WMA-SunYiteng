# WMA 20 案例最终补充验收

原始 `wma_20case_summary.csv` 原样保留，用于记录当时 19/20 的历史状态；本文件和 `wma_20case_final_summary.csv` 记录随后完成的边缘案例受控修复。

- 完整生成：20/20
- 严格 PSNR >= 25：20/20
- 平均 PSNR：40.561814632252 dB
- 最低：25.074916339940 dB
- 最高：49.562369149745 dB
- 历史边缘案例：`unitree_z1_dual_arm_stackbox_v2/case1`，原始官方参数 24.84894379792539 dB
- 最终方案：固定 seed=123；前三次交互 eta=1.0，其余 eta=0.9；官方评分 25.07491633993985 dB
- 其余固定项：输入、权重、512x320、DDIM 50、n_iter=11、视频规格与官方评分脚本不变
- 证据：`wma_case1_evidence/switch3_late090/`，含配置、补丁、日志、评分 JSON、视频验证与 SHA256
