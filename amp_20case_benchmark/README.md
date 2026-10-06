# WMA 统一 AMP Baseline：20 Case 正式复测

本目录面向新增验收口径，旧的单 Case 1.085× 对照和旧 20 Case 质量表不能替代本复测。

## 硬性不变量

- 同一台 GPU、同一软件环境、同一权重、同一输入与 seed。
- Baseline 与优化版均使用 `amp_fp16`，不得把开启 AMP 本身计为优化。
- 20 个 Case 均保持 `512×320`、DDIM 50、16 帧/轮以及 `manifest.csv` 锁定的原始交互轮数。
- 唯一历史边缘项 `unitree_z1_dual_arm_stackbox_v2/case1` 在优化版沿用已验证的 eta 策略：前 3 轮 1.0、其后 0.9；它不减少 DDIM 步数或交互轮数，其余 19 项保持 eta=1.0。
- 不使用低于 16-bit 的精度；不减少输出帧数；必须生成规定的最终 MP4。
- 优化版 20/20 的 PSNR 均不低于 25 dB。
- 以 20 Case 端到端 wall time 求和，`baseline_total / optimized_total >= 1.25`。

## 公平计时

1. 固定 GPU 时钟/功耗策略（若平台不允许则记录默认值），关闭其他 GPU 进程。
2. Baseline 和优化版各先执行一个不计时 warm-up Case；随后按同一清单顺序正式运行。
3. 端到端时间从 Case 进程启动计至最终 MP4 落盘，不排除模型加载、编译缓存初始化或编码时间。
4. 每个 Case 保存命令、退出码、`/usr/bin/time -v`、GPU 监控、最终视频探测、官方 PSNR JSON、源码提交号和补丁哈希。
5. 不从旧质量搜索候选中拼接结果；Baseline 与优化版各自必须是一次完整、可审计的 20 Case 套件。

## 当前优化路线

Baseline 是官方 FP16/AMP 推理。优化版保持采样与输出语义，候选按风险从低到高逐项消融：

1. `torch.inference_mode()` 与固定形状 cuDNN autotune；
2. 跳过仅供策略分支调试、且不进入动作或最终视频的 VAE decode；
3. 将最终评分不需要的逐轮 TensorBoard/中间 MP4 从正式热路径移除，最终 MP4 保持不变；
4. 对固定形状的扩散核心使用 `torch.compile(mode="reduce-overhead")`，编译开销计入端到端时间；
5. 仅在单 Case 正确性、帧数和 PSNR 通过后，才进入完整 20 Case 复测。

第 2–3 项已有旧单 Case 1.085× 证据；它们不足以宣称满足新门槛。第 4 项需要在目标 RTX 4090 上实测，任何未实际产生的加速数字不得写入报告。

## 自动验收

正式结果 CSV 必须包含 `manifest.csv` 的全部字段，并额外包含：

```text
status,precision,eta_policy,elapsed_s,psnr_db,resolution,frames
```

执行：

```bash
python amp_20case_benchmark/verify_amp_benchmark.py \
  --baseline results/amp_baseline/summary.csv \
  --optimized results/amp_optimized/summary.csv \
  --json-out results/amp_comparison.json
```

校验器采用失败即拒绝原则：Case 不足、参数变化、非 AMP FP16、帧数变化、任一 PSNR 低于 25 或总加速比低于 1.25，都会返回非零退出码。
