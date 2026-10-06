# WMA 权重恢复与释放前检查

2026-10-05 已核对官方模型仓库API与文件页。

- 仓库：https://huggingface.co/unitreerobotics/UnifoLM-WMA-0-Dual
- 固定版本：88e73c19d72bead94da3d7fc2433ade6d468167c
- 文件：unifolm_wma_dual.ckpt
- 文件大小：16719237688 字节（约15.57 GiB）
- 官方LFS SHA256：f5c5202b41bbda777e2a861104a8886e9c2b0cd981d496a8bffaa67e0149c7ef
- 固定版本下载地址：https://huggingface.co/unitreerobotics/UnifoLM-WMA-0-Dual/resolve/88e73c19d72bead94da3d7fc2433ade6d468167c/unifolm_wma_dual.ckpt

2026-10-06 已取得020实际运行权重SHA256：f5c5202b41bbda777e2a861104a8886e9c2b0cd981d496a8bffaa67e0149c7ef。服务器文件大小为16719237688字节。两项均与上述官方固定版本完全一致，确认是同一份权重。

本次在无GPU模式下执行sha256sum，结果为全文件校验，不加载模型或运行推理。网关曾断开，最终通过后台任务文件取得结果；哈希进程已结束。无需为了这份官方相同权重再持续开机传输16.7GB。

实验结果、原始输入和20项最终输出已在本地，备份包不包含权重。没有必要重新推理或开启GPU测试。就当前WMA任务已核验的数据而言，可以关闭并释放020；释放会删除服务器文件，未来复现需从上述固定版本重新下载并校验权重。外部托管并不等同于永久离线备份；如果用户要求完全离线复现，仍需自行下载并保存该权重。其他与本任务无关的私有文件不在此次核验范围。
