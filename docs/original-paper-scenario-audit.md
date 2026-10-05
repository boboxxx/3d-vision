# 原论文场景核对与正式实验覆盖

2026-10-04。用户要求实验设计与场景参考原文。本表依据曹等论文
[arXiv v1](https://arxiv.org/abs/2502.12735v1) 的 Section V、Figures 11–13、
Tables I–III/VIII–XI；图中采样点经过 PDF 页面核对，未把曲线读取为测量结果。
本地阅读 PDF SHA256：
`1c4cf5d83fd8c63dd886b680c055257d0a9ded99bf29f070ce6deed1448bc213`。
精确作者通信源码、原始模型及数值评估环境尚未找到。

| 原文设置 | 正式实验要求 | 现有覆盖及缺口 |
|---|---|---|
| KITTI 7481 帧约半数训练、半数验证 | 固定公开 3712/3769 分割；承认作者精确分割未知 | 数据齐全；3340/372 是训练集内探索分割，不能替代正式验证 |
| stereo 传感端到云端，重建后送入 Stereo-RCNN | 保留这个 RGB 基线与原检测器；新方法明确记录直接任务解码边界 | 两个检测器已接入；最终训练后 RGB AP 未产生 |
| YOLOv5n key area；box 的 5 个参数无误传输 | 主场景可靠 ROI 元数据；同时记录元数据大小，另列含开销总资源 | 已缓存 ROI；现有 CRC/BPSK 传输属于实际链路扩展 |
| 源压缩 10×/30×/50× | global/key 空间压缩分别 36/1、36/4、64/16；同时报告实际码长 | 现有固定结构仅为一个 30× 架构描述变体；其平均物理码率未匹配新方法 |
| 源编码 JPEG/JPEG2000/SRCNN/ECSIC | 源编码独立无噪声评估；NN 输出（ECSIC 除外）8-bit | JPEG 与 JP2 工具待实现；SRCNN/ECSIC 仍需具体权重、训练和码流实现 |
| channel-only 30×，learned 与两种 LDPC+QAM | 相同源输入，6-bit 量化给数字基线；公平比较真实 channel uses 与能量 | 现有 9-channel 与 62400 uses 不相等；不能直接声称码率优势 |
| AWGN Figure 11，6–18 dB | 整数点 6,7,…,18；同一固定权重 | 目前内部终点只有 identity/AWGN10，F7 U[0,20] 为探索 |
| Rayleigh Figure 12，完美 CSI、ZF | 点 6,8,…,18；声明衰落相干长度及 SNR 定义 | 当前 noisy-pilot/LMMSE 与主场景不同，单列实用扩展 |
| joint source/channel Figure 13 | AWGN 整数 6–18；JPEG/JP2/ECSIC 各配两种 LDPC+QAM，30× | 完整数字物理链路尚未实现；不能用容量公式替代 LDPC 仿真 |
| Stereo-RCNN Car 2D 左/右、BEV、3D AP，Easy/Moderate/Hard，IoU 0.5 | 原文主比较 IoU 0.5，并列 R11 与 R40；IoU 0.7/R40 为现代补充指标 | 已核对 clean 3769；作者无压缩 65.72/45.60/39.39 与我们的 clean 数值不一致 |
| PSNR/SSIM 的 global 与 key | 全图 g、ROI union k；无 ROI 单独记录 undefined；固定正规定义 | 文中内/外与表注 g 存在歧义；不可静默换成背景指标 |
| 第五阶段含 channel-only + joint | 原文预算 12/10/6/10/45 固定；新方法公平训练曝光另锁 | 原文变体阶段 1–4 审计完成，第 5 阶段仍运行 |

原文没有给出 LDPC 校验矩阵、码长、解码迭代次数、量化范围、QAM 映射、
衰落相干长度、AP 采样版本及精确分割。后续实现必须公开选定变体，不能以
“原论文相同设置”掩盖这些差异。两种数字配置名义净频谱效率均为
4 source bits/complex symbol，但真实块填充、串行化及信令会改变总资源。

当前 83-epoch 变体与 F7 队列保持原有协议及源码冻结。已有主验证 clean
结果已见过，因此后续 compressed 条件只能称“在新的压缩结果产生前固定”，
不声称从未见过主验证集。参数选择仍限训练集内部，禁止按主验证 AP 调参。

原文 PSNR 式 (18) 的写法与常规 MSE 定义存在歧义；新实验显式采用
`10 log10(1/MSE)`，RGB 范围 [0,1]。不把与印刷公式的差异解释为性能差异。
KITTI 3D AP 使用原生三维 IoU 评估，不另外发明“左右 2D 同时匹配”的指标。
