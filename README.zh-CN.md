# VLM Precision Lab

**用自己的图片，检查量化模型的正确率和服务成本，决定能否部署。**

[真实收据案例](docs/cord-study.zh-CN.md) · [合成数据演示](https://crown-sports.github.io/vlm-precision-lab/) · [使用步骤](docs/service-evaluation.md) · [English](README.md)

权重文件变小，不代表部署更快、关键字段读得准。
在同一批标注图片上比较量化前后结果，再按质量、失败率和延迟目标放行或拒绝。

## 使用

```bash
pip install git+https://github.com/crown-sports/vlm-precision-lab.git@v0.2.0
precisionlab evaluate --dataset samples.jsonl \
  --endpoint http://localhost:8000/v1 --model model \
  --concurrency 4 --output runs/quantized
precisionlab gate --dataset samples.jsonl --run runs/quantized \
  --constraints targets.json --output runs/decision.json
```

达标返回状态码 **0**，未达标返回 **2**，可接入 CI。
[数据格式、目标设置和配对比较](docs/service-evaluation.md)见使用文档。

## 真实收据实验已完成

Qwen3-VL-8B-Instruct，在一张 RTX 5090 上用 vLLM 0.11.0 测试，两组固定 2 GiB KV cache。
开发集 100 张收据、273 个问题；测试集 99 张有标注字段的收据、258 个问题。
共记录 **3792 个计分请求**，开发集每档并发重复压测三次。

| 指标 | BF16 | AWQ W4A16 |
| --- | ---: | ---: |
| 整卡工作负载显存采样峰值 | 21.67 GiB | 12.11 GiB（−44.1%） |
| 并发 1 吞吐量中位数，成功请求/秒 | 3.008 | 3.661（+21.7%） |
| 并发 4 吞吐量中位数，成功请求/秒 | 7.002 | 7.874（+12.5%） |
| 测试总金额字段 EM | 82.11% | 84.21% |
| 预设各字段 95% 质量门槛 | **未通过** | **未通过** |

总金额净正确率提高，但 AWQ 新增了 **2 个内容错误**：数字误读，以及把现金付款当成总金额。
总金额配对差值的 95% 区间为 **[−3.16, +7.37] 个百分点**，不足以确认损失在预定 2 个百分点内。
EM 按 CORD 解析标注计分，货币空格可能与原图不同。
[完整案例](docs/cord-study.zh-CN.md)保留原图、错例、波动范围、启动问题的根因和复现步骤；
单卡按 BF16→AWQ 顺序各测一次完整部署，结果不保证其他工作负载同样提速。

## 已验证

- 服务客户端、失败处理、证据核验及归档结果算术：**31 项 CPU 检查通过**。
- 数据、权重和处理器文件已核对；请求、启动命令、失败记录、NVML 原始采样公开归档。
- 旧合成数据 HF 实验：文件 17.53→7.22 GB，峰值 **allocated** 显存 16.47→20.26 GiB。
  其显存指标与本轮整卡 NVML 口径不同。

API 指纹核对客户端请求，不能证明服务器隐藏张量或 GPU 内权重逐位一致。
AWQ、Marlin 复用上游；本项目实现受控评测、兼容修复和部署检查。

[原始实验](experiments/results/2026-10-03/) · [真实数据导入记录](experiments/results/2026-10-04/cord-import.json) · [GPU 复现](docs/reproduce.md)

代码 Apache-2.0；模型与外部数据沿用各自许可。
