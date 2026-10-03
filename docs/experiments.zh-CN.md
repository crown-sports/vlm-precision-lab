# 2026-10-03：首轮基线与研究决策

使用固定 revision 的 Qwen3-VL-8B-Instruct，在同型号 RTX 5090 上分别运行 BF16 和对称 group-128 W4 RTN 模拟量化。数字／接口／图关系各 100 例，总计 300 个开发样例、150 个生成来源。每个来源两个答案改变样例；没有运行测试集。

| 任务与指标 | BF16 | RTN 模拟量化 | 净变化 |
| --- | --- | --- | --- |
| 数字字段严格 EM | 100% | 100% | 0 pp |
| 接口标识严格 EM | 100% | 100% | 0 pp |
| 图关系严格 EM（含字母顺序） | 41% | 32% | −9 pp |
| 图关系无序端点正确率 | 50% | 52% | +2 pp |
| 图关系输出符合字母排序格式 | 78% | 62% | −16 pp |

![内容与格式分开评测](figures/p0-metrics.png)

严格图关系 EM 有 11 个退化与 2 个恢复样例，按来源配对 bootstrap 的净变化区间为 −16 到 −2 pp。但无序端点正确率没有退化样例，有 2 个恢复样例。因此，不把严格 EM 的下降解释成连线感知能力下降。排序两个已知端点是必要的简单修正，应作为强对照，而不是使用昂贵的精度搜索修复格式。

数字和接口指标的零变化区间来自这份有限样本的描述性 bootstrap，不能证明未来真实来源上的等价。原模型图关系正确率仅 50%，提示首先需要检查其绝对能力。合成数据共享生成器家族，不支持真实文档或跨模板泛化主张。

BF16 与 RTN 均保存 300 条完整输出和输入张量 SHA256；输入、模型来源、解码参数经过比较器核验。两者峰值 PyTorch allocated 均为 20,509,668,864 字节（约 19.10 GiB）。RTN 模拟量化仍以 BF16 存储并执行稠密计算，所以这个实验**没有降低文件体积或显存，也没有证明 INT4 加速**。

这两个 P0 运行使用的脚本会因 `BatchFeature.to()` 原地修改而逐渐保留整个开发集的 GPU 输入；因此上述峰值包含输入驻留，不是标准服务请求峰值。原脚本快照已保留在结果目录，后续脚本改为临时设备映射并逐批释放输入。新旧脚本的峰值不直接用来计算量化显存收益；性能结果仍须在统一后端、统一脚本下另外测量。原始质量预测与输入指纹不受这项计量限制影响。

强分层校准从 192 个候选中选出 177 例，使用 73,986 / 74,000 个输入 token，三种任务各 59 例。token 数由固定图像处理器实测，包含视觉 token；没有用字符数或估算样本数替代实际成本。

## 正式导出的兼容与资源记录

固定栈是 PyTorch 2.8.0 / CUDA 12.8、Transformers 4.57.3、LLM Compressor 0.9.0、compressed-tensors 0.13.0。这是本次受控环境，不是最新版支持情况的断言。

- 第 3 次校准因 Qwen3-VL 位置插值读取 meta 设备而失败，退出码 1。复用 [上游 PR #1958](https://github.com/vllm-project/llm-compressor/pull/1958) 的位置设备修正；在两个 dtype、两种 grid 组合的小模型对照中与原算法输出逐元素相同，最大误差为 0。这是已知兼容适配，不计原创算法贡献。
- 第 4 次校准已进入 decoder，但被 SIGKILL 终止，退出码 −9。运行前后容器 `oom_kill` 计数增加 1；CPU cgroup 限额为 17,179,869,184 字节。大量 CPU 权重与缓存造成的压力，是支持该资源诊断的证据，而不是“GPU 卡不够”的证据。
- 第 5 次改为 GPU 驻留权重、第二张 GPU 放中间激活，使用相同 177 个校准样本。旧版 dispatcher 的 CPU 卸载步骤被显式绕过，量化与缩放运算仍调用上游实现。完整结果以退出记录和导出／重载 manifest 为准。

失败退出记录、压缩日志与第 4 次源码快照在原始结果目录中保留；不是只保留成功运行。

## 决策

这批数据没有支持“关键内容能力被量化破坏”的假设，也没有支持新算法优于已有方法。先完成正式 AWQ 导出／重载检查，再在合法真实文档及具有独立来源的更代表性数据上寻找内容回归。没有明确问题前，不扩张精度搜索或宣称算法创新。

## 复现与原始记录

- [固定输入与数据卡](../examples/diagnostic-v2/)
- [BF16 原始预测与 manifest](../experiments/results/2026-10-03/p0-bf16/)
- [RTN 原始预测与 manifest](../experiments/results/2026-10-03/p0-fake-rtn/)
- [比较报告 JSON](../experiments/results/2026-10-03/p0-rtn-report.json)／[离线 HTML](../experiments/results/2026-10-03/p0-rtn-report.html)
- [实际校准选样与成本](../experiments/results/2026-10-03/calibration-stratified.selection.json)
- [明确的研究决策](../experiments/results/2026-10-03/decision.json)

只复现报告无需 GPU：

```bash
precisionlab compare --dataset examples/diagnostic-v2/samples.jsonl \
  --reference experiments/results/2026-10-03/p0-bf16/predictions.jsonl \
  --candidate experiments/results/2026-10-03/p0-fake-rtn/predictions.jsonl \
  --output runs/reproduced-report.html
```

GPU 运行步骤见 README，固定输入使用 `examples/diagnostic-v2/samples.jsonl`。模型权重、字体与私有 API 凭据均不包含在仓库内。
