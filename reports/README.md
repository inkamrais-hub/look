# τ-softplus: s^τ Softmax 替代注意力机制

## 项目定位

**可学习幂律归一化替代 softmax 注意力机制。**

核心思想：用 `softplus(s)^τ / Σ` 替换标准的 `softmax(s)`，其中 τ 是每个注意力头可学习的锐度参数。这给每个 head 独立选择"注意力焦距"的自由度——从广角（τ≈1）到长焦（τ≈20）。

## 技术路线

| 层 | 技术 |
|:---|:-----|
| 核心算法 | Python (NumPy + PyTorch) |
| 高性能内核 | CUDA (fused softplus→log→τ·log→softmax) |
| 实验框架 | HuggingFace Transformers |
| 验证模型 | GPT-2, Llama-3.2, Qwen3, SDXL |

## 多模型验证结果（真实数据）

### τ* 估计

- **Qwen3-0.6B** (448 heads): τ mean ~7.82, 逐头 R²=0.90
- **Qwen2.5-0.5B**: τ mean ~6.1
- **Llama-3.2-1B**: τ mean ~3.0
- **GPT-2**: τ mean ~2.8

### 零训练替换精度

| 指标 | 全局 τ=4.0 | 逐头最优 τ | 改善 |
|:----|:---------:|:----------:|:----:|
| KL 散度 | 0.623 | **0.048** | -92.5% |
| Cosine 相似度 | 0.789 | **0.962** | +21.9% |

### 注意力分布

- 同等 PPL 下注意力稀疏度 +14%
- 信息集中度 +2.7%
- 无需任何训练即可替换 softmax

## 核心原理

```python
# s^τ 归一化 (log-space, 数值稳定)
sp = F.softplus(scores.float()) + 1e-8
logits = tau * sp.log()
w = F.softmax(logits, dim=-1)   # = softplus(s)^τ / Σ

# τ* 闭式估计
tau_star = Cov(scores, log(softplus(scores))) / Var(log(softplus(scores)))
```

## 项目结构

```
src/
  tau_star.py         — TauEstimator: τ* 估计 pipeline
  numpy.py            — 纯 NumPy 参考实现 (数学真相源)
  autocal.py          — TauAutoCal: 一键校准 + 注入
  native/
    native_softmax.py — Fused τ-softplus CUDA kernel
  kernels/
    fused_softmax.py  — Fused mask+softmax+dropout kernel
reports/
  theory/THEORY.md    — 理论分析 (τ 相图, 定理, 公式)
  USAGE.md            — API 参考与使用指南
  benchmarks/         — 测试摘要
  experiments/        — 验证结果
```

## 关键发现

1. **τ 不是宇宙常数** — 它是配置的函数 τ(d_head, PE, L, layer)
2. **给定配置 τ 收敛到唯一吸引子** (σ=0.076, 8 seeds 验证)
3. **RoPE 是 τ 的最大控制器** (dh16: +140%)
4. **s^τ ↔ softmax 双向解析等价** — 存在严格数学映射
5. **PPL > 20 是 τ 学习的必要条件**
6. **逐头最优 τ 可在训练前离线计算** (~30 秒, RTX 5090)