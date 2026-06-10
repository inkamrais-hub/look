# τ-softplus 使用指南

> softplus(s)^τ 注意力机制 — 可学习幂律归一化替代 softmax
>
> 最后更新: 2026-05-15

---

## 目录

1. [快速开始](#1-快速开始)
2. [API 参考](#2-api-参考)
3. [支持的模型](#3-支持的模型)
4. [τ* 估计器对比](#4-τ-估计器对比)
5. [权重量化影响](#5-权重量化影响)
6. [实验脚本](#7-实验脚本)
7. [常见问题](#8-常见问题)

---

## 1. 快速开始

### 安装

```bash
pip install torch transformers numpy matplotlib
```

### 最小示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from stau import TauPipeline

model = AutoModelForCausalLM.from_pretrained('gpt2', torch_dtype=torch.float32,
    device_map='auto', attn_implementation='eager')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model.eval()

pipe = TauPipeline(model, estimator='sp_regressed')
tau_map = pipe.calibrate(['Machine learning is changing the world'])
texts = pipe.generate(['The future of AI is'], tau_map, max_new=40)
for t in texts:
    print(f"生成: {t['generated']}")
```

### 一行搞定

```python
pipe = TauPipeline(model)
tau_map = pipe.calibrate(["calibration text"])
texts = pipe.generate(["prompt"], tau_map)
```

---

## 2. API 参考

### `TauPipeline`

```python
TauPipeline(model, estimator='sp_regressed', device=None)
```

**参数:**
| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | HuggingFace model | 任意 CausalLM (eager attention) |
| `estimator` | str or TauEstimator | 预设: 'sp_regressed'(默认), 'sp_closed', 'clamp_closed', 'clamp_regressed', 'clamp_legacy', 'relu_closed' |

**方法:**
| 方法 | 返回 | 说明 |
|------|------|------|
| `capture_scores(texts, max_len=128)` | dict | 收集 pre-softmax scores |
| `estimate_tau(captured)` | (dict, list) | 从 scores 估计 τ* |
| `calibrate(texts, max_len=128)` | dict | 一站式 capture + estimate |
| `generate(prompts, tau_map, max_new=40, **kwargs)` | list | 注入 + 生成 |
| `analyze(tau_map, model_name=None)` | dict | τ 分布统计 + 直方图 |
| `compare_estimators(calib_texts, gen_prompts)` | list | 对比所有估计器 |
| `inject(tau_map)` | context manager | `with pipe.inject(tau_map): ...` |

### `TauEstimator`

```python
TauEstimator(f='softplus', stats=['cov_ratio'], coef=None,
             tau_min=1.05, tau_max=20.0, eps=1e-8, cov_on='s')
```

**预置构造器:**
| 构造器 | f(s) | 回归 | 说明 |
|--------|------|------|------|
| `TauEstimator.sp_closed()` | softplus | 无 | 纯 Cov/Var |
| `TauEstimator.sp_regressed()` | softplus | Cov+Skew+Kurt | 推荐, R²≈0.79 |
| `TauEstimator.clamp_closed()` | clamp | 无 | 纯 Cov/Var |
| `TauEstimator.clamp_regressed()` | clamp | Cov+Skew+Kurt | 旧回归公式 |

---

## 3. 支持的模型

| 模型 | 类型检测 | 状态 | 典型 τ mean |
|------|---------|------|------------|
| GPT-2 | `gpt2` | ✅ 测试通过 | ~2.8 |
| Llama-3.2-1B | `llama` | ✅ 测试通过 | ~3.0 |
| Qwen2.5-0.5B | `qwen2` | ✅ 测试通过 | ~6.1 |
| Qwen3-0.6B | `qwen3` | ✅ 测试通过 | ~7.8 |
| Llama-3.1/3 | `llama` | ✅ 兼容 | 待测 |

**注意事项:**
- 所有模型必须使用 `attn_implementation='eager'`
- GQA 模型 (Qwen/Llama) 自动处理 repeat_kv
- Pipeline 自动检测模型类型

---

## 4. τ* 估计器对比

### 数学原理

```python
# 核心公式: τ* = Cov(s, log(f(s))) / Var(log(f(s)))
phi = F.softplus(s) + EPS
log_phi = phi.log()
cov = ((s - s.mean()) * (log_phi - log_phi.mean())).mean()
var_log = log_phi.var().clamp(min=EPS)
tau_star = (cov / var_log).clamp(1.05, 20.0)
```

### 注入函数 (必须 log-space)

```python
# ✅ 正确: log-space, 数值稳定
sp = F.softplus(scores.float()) + EPS
logits = tau * sp.log()
w = F.softmax(logits, dim=-1).to(query.dtype)

# ❌ 错误: 直接幂运算, 高 τ 值下溢
w = sp.pow(tau) / sp.pow(tau).sum(dim=-1)
```

### 实验结果 (Qwen3-0.6B)

| 估计器 | τ mean | 生成质量 |
|--------|--------|---------|
| **sp_regressed** | 7.82 | ✅✅ 最佳 |
| sp_closed | 6.96 | ✅ 良好 |
| clamp_regressed | 6.38 | ⚠️ 部分重复 |

---

## 5. 权重量化影响

| 量化 | τ 分布变化 | 生成质量 |
|------|-----------|---------|
| int8 | 几乎不变 | ✅ 正常 |
| int4 | τ mean 微升 (+0.16) | ⚠️ 开始偏离 |
| int2 | τ 分布塌缩 (std↓70%) | ❌ 乱码 |

浮点 4-bit (fp4/nf4) 在 τ 分布保护上明显优于 int4。

### 关键发现

1. τ 分布是量化损伤的敏感指标
2. int8 量化对 τ* 零影响
3. NF4 权重 + int8 激活 = τ-softplus 全量化的最小可行配置
4. 4-bit 激活在 τ* 框架下不可用

---

## 7. 实验脚本

| 脚本 | 位置 | 说明 |
|------|------|------|
| `tau_sp_validate.py` | `experiments/` | softplus τ 验证 (Qwen3/Llama) |
| `tau_survey.py` | `experiments/` | 多模型统计普查 |
| `tau_quant_fp_comparison.py` | `experiments/` | int4 vs fp4 vs nf4 量化对比 |

### 运行实验

```bash
python experiments/tau_sp_validate.py           # Qwen2.5-0.5B 验证
python stau/tests/test_tau_star_sp.py           # 单元测试
python stau/demo_bandit.py                      # 老虎机验证 demo
```

---

## 8. 常见问题

### Q: 为什么必须用 `attn_implementation='eager'`?
τ* 注入通过 patch `eager_attention_forward` 实现。

### Q: `sp_regressed` 和 `sp_closed` 有什么区别?
- `sp_closed`: 纯 Cov/Var，无回归修正
- `sp_regressed`: Cov/Var + 偏度修正 + 峰度修正 (R²≈0.79)

### Q: τ 值太大 (>10) 怎么办?
Pipeline 自动 clamp τ 到 [1.05, 20.0]。

### Q: 支持 training 模式吗?
当前只支持 eval 模式。训练模式需额外处理 dropout 和梯度。