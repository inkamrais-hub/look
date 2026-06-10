# ULNP —— 超轻量神经程序架构

**Ultra-Lightweight Neural Program**: 用几何代数结构替代 Transformer 注意力的序列建模框架。

## 核心思路

传统 Transformer 用 `softmax(Q·K^T)·V` 做 token 混合。ULNP 用 **Program + Router + Mixer** 三组积木：

- **Program** (6种): 指令积木 —— K 条可学习指令 Wₖ，每条是一个 d×d 变换
- **Router** (2种): 路由积木 —— 正弦编码 + 线性投影 → 指令门控权重 gₖ
- **Mixer** (26种): 混合积木 —— 可插拔的 token 混合策略 (Conv/FFT/EMA/双线性/UPA...)

残差更新方程 (统一哈密顿模式):

```
θ = sigmoid(θ_raw) · (π/3)
h' = h · Σgₖcosθₖ + Σgₖ·sinθₖ·Wₖ·ĥ + [Mixer(ĥ) + Stream(ĥ,gate)]·sinθ_conv
```

## 14 种几何代数结构

ULNP 的 Program + Residual 由几何注册表统一调度:

| 几何 | 文件 | 不变量 | 状态 |
|------|------|--------|------|
| O(2) 欧氏旋转 | o2.py | ∥h∥² | ✅ 已验证 |
| Sp(n) 辛几何 | sp.py | det(J)=1 | ✅ 已验证 |
| U(1) 复相位 | u1.py | \|z\|² | ✅ 实现 |
| SU(2) 四元数 | su2.py | ∥q∥² | ✅ 实现 |
| SL(n) 保体积 | sl.py | det=1 | ✅ 实现 |
| Lorentz SO(1,n-1) | lorentz.py | t²-∥x∥² | ✅ 实现 |
| CO(n) 保角 | co.py | 方向/∥h∥ | ✅ 实现 |
| Time Crystal | time_crystal.py | 周期相位 | ✅ 实现 |
| Gauge | gauge.py | 局部规范 | ✅ 实现 |
| Tropical | trop.py | max-plus | ✅ 实现 |
| Cobordism | cobordism.py | 拓扑平滑 | ✅ 实现 |
| TQFT | tqft.py | 谱不变量 | ✅ 实现 |
| Soliton | soliton.py | 孤立子 | ✅ 实现 |
| Weyl | weyl.py | 反射群 | ✅ 实现 |

## 验证结果

### 语言建模 (WikiText, d=192, L=12, K=6)

| 配置 | PPL |
|------|-----|
| ULNP 统一哈密顿 (baseline) | **3.93** |
| Transformer baseline | 3.99 |
| Hamilton variant | 3.95 |

### 几何对比 (Shakespeare, d=32, L=2, K=4, 300步)

| 几何 | 参数 | PPL |
|------|------|-----|
| U(1) | ~100K | 7.8 |
| Sp(n) | ~100K | 8.1 |
| SU(2) | ~100K | 8.3 |
| O(2) | ~100K | 9.5 |

### BCI 4-way 运动想象分类 (BCIC IV 2a)

| 几何 | 准确率 |
|------|--------|
| Sp(n) | 最高 (辛几何优势) |
| O(2) | 基线 |
| 更多几何 | 实验进行中 |

### Sp(n) vs O(2) 对比分析

| 维度 | 发现 |
|------|------|
| 角度吸引子 | Sp(n) Iwasawa θ≈0.41 普适吸引子, 与输入无关 |
| q/p 分裂 | 跨模态一致 (语言 vs 运动想象) |
| 配对模式 | 4 种模式轮转: 自/邻/远距/对称 |
| 辛正则化 | det(J)=1 数学自洽, 正则化损失 ≈0 |

## 技术栈

- Python 3.10+ / PyTorch 2.0+
- 纯 PyTorch 实现, 无额外 CUDA kernel 依赖
- Optional: Triton fused residual kernel (实验中)
- 验证环境: Shakespeare / WikiText / BCIC IV 2a

## 项目结构

```
ulnp/
├── src/
│   ├── core.py          # 顶层容器 — 组装积木
│   ├── program.py       # 6 种指令积木
│   ├── router.py        # 2 种路由积木
│   ├── mixers.py        # 26 种混合积木
│   └── geometry/        # 14 种几何代数结构
├── reports/
│   ├── README.md        # 项目简介
│   ├── theory/          # 理论笔记
│   │   ├── BEYOND_ULNP.md
│   │   ├── GEOMETRY_BRAINSTORM.md
│   │   └── ROADMAP.md
│   └── experiments/     # 验证结果
│       └── validation_summary.md
```

## 引用

```
@misc{ulnp2026,
  author = {τ Project},
  title = {ULNP: Ultra-Lightweight Neural Program — Geometry as Attention},
  year = {2026}
}
```