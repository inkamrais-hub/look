# 几何研究路线图

## 使用方式

```python
from ulnp import ULNP
import torch.nn.functional as F

# ── 基本用法 ──────────────────────────────

# O(2) 欧氏旋转 (默认, 标配主力)
model = ULNP(vocab_size=5000, d_model=64, geometry='o2')

# Sp(n) 辛几何 (Iwasawa KAN + Størmer-Verlet)
model = ULNP(vocab_size=5000, d_model=64, geometry='sp')

# 每层独立配对模式 (自/邻/远距/对称 轮转)
# depth ≤ 4: 4 种模式各用一次
# depth  > 4: 循环复用

# ── 训练 ──────────────────────────────────

# Sp(n) 带辛正则化 (通常为 0, Iwasawa 数学自洽)
logits = model(x)
ce_loss = F.cross_entropy(logits.view(-1, vocab_size), x.view(-1))
sp_reg = model.symplectic_regularization()
loss = ce_loss + 0.01 * sp_reg
loss.backward()

# or direct per-layer
# for layer in model.program:
#     loss += 0.01 * layer.symplectic_loss()

# ── 消融开关 ──────────────────────────────

model._use_stream = False  # 关闭 InstructionStream

# ── 架构参数 (O(2) only) ──────────────────

model = ULNP(100, geometry='o2',
    program='linear',       # linear / nonlinear / vector / rank1 / hierarchical
    router='sinusoidal',    # sinusoidal / position
    mixer='conv',           # conv / fourier / multiscale_ema / ...
    activation='gelu',      # gelu / relu / silu / None
    d_model=96, n_instr=6, depth=6, kernel_size=5,
)
```

---

## 已实现 ✓

| 几何 | 文件 | 不变量 | 状态 |
|------|------|--------|------|
| O(2) 欧氏旋转 | o2.py | ∥h∥² | ✅ 已验证主力 |
| Sp(n) 辛几何 | sp.py | det(J)=1 | ✅ qb 完整移植 (Iwasawa KAN + 配对模式) |
| U(1) 复相位 | u1.py | \|z\|² | 🏗️ 骨架 (残差, 无 Program) |
| SU(2) 四元数 | su2.py | ∥q∥² | 🏗️ 骨架 (残差, 无 Program) |
| SL(n) 保体积 | sl.py | det=1 | 🏗️ 骨架 |
| Lorentz SO(1,n-1) | lorentz.py | t²-∥x∥² | 🏗️ 骨架 |
| CO(n) 保角 | co.py | 方向/∥h∥ | 🏗️ 骨架 |

## Sp(n) 实现细节

移植自 `_archive/ulnp-qb/experiments/symplectic_deep/`：

- **Program**: SymplecticLinear — Iwasawa KAN 分解 (θ, α, β → 2×2 辛矩阵)
  - 理论：Sp(2n,R) = K·A·N, θ≈0.41 普适吸引子
  - det ≡ 1 自洽，与参数无关
  - 4 种配对模式轮转：(i,i) / (i,i+1) / (i,i+n/2) / (i,n-1-i)

- **Residual**: deep_sp_residual — Størmer-Verlet + damping
  - 3 条 damping 是 θ≈0.41 和跨域一致性的必要组件

- **架构**: 每层独立 SpProgram (`nn.ModuleList`)

## 添加新几何的步骤

1. 写 `geometry/xxx.py`，导出 `XxxProgram(nn.Module)` + `xxx_residual(function)`
2. 在 `geometry/__init__.py` 注册 `GEOMETRIES['xxx'] = {'Program': XxxProgram, 'residual': xxx_residual}`
3. 完成 — `core.py` 会自动创建 per-layer `nn.ModuleList` 并分发 `layer_idx`

## Tier 1: 辛假说延伸

- [ ] 辛谐振子: Sp(n) + 能量守恒 (sp.py 变体)
- [ ] 度量辛: Sp(n) + 正定度量 (sp.py 变体)
- [ ] Lie-Poisson: Sp(n) + 动量映射 (sp.py 变体)

## Tier 2: 经典李群扩展

- [x] SL(n) — 特殊线性群 (骨架)
- [x] SO(1,n-1) — 洛伦兹群 (骨架)
- [x] CO(n) — 保角群 (骨架)
- [ ] PO(n) — 射影群
- [ ] OSp(n|m) — 正交辛超群

## Tier 3/4: 远期

- [ ] 信息几何 (Fisher 度量)
- [ ] 接触几何 (能量预算)
- [ ] 路径积分
- [ ] 拓扑序/TQFT

## 实验记录

| 日期 | 几何 | 结论 |
|------|------|------|
| 2026-05-26 | O2/U1/SU2/Sp | U(1) > Sp > SU(2) > O(2) (d=32..256) |
| 2026-05-27 | Sp cross-modal | q/p 分裂跨模态一致 (lang vs motion) |
