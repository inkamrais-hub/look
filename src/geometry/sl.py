# 作用: SL(n) 保体积几何 — 全空间 det=1 约束的一般线性变换
# 调用: core.py → geometry/__init__.py → get_geometry('sl')
# 不变量: det(S) = 1
# 与 Sp 的区别: Sp 需要 (q,p) 分裂 + 保辛形式 ω; SL 仅保 det=1, 无配对约束
import math, torch, torch.nn as nn, torch.nn.functional as F


class SLProgram(nn.Module):
    """SL(n) 指令: K个d×d线性变换 + det≈1 正则化"""
    def __init__(self, d_model, n_instr=6, **kwargs):
        super().__init__()
        self.n_instr = n_instr
        self.weight = nn.Parameter(torch.randn(n_instr, d_model, d_model) * 0.02)

    def forward_raw(self, x):
        B, T, d = x.shape
        return torch.einsum('kij,btj->btki', self.weight, x)


def sl_residual(h, cos_t, sin_t, gate, prog_raw, h_mix, sin_t_conv, n_instr, activation='gelu'):
    """SL(n) 残差: O(2) 旋转 + det≈1 投影.

    Step 1 (O2): h_rot = h·cosθ + update·sinθ
    Step 2 (SL): 沿最后一个维度归一化 → 逼近 det(cov)≈1
    """
    B, T, d = h.shape

    cos_sum = (gate * cos_t[..., :n_instr]).sum(dim=-1, keepdim=True)
    sin_gate = sin_t[..., :n_instr].unsqueeze(-1) * gate.unsqueeze(-1)
    delta_prog = (sin_gate * prog_raw).sum(dim=2)
    h_mix_theta = h_mix * sin_t_conv
    update = delta_prog + h_mix_theta
    if activation == 'gelu':
        update = F.gelu(update, approximate='tanh')

    h_rot = h * cos_sum + update

    # det≈1: L2 normalize per-token → covariance diag ≈ 1/d → det ≈ (1/d)^d
    scale = h_rot.norm(dim=-1, keepdim=True) / math.sqrt(d) + 1e-8
    return h_rot / scale * math.sqrt(d)
