"""Weyl 群反射残差 — 反射替代旋转

O(2) 残差:  h' = h·cosθ + Δ·sinθ       (旋转)
Weyl 残差: h' = h·cos_sum - Σ(gate_k · 2(h·n_k)n_k/||n_k||²) + mix·sinθ_conv
                          └── 超平面反射

每条指令 W_k 产生法向量 n_k = W_k·h。
Reflection 比 rotation 更"锐利"——离散翻转而非平滑旋转，
可能更适合离散语法结构。

Weyl群定理: 任何正交变换可分解为有限次反射。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeylProgram(nn.Module):
    def __init__(self, d_model, n_instr=6):
        super().__init__()
        self.n_instr = n_instr
        self.weight = nn.Parameter(torch.randn(n_instr, d_model, d_model) * 0.02)

    def forward_raw(self, x):
        B, T, d = x.shape
        return torch.einsum('kij,btj->btki', self.weight, x)

    def forward(self, x, gate):
        raw = self.forward_raw(x)
        return (gate.unsqueeze(-1) * raw).sum(dim=2)


def weyl_residual(h, cos_t, sin_t, gate, prog_raw, h_mix, sin_t_conv, n_instr, activation='gelu'):
    h_exp = h.unsqueeze(2)
    dot = (h_exp * prog_raw).sum(dim=-1, keepdim=True)
    norm2 = (prog_raw * prog_raw).sum(dim=-1, keepdim=True) + 1e-8
    reflect = 2 * dot * prog_raw / norm2
    delta = (gate.unsqueeze(-1) * reflect).sum(dim=2)

    h_mix_theta = h_mix * sin_t_conv
    update = h_mix_theta - delta
    if activation == 'gelu':
        update = F.gelu(update, approximate='tanh')

    cos_sum = (gate * cos_t[..., :n_instr]).sum(dim=-1, keepdim=True)
    return h * cos_sum + update
