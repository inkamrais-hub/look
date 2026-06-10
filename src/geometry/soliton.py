"""孤立子残差 v2 — 自门控更新

v1: sech(B(h-v)) → h≈v 概率≈0, 项永远≈0
v2: sech(B·||update||) → 更新自门控

  大更新(||update||大): sech→0, ×1      不放大噪声
  小更新(||update||小): sech→1, ×(1+A)  放大细微信号

本质: 软饱和机制, 与 O(2) 旋转互补 (方向 × 幅度)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SolitonProgram(nn.Module):
    def __init__(self, d_model, n_instr=6):
        super().__init__()
        self.n_instr = n_instr
        self.weight = nn.Parameter(torch.randn(n_instr, d_model, d_model) * 0.02)
        self.A = nn.Parameter(torch.tensor(1.5))
        self.B = nn.Parameter(torch.tensor(0.5))

    def forward_raw(self, x):
        B, T, d = x.shape
        return torch.einsum('kij,btj->btki', self.weight, x)

    def forward(self, x, gate):
        raw = self.forward_raw(x)
        return (gate.unsqueeze(-1) * raw).sum(dim=2)


def soliton_residual(h, cos_t, sin_t, gate, prog_raw, h_mix, sin_t_conv, n_instr, activation='gelu'):
    cos_sum = (gate * cos_t[..., :n_instr]).sum(dim=-1, keepdim=True)
    sin_gate = sin_t[..., :n_instr].unsqueeze(-1) * gate.unsqueeze(-1)
    delta_prog = (sin_gate * prog_raw).sum(dim=2)
    h_mix_theta = h_mix * sin_t_conv

    update = delta_prog + h_mix_theta

    norm = update.norm(dim=-1, keepdim=True)
    soliton_factor = 1.0 + 1.5 / torch.cosh(0.5 * norm)
    update = update * soliton_factor

    if activation == 'gelu':
        update = F.gelu(update, approximate='tanh')

    return h * cos_sum + update