# 作用: 时晶几何 — O(2) 旋转 + 位置相关相位 → 自发周期性结构
# 调用: core.py → geometry/__init__.py → get_geometry('tc')
# 不变量: ∥h∥²
# 核心公式: h' = h·cos(θ + 2π·pos/T) + update·sin(θ + 2π·pos/T)
import math, torch, torch.nn as nn, torch.nn.functional as F


class TCProgram(nn.Module):
    """时晶指令: K个d×d线性变换 (同 O2Program)"""
    def __init__(self, d_model, n_instr=6, **kwargs):
        super().__init__()
        self.n_instr = n_instr
        self.weight = nn.Parameter(torch.randn(n_instr, d_model, d_model) * 0.02)

    def forward_raw(self, x):
        B, T, d = x.shape
        return torch.einsum('kij,btj->btki', self.weight, x)


def tc_residual(h, cos_t, sin_t, gate, prog_raw, h_mix, sin_t_conv, n_instr, activation='gelu'):
    """时晶残差: O(2) + 位置相位.

    θ'_k = θ_k + ω·pos   (ω = 2π/T, 一个完整振荡)
    效果: 浅层位置用"新信息"(θ大), 深层位置用"旧信息"(θ小)
    不同指令 θ_k 不同 → 指令 k 在不同位置有不同旋转量.
    """
    B, T, d = h.shape

    theta_orig = torch.atan2(sin_t[..., :n_instr], cos_t[..., :n_instr])
    pos = torch.arange(T, device=h.device, dtype=h.dtype).unsqueeze(0).unsqueeze(-1) / T
    theta_shifted = theta_orig + math.pi * pos
    cos_s = torch.cos(theta_shifted)
    sin_s = torch.sin(theta_shifted)

    cos_sum = (gate * cos_s).sum(dim=-1, keepdim=True)
    sin_gate = sin_s.unsqueeze(-1) * gate.unsqueeze(-1)
    delta_prog = (sin_gate * prog_raw).sum(dim=2)
    h_mix_theta = h_mix * sin_t_conv
    update = delta_prog + h_mix_theta
    if activation == 'gelu':
        update = F.gelu(update, approximate='tanh')
    elif activation == 'relu':
        update = F.relu(update)
    elif activation == 'silu':
        update = F.silu(update)

    return h * cos_sum + update
