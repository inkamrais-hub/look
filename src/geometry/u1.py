# 作用: U(1) 复相位几何 — 相邻维配对复数旋转 + O(2) 残差
# 调用: core.py → geometry/__init__.py → get_geometry('u1')
# 不变量: |z|² (每对复数模长)
import math, torch, torch.nn as nn, torch.nn.functional as F


class U1Program(nn.Module):
    """U(1) 指令: K个d×d线性变换 (同 O2Program)"""
    def __init__(self, d_model, n_instr=6, **kwargs):
        super().__init__()
        self.n_instr = n_instr
        self.d_model = d_model
        self.weight = nn.Parameter(torch.randn(n_instr, d_model, d_model) * 0.02)

    def forward_raw(self, x):
        B, T, d = x.shape
        return torch.einsum('kij,btj->btki', self.weight, x)

    def forward(self, x, gate):
        raw = self.forward_raw(x)
        return (gate.unsqueeze(-1) * raw).sum(dim=2)


def u1_residual(h, cos_t, sin_t, gate, prog_raw, h_mix, sin_t_conv, n_instr, activation='gelu'):
    """U(1) 残差: 相邻维度复数旋转 + O(2) 旋转混合.

    两步:
      1. h 内部做复数旋转: z' = z·e^{iθ} (保 |z|²)
      2. O(2) 残差: h' = h_rot·cosθ + update·sinθ (保 ∥h∥²)
    """
    B, T, d = h.shape
    d2 = (d // 2) * 2

    # 加权相位角
    theta = (gate * sin_t[..., :n_instr]).sum(dim=-1) * (math.pi / 2)
    cos_rot = torch.cos(theta).unsqueeze(-1)
    sin_rot = torch.sin(theta).unsqueeze(-1)

    h_rot = h.clone()
    h_even = h[..., :d2:2]
    h_odd = h[..., 1:d2:2]
    h_rot[..., :d2:2] = h_even * cos_rot - h_odd * sin_rot
    h_rot[..., 1:d2:2] = h_even * sin_rot + h_odd * cos_rot

    # O(2) blend
    cos_sum = (gate * cos_t[..., :n_instr]).sum(dim=-1, keepdim=True)
    sin_gate = sin_t[..., :n_instr].unsqueeze(-1) * gate.unsqueeze(-1)
    delta_prog = (sin_gate * prog_raw).sum(dim=2)
    h_mix_theta = h_mix * sin_t_conv
    update = delta_prog + h_mix_theta
    if activation == 'gelu':
        update = F.gelu(update, approximate='tanh')
    elif activation == 'relu':
        update = F.relu(update)
    elif activation == 'silu':
        update = F.silu(update)

    return h_rot * cos_sum + update
