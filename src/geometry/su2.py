# 作用: SU(2) 四元数几何 — 4维组四元数旋量旋转 + O(2) 残差
# 调用: core.py → geometry/__init__.py → get_geometry('su2')
# 不变量: ∥q∥² (每个四元数范数)
import math, torch, torch.nn as nn, torch.nn.functional as F


class SU2Program(nn.Module):
    """SU(2) 指令: K个d×d线性变换 (同 O2Program)"""
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


def su2_residual(h, cos_t, sin_t, gate, prog_raw, h_mix, sin_t_conv, n_instr, activation='gelu'):
    """SU(2) 残差: 4维组四元数旋量旋转 + O(2) 旋转混合.

    两步:
      1. h 内部做四元数旋转: q' = u·q·u⁻¹ (保 ∥q∥²)
      2. O(2) 残差: h' = h_rot·cosθ + update·sinθ
    """
    B, T, d = h.shape
    n = (d // 4) * 4

    # 加权旋转角 → 单位四元数 u
    theta = (gate * sin_t[..., :n_instr]).sum(dim=-1) * (math.pi / 2)
    half = theta * 0.5
    qw = torch.cos(half)
    qi = torch.sin(half) * 0.577
    qj = torch.sin(half) * 0.577
    qk = torch.sin(half) * 0.577

    h_rot = h.clone()
    for g in range(n // 4):
        a, b, c, d_ = (h[..., g*4], h[..., g*4+1], h[..., g*4+2], h[..., g*4+3])
        aw, ax, ay, az = qw, qi, qj, qk
        # Hamilton: u * q
        ra = aw*a - ax*b - ay*c - az*d_
        rb = aw*b + ax*a + ay*d_ - az*c
        rc = aw*c - ax*d_ + ay*a + az*b
        rd = aw*d_ + ax*c - ay*b + az*a
        # (u*q) * ū
        h_rot[..., g*4]   = ra*aw + rb*ax + rc*ay + rd*az
        h_rot[..., g*4+1] = -ra*ax + rb*aw + rc*az - rd*ay
        h_rot[..., g*4+2] = -ra*ay - rb*az + rc*aw + rd*ax
        h_rot[..., g*4+3] = -ra*az + rb*ay - rc*ax + rd*aw

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
