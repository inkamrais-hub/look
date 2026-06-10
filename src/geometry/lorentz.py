# 作用: Lorentz SO(1,n-1) 双曲几何 — 时间-特征双曲旋转
# 调用: core.py → geometry/__init__.py → get_geometry('lorentz')
# 不变量: t² - Σx² (Minkowski 内积, 保因果结构)
import math, torch, torch.nn as nn, torch.nn.functional as F


class LorentzProgram(nn.Module):
    """Lorentz 指令: K个d×d线性变换"""
    def __init__(self, d_model, n_instr=6, **kwargs):
        super().__init__()
        self.n_instr = n_instr
        self.weight = nn.Parameter(torch.randn(n_instr, d_model, d_model) * 0.02)

    def forward_raw(self, x):
        B, T, d = x.shape
        return torch.einsum('kij,btj->btki', self.weight, x)


def lorentz_residual(h, cos_t, sin_t, gate, prog_raw, h_mix, sin_t_conv, n_instr, activation='gelu'):
    """Lorentz 残差: O(2) + 时序特征双曲分离.

    第一个维度=时间t, 其余=空间x.
    h' = h·cosθ + update·sinθ (standard O2)
    额外: 用 sigmoid(rapidity) 做温和的双曲混合.
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

    # Lorentz 双曲混合: rapidity φ 来自 sin_t_conv [B,T,1]
    phi = torch.sigmoid(sin_t_conv.squeeze(-1)) * 0.5  # [B,T], φ ∈ (0, 0.5)
    h_time = h_rot[..., :1]
    h_space = h_rot[..., 1:]
    # boost: t' = t·coshφ + h_space[0]·sinhφ
    # But we need to pick one spatial dimension for the boost.
    # Use the first spatial dimension as boost direction.
    h_t = h_time * torch.cosh(phi).unsqueeze(-1) + h_space[..., :1] * torch.sinh(phi).unsqueeze(-1)
    h_s = h_space * torch.cosh(phi).unsqueeze(-1)
    h_s[..., :1] = h_space[..., :1] * torch.cosh(phi).unsqueeze(-1)

    return torch.cat([h_t, h_s], dim=-1)
