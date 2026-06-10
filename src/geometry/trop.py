# 作用: 热带几何 — max-plus 代数替代加权求和
# 调用: core.py → geometry/__init__.py → get_geometry('trop')
# 核心: h' = h·max_k(cos_k) + max_k(sin_k·prog_k)  (硬选择, 非软混合)
import math, torch, torch.nn as nn, torch.nn.functional as F


class TropProgram(nn.Module):
    """热带指令: K个d×d线性变换"""
    def __init__(self, d_model, n_instr=6, **kwargs):
        super().__init__()
        self.n_instr = n_instr
        self.weight = nn.Parameter(torch.randn(n_instr, d_model, d_model) * 0.02)

    def forward_raw(self, x):
        B, T, d = x.shape
        return torch.einsum('kij,btj->btki', self.weight, x)


def trop_residual(h, cos_t, sin_t, gate, prog_raw, h_mix, sin_t_conv, n_instr, activation='gelu'):
    """热带残差: max 替代 sum — 每条指令"赢者通吃".

    O(2): h' = Σ gate_k·cosθ_k · h  +  Σ gate_k·sinθ_k · prog_k
    Trop: h' =   max_k(cosθ_k) · h  +   max_k(sinθ_k · prog_k)
                 (沿指令维)
    """
    B, T, d = h.shape

    # 加权 cos/sin for ranking
    wcos = gate * cos_t[..., :n_instr]  # [B,T,K]
    wsin = sin_t[..., :n_instr].unsqueeze(-1) * gate.unsqueeze(-1)  # [B,T,K,d]

    # max along instruction dim
    cos_max = wcos.max(dim=-1, keepdim=True).values  # [B,T,1]
    update = (wsin * prog_raw).max(dim=2).values  # [B,T,d]

    h_mix_theta = h_mix * sin_t_conv
    update = update + h_mix_theta
    if activation == 'gelu':
        update = F.gelu(update, approximate='tanh')
    elif activation == 'relu':
        update = F.relu(update)
    elif activation == 'silu':
        update = F.silu(update)

    return h * cos_max + update
