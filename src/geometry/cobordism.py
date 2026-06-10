# 作用: 配边几何 — 相邻 token 拓扑平滑 + O(2) 旋转
# 数学: ∂W = M ⊔ N → h[t+1] 和 h[t] 在"配边"中有光滑边界
# 调用: core.py → geometry/__init__.py → get_geometry('cob')
import math, torch, torch.nn as nn, torch.nn.functional as F


class CobProgram(nn.Module):
    def __init__(self, d_model, n_instr=6, **kwargs):
        super().__init__()
        self.n_instr = n_instr
        self.weight = nn.Parameter(torch.randn(n_instr, d_model, d_model) * 0.02)

    def forward_raw(self, x):
        B, T, d = x.shape
        return torch.einsum('kij,btj->btki', self.weight, x)


def cob_residual(h, cos_t, sin_t, gate, prog_raw, h_mix, sin_t_conv, n_instr, activation='gelu'):
    """配边残差: O(2) 旋转 + 相邻 token 拓扑平滑.

    核心: ∂W = M ⊔ N — 配边的"边界"由前一个 token 的状态 M
          和后一个 token 的状态 N 组成.
    实现: h'_t 包含 h_{t-1} 的边界信息 (平滑混合), 保证配边光滑.
    """
    B, T, d = h.shape

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

    h_rot = h * cos_sum + update

    # 因果配边平滑: ∂W = M ⊔ N — 只从过去流向现在
    # h[t] = (1-λ)·h_rot[t] + λ·h_rot[t-1]  (单向, 因果)
    h_cob = h_rot.clone()
    h_cob[:, 1:] = h_cob[:, 1:] * 0.95 + h_rot[:, :-1] * 0.05

    return h_cob
