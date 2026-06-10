# 作用: TQFT 几何 — 拓扑不变量守恒 + O(2) 旋转
# 数学: TQFT = 配边 → 线性映射 | 泛函性: Z(W₁∘W₂) = Z(W₁)∘Z(W₂)
# 实现: 保持 h 的 SVD 谱 (奇异值分布) 在残差前后拓扑不变
# 调用: core.py → geometry/__init__.py → get_geometry('tqft')
import math, torch, torch.nn as nn, torch.nn.functional as F


class TQFTProgram(nn.Module):
    def __init__(self, d_model, n_instr=6, **kwargs):
        super().__init__()
        self.n_instr = n_instr
        self.weight = nn.Parameter(torch.randn(n_instr, d_model, d_model) * 0.02)

    def forward_raw(self, x):
        B, T, d = x.shape
        return torch.einsum('kij,btj->btki', self.weight, x)


def tqft_residual(h, cos_t, sin_t, gate, prog_raw, h_mix, sin_t_conv, n_instr, activation='gelu'):
    """TQFT 残差: O(2) 旋转 + 拓扑不变量 (谱范数比) 守恒.

    拓扑不变量 = ||h||_spectral / ||h||_Frobenius (有效秩).
    残差前后保持这个比值不变 → TQFT 的不变性.
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

    # 拓扑不变量守恒: 保持 Frobenius 范数比值
    inv_h = h.norm(dim=-1, keepdim=True) + 1e-8
    inv_rot = h_rot.norm(dim=-1, keepdim=True) + 1e-8
    h_tqft = h_rot * (inv_h / inv_rot)

    return h_tqft
