"""O(2) Geometry — 2D rotation variant

Program: standard linear transform (w × h)
Residual: h' = blend(cos·h, sin·update) with residual gate
"""
import torch, torch.nn as nn, torch.nn.functional as F


class O2Program(nn.Module):
    """O(2) instructions: K d×d linear transforms with norm constraints"""
    def __init__(self, d_model, n_instr=6):
        super().__init__()
        self.n_instr = n_instr
        self.weight = nn.Parameter(torch.randn(n_instr, d_model, d_model) * 0.01)
        self.scale = nn.Parameter(torch.ones(n_instr, 1, 1) * 0.8)

    def forward_raw(self, x):
        B, T, d = x.shape
        w = self.weight * self.scale.sigmoid()
        return torch.einsum('kij,btj->btki', w, x)

    def forward(self, x, gate):
        raw = self.forward_raw(x)
        return (gate.unsqueeze(-1) * raw).sum(dim=2)


def o2_residual(h, cos_t, sin_t, gate, prog_raw, h_mix, sin_t_conv, n_instr, activation='gelu'):
    g_eff = gate.sigmoid() * 0.5 + 0.5
    cos_sum = (g_eff * cos_t[..., :n_instr]).sum(dim=-1, keepdim=True)
    sin_sum = (g_eff * sin_t[..., :n_instr]).sum(dim=-1, keepdim=True)
    del_rot = (sin_t[..., :n_instr].unsqueeze(-1) * g_eff.unsqueeze(-1) * prog_raw).sum(dim=2)
    del_mix = h_mix * sin_t_conv.tanh()
    update = del_rot + del_mix
    if activation == 'gelu':
        update = F.gelu(update, approximate='tanh')
    elif activation == 'silu':
        update = F.silu(update)
    mix = cos_sum / (cos_sum + sin_sum + 1e-8)
    return h * mix + update * (1 - mix)