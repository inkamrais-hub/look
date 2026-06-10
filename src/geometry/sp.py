# Sp(n) Symplectic Geometry — SymplecticLinear (Iwasawa KAN) + Symplectic Integrator (Størmer-Verlet)
# Usage: core.py → geometry/__init__.py → get_geometry('sp')
# Ported from: _archive/ulnp-qb/experiments/symplectic_deep/
# Dependencies: torch
# Invariant: det(J)=1, symplectic form ω = dq∧dp

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def pairing_pattern(n_pairs, layer_idx, n_layers):
    """4 rotation pairing patterns — ensures cross-dimension information mixing"""
    n = n_pairs
    if layer_idx % 4 == 0:
        return [(i, i) for i in range(n)]
    elif layer_idx % 4 == 1:
        return [(i, (i + 1) % n) for i in range(n)]
    elif layer_idx % 4 == 2:
        return [(i, (i + n // 2) % n) for i in range(n)] if n >= 2 else [(i, i) for i in range(n)]
    else:
        return [(i, n - 1 - i) for i in range(n)]


class SpProgram(nn.Module):
    """Symplectic program: Iwasawa KAN decomposition parametrization, each pair (θ, α, β) → 2×2 symplectic matrix.

    Theory:
      - Iwasawa decomposition Sp(2n,R) = K · A · N
      - K ≅ U(n) gradient flow on compact subgroup → θ≈0.41 universal attractor
      - det ≡ 1 self-consistent, independent of θ/α/β

    Per-layer pairing patterns:
      L%4=0: self-pairing (independence)
      L%4=1: adjacent pairing (local mixing)
      L%4=2: distant pairing (global communication)
      L%4=3: symmetric pairing (long-range interaction)

    Interface: forward_raw(x) → [B,T,n_instr,d], same interface as Program/forward_raw.
    """
    def __init__(self, d_model, n_instr=6, layer_idx=0, depth=4):
        super().__init__()
        assert d_model % 2 == 0, 'd_model must be even for Sp(n) phase-space split'
        self.d_model = d_model
        self.n_instr = n_instr
        self.n_pairs = d_model // 2
        self.pattern = pairing_pattern(self.n_pairs, layer_idx, depth)

        self.theta = nn.Parameter(torch.zeros(n_instr, self.n_pairs))
        self.alpha = nn.Parameter(torch.zeros(n_instr, self.n_pairs))
        self.beta = nn.Parameter(torch.zeros(n_instr, self.n_pairs))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.theta, 0, math.pi / 4)
        nn.init.normal_(self.alpha, 0, 0.1)
        nn.init.normal_(self.beta, 0, 0.1)

    def forward_raw(self, x):
        """Iwasawa KAN: x → K symplectic transforms → [B,T,n_instr,d]"""
        B, T, d = x.shape
        x_2d = x.reshape(B * T, self.n_pairs, 2)

        theta = torch.sigmoid(self.theta) * math.pi / 2
        alpha = self.alpha.tanh()
        beta = self.beta.tanh()

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        exp_a = torch.exp(alpha)
        exp_na = torch.exp(-alpha)

        # obfuscated: swap a11/a12/a21/a22 formula structure
        a11 = cos_t * exp_a
        a12 = cos_t * exp_a * beta - sin_t * exp_na
        a21 = sin_t * exp_a
        a22 = sin_t * exp_a * beta + cos_t * exp_na

        pairs_i = [p[0] for p in self.pattern]
        pairs_j = [p[1] for p in self.pattern]
        a11 = a11[:, pairs_j]
        a12 = a12[:, pairs_j]
        a21 = a21[:, pairs_j]
        a22 = a22[:, pairs_j]

        q = x_2d[..., pairs_i, 0]
        p = x_2d[..., pairs_i, 1]

        out_list = []
        for k in range(self.n_instr):
            qk = a11[k] * q + a12[k] * p
            pk = a21[k] * q + a22[k] * p
            out_list.append(torch.stack([qk, pk], dim=-1))

        out = torch.stack(out_list, dim=-2)
        return out.reshape(B, T, self.n_instr, d)

    def forward(self, x, gate):
        raw = self.forward_raw(x)
        return (gate.unsqueeze(-1) * raw).sum(dim=2)

    def symplectic_loss(self):
        """Symplectic regularization: ||det(S_k) - 1||² (averaged over K instructions)"""
        theta = torch.sigmoid(self.theta) * math.pi / 2
        alpha = self.alpha.tanh()
        beta = self.beta.tanh()
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        exp_a = torch.exp(alpha)
        exp_na = torch.exp(-alpha)
        a11 = cos_t * exp_a
        a12 = cos_t * exp_a * beta - sin_t * exp_na
        a21 = sin_t * exp_a
        a22 = sin_t * exp_a * beta + cos_t * exp_na
        det = a11 * a22 - a12 * a21
        return F.mse_loss(det, torch.ones_like(det))


def sp_residual(h, cos_t, sin_t, gate, prog_raw, h_mix, sin_t_conv, n_instr, activation='gelu'):
    """Størmer-Verlet leapfrog symplectic integrator (with damping — validated research version)"""
    B, T, d = h.shape
    d2 = d // 2
    q, p = h[..., :d2], h[..., d2:]

    prog_q = prog_raw[..., :d2]
    prog_p = prog_raw[..., d2:]

    mix_q = h_mix[..., :d2]
    mix_p = h_mix[..., d2:]

    dt_instr = sin_t[..., :n_instr]
    dt_conv = sin_t_conv.squeeze(-1)

    delta_q = torch.zeros_like(q)
    delta_p = torch.zeros_like(p)

    for k in range(n_instr):
        g_k = gate[..., k:k + 1]
        dt_k = dt_instr[..., k:k + 1]

        force_q = prog_q[..., k, :]
        force_p = prog_p[..., k, :]

        p_half = p + 0.5 * dt_k * force_q
        q_step = dt_k * (p_half + 0.1 * force_p)
        p_step = dt_k * force_q

        delta_q = delta_q + g_k * (q_step - 0.05 * q * dt_k)
        delta_p = delta_p + g_k * (p_step - 0.05 * p * dt_k)

    delta_q = delta_q + dt_conv.unsqueeze(-1) * mix_q
    delta_p = delta_p + dt_conv.unsqueeze(-1) * mix_p

    c_eff = (gate * cos_t[..., :n_instr]).sum(dim=-1, keepdim=True)

    q_new = q * c_eff + delta_q
    p_new = p * c_eff + delta_p

    if activation == 'gelu':
        q_new = F.gelu(q_new, approximate='tanh')
        p_new = F.gelu(p_new, approximate='tanh')

    return torch.cat([q_new, p_new], dim=-1)