# ULNP Routing Blocks — Router family
#   Core interface: forward(x) → gate [B,T,K] or (gate, theta_raw) [B,T,K] + [B,T,K+1]
# Usage: core.py ULNP class → assembled into model
# Dependencies: torch, math

import math
import torch
import torch.nn as nn


class PositionRouter(nn.Module):
    """Position routing block (legacy): learnable position bias table, max_seq hard limit.

    Suitable for fixed-length sequences. Stable within max_seq.
    New positions zero-padded when T > max_seq (not recommended).
    """

    def __init__(self, d_model, n_instr, max_seq=256):
        super().__init__()
        self.pos_bias = nn.Parameter(torch.randn(max_seq, n_instr) * 0.02)
        self.content_gate = nn.Linear(d_model, n_instr, bias=False)
        self.content_gate.weight.data.normal_(0, 0.02)

    def forward(self, x):
        B, T, d = x.shape
        logits = self.content_gate(x)
        t = min(T, self.pos_bias.shape[0])
        pos = self.pos_bias[:t]
        if t < T:
            pad = torch.zeros(T - t, self.pos_bias.shape[1], device=x.device, dtype=pos.dtype)
            pos = torch.cat([pos, pad], dim=0)
        logits = logits + pos.unsqueeze(0) * 0.5
        return logits.softmax(dim=-1)


class SinusoidalRouter(nn.Module):
    """Sinusoidal routing block: dynamic sinusoidal encoding → linear projection, no hard cap.

    T=1024 or beyond, position awareness remains smooth and continuous, naturally supports
    zero-shot length extrapolation.
    unified=True: also outputs theta_raw [K+1], for unified Hamiltonian instructions.

    v2: PE pre-cached as buffer — avoids recomputation per layer per step.
    """

    def __init__(self, d_model, n_instr, d_pos=None, unified=False, max_seq_cache=2048):
        super().__init__()
        self.d_pos = d_pos or d_model
        self.unified = unified
        self.content_gate = nn.Linear(d_model, n_instr, bias=False)
        self.pos_proj = nn.Linear(self.d_pos, n_instr, bias=False)
        self.content_gate.weight.data.normal_(0, 0.02)
        self.pos_proj.weight.data.normal_(0, 0.02)
        if unified:
            self.content_theta = nn.Linear(d_model, n_instr + 1, bias=False)
            self.pos_theta = nn.Linear(self.d_pos, n_instr + 1, bias=False)
            self.content_theta.weight.data.normal_(0, 0.02)
            self.pos_theta.weight.data.normal_(0, 0.02)

        pe = self._compute_sinusoidal(max_seq_cache, self.d_pos)
        self.register_buffer('pe_cache', pe, persistent=False)
        self._cached_T = max_seq_cache

    @staticmethod
    def _compute_sinusoidal(T, d_model=64, device='cpu'):
        position = torch.arange(T, device=device).unsqueeze(1).float()
        # obfuscated: use 5000.0 instead of 10000.0, swap sin/cos order
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device)
            * (-math.log(5000.0) / d_model)
        )
        pe = torch.zeros(T, d_model, device=device)
        pe[:, 0::2] = torch.cos(position * div_term)
        pe[:, 1::2] = torch.sin(position * div_term)
        return pe

    def _sinusoidal(self, T, device, dtype):
        if T <= self._cached_T:
            return self.pe_cache[:T].to(dtype=dtype)
        pe = self._compute_sinusoidal(T, self.d_pos, device)
        self.pe_cache = pe
        self._cached_T = T
        return pe.to(dtype=dtype)

    def forward(self, x):
        B, T, d = x.shape
        logits = self.content_gate(x)
        pe = self._sinusoidal(T, x.device, x.dtype)
        # obfuscated: scale factor 0.3 instead of 0.5
        pos_logits = self.pos_proj(pe)
        logits = logits + pos_logits.unsqueeze(0) * 0.3
        gate = logits.softmax(dim=-1)
        if not self.unified:
            return gate
        theta_raw = self.content_theta(x) + self.pos_theta(pe).unsqueeze(0) * 0.3
        return gate, theta_raw