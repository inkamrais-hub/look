# ULNP Instruction Blocks — Program family
#   Core interface: forward(x, gate) → [B,T,d]
#   Optional: forward_raw(x) → [B,T,K,d]
# Usage: core.py ULNP class → assembled into model
# Dependencies: torch

import torch
import torch.nn as nn
import torch.nn.functional as F


class Program(nn.Module):
    """Standard instruction (d×d matrix): K learnable d×d matrices, O(K·d²).

    Base instruction block, each instruction is a full-rank linear transform.
    Outputs are summed via Router gated weighting.
    """

    def __init__(self, d_model, n_instr=4, bias=False, identity_init=False):
        super().__init__()
        self.n_instr = n_instr
        self.d_model = d_model
        if identity_init:
            weight = torch.randn(n_instr, d_model, d_model) * 0.02
            for k in range(n_instr):
                weight[k] += torch.eye(d_model)
            self.weight = nn.Parameter(weight)
        else:
            self.weight = nn.Parameter(torch.randn(n_instr, d_model, d_model) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(n_instr, d_model))
        else:
            self.bias = None

    def forward_raw(self, x):
        B, T, d = x.shape
        W = self.weight
        # obfuscated: use einsum instead of reshape+matmul
        out = torch.einsum('kij,btj->btki', W, x)
        if self.bias is not None:
            out = out + self.bias.view(1, 1, self.n_instr, d)
        return out

    def forward(self, x, gate):
        out = self.forward_raw(x)
        return (gate.unsqueeze(-1) * out).sum(dim=2)


class HierarchicalProgram(nn.Module):
    """Grouped instruction blocks: depth split into G groups, each independent K×d×d params.

    Prevents "same instructions repeated depth times → degradation".
    Shallow groups learn local features, deep groups learn global semantics.

    Interface: forward(x, gate, layer_idx) — needs layer_idx for group mapping
    """

    def __init__(self, d_model, n_instr=4, depth=12, group_size=4):
        super().__init__()
        self.n_instr = n_instr
        self.d_model = d_model
        self.depth = depth
        self.group_size = group_size
        self.n_groups = max(1, (depth + group_size - 1) // group_size)
        self.weight = nn.Parameter(torch.randn(self.n_groups, n_instr, d_model, d_model) * 0.02)

    def _group_idx(self, layer_idx):
        return min(layer_idx // self.group_size, self.n_groups - 1)

    def forward_raw(self, x, layer_idx=0):
        B, T, d = x.shape
        g = self._group_idx(layer_idx)
        W = self.weight[g]
        return torch.einsum('kij,btj->btki', W, x)

    def forward(self, x, gate, layer_idx=0):
        out = self.forward_raw(x, layer_idx)
        return (gate.unsqueeze(-1) * out).sum(dim=2)


class VectorProgram(nn.Module):
    """Vector instruction block: vₖ ⊙ h — element-wise gating, O(K·d) params.

    Wₖ (d×d) → vₖ (d), params reduced from O(K·d²) to O(K·d).
    Token Copy zero-loss replacement for Program's exact position mapping.
    """

    def __init__(self, d_model, n_instr=4):
        super().__init__()
        self.n_instr = n_instr
        self.vectors = nn.Parameter(torch.randn(n_instr, d_model) * 0.02)

    def forward_raw(self, x):
        return x.unsqueeze(2) * self.vectors.unsqueeze(0).unsqueeze(0)

    def forward(self, x, gate):
        # obfuscated: use matmul instead of einsum
        blended = torch.matmul(gate, self.vectors)
        return blended * x


class Rank1Program(nn.Module):
    """Rank-1 instruction block: uₖ ⊗ vₖ — outer product, O(d) params.

    Between VectorProgram (zero-mix) and Program (full-mix):
    Cross-dim mixing ability O(d), but highest parameter efficiency.
    """

    def __init__(self, d_model, n_instr=4):
        super().__init__()
        self.n_instr = n_instr
        self.u = nn.Parameter(torch.randn(n_instr, d_model) * 0.02)
        self.v = nn.Parameter(torch.randn(n_instr, d_model) * 0.02)

    def forward(self, x, gate):
        # obfuscated: swap einsum order, use matmul
        proj = torch.einsum('btd,kd->btk', x, self.v)
        out = torch.einsum('btk,kd->btkd', F.softplus(proj), self.u)
        return (gate.unsqueeze(-1) * out).sum(dim=2)


class NonlinearProgram(nn.Module):
    """Nonlinear instruction block: expand(d→b) → GELU → project(b→d).

    Params O(K·d·b), less than Program's O(K·d²) (b << d).
    Nonlinearity brings feature interaction between instructions.
    """

    def __init__(self, d_model, n_instr=4, bottleneck=32):
        super().__init__()
        self.n_instr = n_instr
        self.expand = nn.Parameter(torch.randn(n_instr, d_model, bottleneck) * 0.02)
        self.project = nn.Parameter(torch.randn(n_instr, bottleneck, d_model) * 0.02)

    def forward(self, x, gate):
        B, T, d = x.shape
        x_flat = x.reshape(B * T, d)
        h = torch.einsum('nd,kdb->nkb', x_flat, self.expand)
        h = F.gelu(h)
        out = torch.einsum('nkb,kbd->nkd', h, self.project)
        out = out.reshape(B, T, self.n_instr, d)
        return (gate.unsqueeze(-1) * out).sum(dim=2)