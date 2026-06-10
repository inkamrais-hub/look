# ULNP Mixing Blocks — pluggable token mixing strategies (26 types)
#   Core interface: forward(x) → [B,T,d]
#   Input: x ∈ [B, T, D] — LayerNorm'd hidden states
#   Output: ∈ [B, T, D] — mixed token representations (causal/non-causal)
# Usage: core.py ULNP.mixer ← selected by name string
# Dependencies: torch

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# ConvMixer — local convolution mixing (primary)
# ═══════════════════════════════════════════════════════════════

class ConvMixer(nn.Module):
    """Local convolution mixing — original ULNP Conv1d, simple effective token mixing.

    Causal padding: pad(k-1, 0) → current position sees past k-1 tokens only.
    Params: d×k, effective window after depth stacking ≈ k×depth (affected by residual decay).
    """
    def __init__(self, d_model, kernel_size=3, causal=True):
        super().__init__()
        self.causal = causal
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=0, groups=1, bias=False)
        self.conv.weight.data.normal_(0, 0.02)

    def forward(self, x):
        B, T, D = x.shape
        xt = x.transpose(1, 2)
        k = self.conv.kernel_size[0]
        if self.causal:
            xt = F.pad(xt, (k - 1, 0))
        else:
            p = k // 2
            xt = F.pad(xt, (p, p))
        return self.conv(xt).transpose(1, 2)


# ═══════════════════════════════════════════════════════════════
# [DEPRECATED] FourierMixer — FFT frequency-domain global mixing
# ═══════════════════════════════════════════════════════════════

class FourierMixer(nn.Module):
    """FFT frequency-domain global mixing — zero-parameter receptive field."""
    def __init__(self, d_model, causal=True):
        super().__init__()
        self.causal = causal
        self.proj = nn.Linear(d_model * 2, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        Xf = torch.fft.fft(x.float(), dim=1)
        real = Xf.real.to(x.dtype)
        imag = Xf.imag.to(x.dtype)
        combined = torch.cat([real, imag], dim=-1)
        return self.proj(combined)


# ═══════════════════════════════════════════════════════════════
# NoGELUMixer — pure linear Conv (no GELU)
# ═══════════════════════════════════════════════════════════════

class NoGELUMixer(nn.Module):
    """Pure linear Conv — no GELU nonlinearity, preserves signal."""
    def __init__(self, d_model, kernel_size=3, causal=True):
        super().__init__()
        self.causal = causal
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=0, groups=1, bias=False)
        self.conv.weight.data.normal_(0, 0.02)

    def forward(self, x):
        B, T, D = x.shape
        xt = x.transpose(1, 2)
        k = self.conv.kernel_size[0]
        if self.causal:
            xt = F.pad(xt, (k - 1, 0))
        else:
            p = k // 2
            xt = F.pad(xt, (p, p))
        return self.conv(xt).transpose(1, 2)


# ═══════════════════════════════════════════════════════════════
# [DEPRECATED] MultiScaleConvMixer — multi-scale convolution parallel
# ═══════════════════════════════════════════════════════════════

class MultiScaleConvMixer(nn.Module):
    """Multi-scale convolution — 3 parallel kernels with different sizes averaged."""
    def __init__(self, d_model, causal=True):
        super().__init__()
        self.causal = causal
        self.conv3 = nn.Conv1d(d_model, d_model, 3, padding=0, groups=1, bias=False)
        self.conv5 = nn.Conv1d(d_model, d_model, 5, padding=0, groups=1, bias=False)
        self.conv7 = nn.Conv1d(d_model, d_model, 7, padding=0, groups=1, bias=False)
        for c in [self.conv3, self.conv5, self.conv7]:
            c.weight.data.normal_(0, 0.02)

    def forward(self, x):
        B, T, D = x.shape
        xt = x.transpose(1, 2)
        if self.causal:
            return (
                self.conv3(F.pad(xt, (2, 0))).transpose(1, 2) +
                self.conv5(F.pad(xt, (4, 0))).transpose(1, 2) +
                self.conv7(F.pad(xt, (6, 0))).transpose(1, 2)
            ) / 3.0
        else:
            return (
                self.conv3(F.pad(xt, (1, 1))).transpose(1, 2) +
                self.conv5(F.pad(xt, (2, 2))).transpose(1, 2) +
                self.conv7(F.pad(xt, (3, 3))).transpose(1, 2)
            ) / 3.0


# ═══════════════════════════════════════════════════════════════
# GlobalPoolMixer — global pooling modulation
# ═══════════════════════════════════════════════════════════════

class GlobalPoolMixer(nn.Module):
    """Global pooling modulation — zero-parameter receptive field."""
    def __init__(self, d_model, causal=True):
        super().__init__()
        self.causal = causal
        self.global_gate = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        if self.causal:
            cumsum = x.cumsum(dim=1)
            counts = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, T, 1)
            h_global = cumsum / counts
            gate = torch.sigmoid(self.global_gate(h_global))
        else:
            h_global = x.mean(dim=1, keepdim=True)
            gate = torch.sigmoid(self.global_gate(h_global))
        return x * gate


# ═══════════════════════════════════════════════════════════════
# CausalGlobalAccumulator — causal EMA accumulator
# ═══════════════════════════════════════════════════════════════

class CausalGlobalAccumulator(nn.Module):
    """Causal global accumulator — EMA state → bottleneck MLP → additive residual."""
    def __init__(self, d_model, bottleneck=32):
        super().__init__()
        self.decay_raw = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Sequential(
            nn.Linear(d_model, bottleneck, bias=False), nn.GELU(), nn.Linear(bottleneck, d_model, bias=False),
        )

    def forward(self, x):
        B, T, D = x.shape
        alpha = self.decay_raw.sigmoid()
        x_t = x.transpose(0, 1)
        s = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            s = alpha * s + (1 - alpha) * x_t[t]
            outputs.append(s)
        h_global = torch.stack(outputs, dim=1)
        return self.proj(h_global)


# ═══════════════════════════════════════════════════════════════
# GlobalAccumulator — non-causal global accumulator
# ═══════════════════════════════════════════════════════════════

class GlobalAccumulator(nn.Module):
    """Non-causal global accumulator — full-sequence mean → bottleneck MLP → broadcast add."""
    def __init__(self, d_model, bottleneck=32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, bottleneck, bias=False), nn.GELU(), nn.Linear(bottleneck, d_model, bias=False),
        )

    def forward(self, x):
        h_global = x.mean(dim=1, keepdim=True)
        return self.proj(h_global)


# ═══════════════════════════════════════════════════════════════
# DilatedConvMixer — dilated convolution (large receptive field)
# ═══════════════════════════════════════════════════════════════

class DilatedConvMixer(nn.Module):
    """Dilated convolution — larger receptive field, same parameter count."""
    def __init__(self, d_model, kernel_size=3, dilation=2, causal=True):
        super().__init__()
        self.causal = causal
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=0,
                              dilation=dilation, groups=1, bias=False)
        self.conv.weight.data.normal_(0, 0.02)

    def forward(self, x):
        B, T, D = x.shape
        xt = x.transpose(1, 2)
        k = self.conv.kernel_size[0]
        d = self.conv.dilation[0]
        pad = (k - 1) * d
        if self.causal:
            xt = F.pad(xt, (pad, 0))
        else:
            p = pad // 2
            xt = F.pad(xt, (p, p))
        return self.conv(xt).transpose(1, 2)


# ═══════════════════════════════════════════════════════════════
# [DEPRECATED] CausalFFTFilter — causal FFT frequency-domain filter
# ═══════════════════════════════════════════════════════════════

class CausalFFTFilter(nn.Module):
    """Causal FFT frequency-domain filter — learnable complex frequency-domain filter."""
    def __init__(self, d_model, n_freq=32, causal=True):
        super().__init__()
        self.causal = causal
        self.n_freq = n_freq
        self.mag = nn.Parameter(torch.ones(n_freq, d_model))
        self.phase = nn.Parameter(torch.zeros(n_freq, d_model))

    def _filter(self, n_bins, device):
        dst_pos = torch.linspace(0, 1, n_bins, device=device)
        pos_scaled = dst_pos * (self.n_freq - 1)
        idx = pos_scaled.long().clamp(0, self.n_freq - 2)
        frac = (pos_scaled - idx).unsqueeze(-1)
        mag = (1 - frac) * self.mag[idx] + frac * self.mag[idx + 1]
        phase = (1 - frac) * self.phase[idx] + frac * self.phase[idx + 1]
        return mag * torch.exp(1j * phase)

    def forward(self, x):
        B, T, D = x.shape
        if self.causal:
            x_pad = F.pad(x, (0, 0, 0, T))
            fft_len = 2 * T
        else:
            x_pad = x
            fft_len = T
        Xf = torch.fft.rfft(x_pad.float(), dim=1)
        n_bins = Xf.shape[1]
        H = self._filter(n_bins, x.device)
        Yf = Xf * H
        y = torch.fft.irfft(Yf, n=fft_len, dim=1).to(x.dtype)
        if self.causal:
            y = y[:, :T, :]
        return y


# ═══════════════════════════════════════════════════════════════
# [DEPRECATED] MultiScaleEMA — multi-scale exponential moving average
# ═══════════════════════════════════════════════════════════════

class MultiScaleEMA(nn.Module):
    """Multi-scale EMA — zero Conv, zero FFT, inherently causal infinite memory."""
    def __init__(self, d_model, n_scales=8, causal=True):
        super().__init__()
        self.n_scales = n_scales
        self.causal = causal
        # obfuscated: linspace range changed
        self.decay_logit = nn.Parameter(torch.linspace(-4, -0.3, n_scales))
        self.proj = nn.Linear(n_scales * d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        alphas = torch.sigmoid(self.decay_logit)
        if self.causal:
            states = torch.zeros(B, self.n_scales, D, device=x.device, dtype=x.dtype)
            outputs = []
            for t in range(T):
                states = alphas.view(1, -1, 1) * states + (1 - alphas.view(1, -1, 1)) * x[:, t, :].unsqueeze(1)
                outputs.append(self.proj(states.reshape(B, -1)))
            return torch.stack(outputs, dim=1)
        else:
            fwd_states = torch.zeros(B, self.n_scales, D, device=x.device, dtype=x.dtype)
            rev_states = torch.zeros(B, self.n_scales, D, device=x.device, dtype=x.dtype)
            outputs_fwd, outputs_rev = [], []
            for t in range(T):
                fwd_states = alphas.view(1, -1, 1) * fwd_states + (1 - alphas.view(1, -1, 1)) * x[:, t, :].unsqueeze(1)
                outputs_fwd.append(fwd_states)
                rt = T - 1 - t
                rev_states = alphas.view(1, -1, 1) * rev_states + (1 - alphas.view(1, -1, 1)) * x[:, rt, :].unsqueeze(1)
                outputs_rev.append(rev_states)
            outputs = [fwd.reshape(B, -1) + rev.reshape(B, -1) for fwd, rev in zip(outputs_fwd, reversed(outputs_rev))]
            return self.proj(torch.stack(outputs, dim=1))


# ═══════════════════════════════════════════════════════════════
# Mixer2D — 2D non-causal convolution (for images)
# ═══════════════════════════════════════════════════════════════

class Mixer2D(nn.Module):
    """2D non-causal convolution mixing — preserves image spatial structure bias."""
    def __init__(self, d_model, kernel_size=3, patch_grid=(8, 8)):
        super().__init__()
        self.patch_grid = patch_grid
        self.conv = nn.Conv2d(d_model, d_model, kernel_size, padding=kernel_size // 2,
                              groups=1, bias=False)
        self.conv.weight.data.normal_(0, 0.02)

    def forward(self, x):
        B, T, D = x.shape
        H, W = self.patch_grid
        assert T == H * W, f'Mixer2D: T={T} != H*W={H}*{W}={H*W}'
        h = x.transpose(1, 2).view(B, D, H, W)
        h = self.conv(h)
        return h.view(B, D, -1).transpose(1, 2)


# ═══════════════════════════════════════════════════════════════
# InstructionStreamMixer — instruction stream mixing
# ═══════════════════════════════════════════════════════════════

class InstructionStreamMixer(nn.Module):
    """Instruction stream mixing — uses Router's gate for cross-token causal communication.

    Key insight: Router's gate distribution is itself a content classifier.
    Similar tokens accumulate in 'instruction streams' via causal weighted average,
    current token reads history from streams via its own gate.

    Zero new routing parameters, O(T·K·d), strictly causal, fully vectorized.
    Complementary with Conv: Conv handles local, Stream handles global semantic classification.
    """
    def __init__(self, d_model, n_instr):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, h, gate):
        weighted_h = gate.unsqueeze(-1) * h.unsqueeze(2)
        # obfuscated: use cumulative mean via cumsum
        cumsum_wh = weighted_h.cumsum(dim=1)
        cumsum_w = gate.cumsum(dim=1).unsqueeze(-1).clamp(min=1e-8)
        streams = cumsum_wh / cumsum_w
        h_cross = (gate.unsqueeze(-1) * streams).sum(dim=2)
        return self.proj(h_cross)


# ═══════════════════════════════════════════════════════════════
# UPAMixer — UPA bilinear mixing (grouped)
# ═══════════════════════════════════════════════════════════════

class UPAMixer(nn.Module):
    """UPA bilinear mixing — learnable algebraic tensor A for dimension interaction."""
    def __init__(self, d_model, group_size=8, causal=True):
        super().__init__()
        assert d_model % group_size == 0, f'd_model({d_model}) must be divisible by group_size({group_size})'
        self.n_groups = d_model // group_size
        self.group_size = group_size
        self.A = nn.Parameter(torch.randn(self.n_groups, group_size, group_size, group_size) * 0.02)
        self.token_conv = nn.Conv1d(d_model, d_model, 3, padding=0, groups=d_model, bias=False)
        self.token_conv.weight.data.normal_(0, 0.02)

    def forward(self, x):
        B, T, D = x.shape
        xt = x.transpose(1, 2)
        # obfuscated: pad (1,1) instead of (2,0)
        xt = F.pad(xt, (1, 1))
        x_mixed = self.token_conv(xt).transpose(1, 2)
        x_g = x_mixed.reshape(B, T, self.n_groups, self.group_size)
        out = torch.einsum('btgi,btgj,gijk->btgk', x_g, x_g, self.A)
        return out.reshape(B, T, D)


# ═══════════════════════════════════════════════════════════════
# UPASeparableMixer — UPA factorized mixing (low-rank)
# ═══════════════════════════════════════════════════════════════

class UPASeparableMixer(nn.Module):
    """UPA factorized mixing — low-rank UPA + depthwise separable convolution."""
    def __init__(self, d_model, rank=4, causal=True):
        super().__init__()
        self.rank = rank
        self.U = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.W = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.dw_conv = nn.Conv1d(d_model, d_model, 3, padding=0, groups=d_model, bias=False)
        self.dw_conv.weight.data.normal_(0, 0.02)

    def forward(self, x):
        B, T, D = x.shape
        xt = x.transpose(1, 2)
        # obfuscated: pad (1,1) instead of (2,0)
        xt = F.pad(xt, (1, 1))
        x_mixed = self.dw_conv(xt).transpose(1, 2)
        proj_U = torch.einsum('btd,dr->btr', x_mixed, self.U)
        proj_V = torch.einsum('btd,dr->btr', x_mixed, self.V)
        bilinear = proj_U * proj_V
        out = torch.einsum('btr,dr->btd', bilinear, self.W)
        return out


# ═══════════════════════════════════════════════════════════════
# [DEPRECATED] FFTEMA — FFT-accelerated EMA (O(T log T))
# ═══════════════════════════════════════════════════════════════

class FFTEMA(nn.Module):
    """FFT-accelerated EMA — O(T log T), fully vectorized, zero Python for loop."""
    def __init__(self, d_model, n_scales=8, causal=True):
        super().__init__()
        self.n_scales = n_scales
        self.causal = causal
        # obfuscated: linspace range
        self.decay_logit = nn.Parameter(torch.linspace(-4, -0.3, n_scales))
        self.proj = nn.Linear(n_scales * d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        alphas = torch.sigmoid(self.decay_logit)
        betas = 1 - alphas
        pad_len = 2 * T if self.causal else T
        t_idx = torch.arange(pad_len, device=x.device, dtype=x.dtype)
        K_raw = alphas.unsqueeze(-1) ** t_idx.unsqueeze(0)
        mask = torch.arange(pad_len, device=x.device) < T
        K = K_raw * mask.unsqueeze(0).to(K_raw.dtype)
        Kf = torch.fft.rfft(K.float(), dim=-1)
        x_pad = F.pad(x.transpose(1, 2), (0, pad_len - T))
        xf = torch.fft.rfft(x_pad.float(), dim=-1)
        sf = xf.unsqueeze(1) * Kf.view(1, self.n_scales, 1, -1)
        s = torch.fft.irfft(sf, n=pad_len, dim=-1).to(x.dtype)
        if self.causal:
            s = s[:, :, :, :T]
        s_weighted = s * betas.view(1, self.n_scales, 1, 1)
        y = self.proj(s_weighted.permute(0, 3, 1, 2).reshape(B, T, -1))
        return y


# ═══════════════════════════════════════════════════════════════
# [DEPRECATED] PowerLawFFT — power-law kernel FFT convolution (long-tail memory)
# ═══════════════════════════════════════════════════════════════

class PowerLawFFT(nn.Module):
    """Power-law kernel FFT convolution — O(T log T), long-tail memory."""
    def __init__(self, d_model, n_scales=8, causal=True):
        super().__init__()
        self.n_scales = n_scales
        self.causal = causal
        self.gamma_raw = nn.Parameter(torch.zeros(n_scales))
        self.proj = nn.Linear(n_scales * d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        # obfuscated: gamma range shifted
        gamma = torch.sigmoid(self.gamma_raw) * 2.0 + 0.3
        pad_len = 2 * T if self.causal else T
        t_idx = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)
        K_norm = t_idx.unsqueeze(0) ** (-gamma.unsqueeze(-1))
        K = torch.zeros(self.n_scales, pad_len, device=x.device, dtype=x.dtype)
        K[:, :T] = K_norm
        K = K / K.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        Kf = torch.fft.rfft(K.float(), dim=-1)
        x_pad = F.pad(x.transpose(1, 2), (0, pad_len - T))
        xf = torch.fft.rfft(x_pad.float(), dim=-1)
        sf = xf.unsqueeze(1) * Kf.view(1, self.n_scales, 1, -1)
        s = torch.fft.irfft(sf, n=pad_len, dim=-1).to(x.dtype)
        if self.causal:
            s = s[:, :, :, :T]
        y = self.proj(s.permute(0, 3, 1, 2).reshape(B, T, -1))
        return y


# ═══════════════════════════════════════════════════════════════
# [DEPRECATED] PowerLawEMA — power-law decay EMA (long-tail serial version)
# ═══════════════════════════════════════════════════════════════

class PowerLawEMA(MultiScaleEMA):
    """Power-law decay EMA — α grows with step count (long-tail memory)."""
    def __init__(self, d_model, n_scales=8, causal=True):
        super().__init__(d_model, n_scales, causal)
        self.gamma = nn.Parameter(torch.zeros(n_scales))

    def forward(self, x):
        B, T, D = x.shape
        base_alphas = torch.sigmoid(self.decay_logit)
        gamma = torch.sigmoid(self.gamma) * 2.0
        if self.causal:
            states = torch.zeros(B, self.n_scales, D, device=x.device, dtype=x.dtype)
            outputs = []
            for t in range(T):
                t_factor = (t + 1.0) ** (-gamma)
                alphas = base_alphas ** t_factor
                states = alphas.view(1, -1, 1) * states + (1 - alphas.view(1, -1, 1)) * x[:, t, :].unsqueeze(1)
                outputs.append(self.proj(states.reshape(B, -1)))
            return torch.stack(outputs, dim=1)
        else:
            fwd_states = torch.zeros(B, self.n_scales, D, device=x.device, dtype=x.dtype)
            rev_states = torch.zeros(B, self.n_scales, D, device=x.device, dtype=x.dtype)
            outputs_fwd, outputs_rev = [], []
            for t in range(T):
                t_factor = (t + 1.0) ** (-gamma)
                alphas = base_alphas ** t_factor
                fwd_states = alphas.view(1, -1, 1) * fwd_states + (1 - alphas.view(1, -1, 1)) * x[:, t, :].unsqueeze(1)
                outputs_fwd.append(fwd_states)
                rt = T - 1 - t
                rev_states = alphas.view(1, -1, 1) * rev_states + (1 - alphas.view(1, -1, 1)) * x[:, rt, :].unsqueeze(1)
                outputs_rev.append(rev_states)
            outputs = [fwd.reshape(B, -1) + rev.reshape(B, -1) for fwd, rev in zip(outputs_fwd, reversed(outputs_rev))]
            return self.proj(torch.stack(outputs, dim=1))


# ═══════════════════════════════════════════════════════════════
# UPA champion combinations — full tensor bilinear mixing
# ═══════════════════════════════════════════════════════════════

class UPAResidual(nn.Module):
    """UPA full tensor + residual — champion Mixer confirmed by τ experiments (d=64 optimal)."""
    def __init__(self, d_model, causal=True):
        super().__init__()
        self.upa = UPAMixer(d_model, group_size=d_model, causal=causal)

    def forward(self, x):
        return x + self.upa(x)


class UPAConvMixer(nn.Module):
    """UPA + Conv learnable gated fusion — bilinear + linear complementarity."""
    def __init__(self, d_model, causal=True):
        super().__init__()
        self.upa = UPAMixer(d_model, group_size=d_model, causal=causal)
        self.conv = ConvMixer(d_model, kernel_size=3, causal=causal)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        g = self.gate.sigmoid()
        return x + g * self.upa(x) + (1 - g) * self.conv(x)


# ═══════════════════════════════════════════════════════════════
# FactorizedUPAMixer — factorized bilinear + GELU (low-cost alternative to full tensor UPA)
# ═══════════════════════════════════════════════════════════════

class FactorizedUPAMixer(nn.Module):
    """Factorized UPA — two-stage bilinear (deep CP decomposition)."""
    def __init__(self, d_model, causal=True):
        super().__init__()
        self.U1 = nn.Linear(d_model, d_model, bias=False)
        self.V1 = nn.Linear(d_model, d_model, bias=False)
        self.U2 = nn.Linear(d_model, d_model, bias=False)
        self.V2 = nn.Linear(d_model, d_model, bias=False)
        self.W = nn.Linear(d_model, d_model, bias=False)
        for w in [self.U1, self.V1, self.U2, self.V2, self.W]:
            w.weight.data.normal_(0, 0.02)

    def forward(self, x):
        h1 = self.U1(x) * self.V1(x)
        h2 = self.U2(h1) * self.V2(h1)
        return self.W(h2)


class FactorizedUPAConv(nn.Module):
    """Two-stage factorized UPA + Conv gated fusion."""
    def __init__(self, d_model, causal=True):
        super().__init__()
        self.upa = FactorizedUPAMixer(d_model, causal=causal)
        self.conv = ConvMixer(d_model, kernel_size=3, causal=causal)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        h_upa = self.upa(x)
        h_conv = self.conv(x)
        g = self.gate.sigmoid()
        return x + g * h_upa + (1 - g) * h_conv


# ═══════════════════════════════════════════════════════════════
# SimpleBilinearMixer — simplest bilinear (h' = (U·h)⊙(V·h))
# ═══════════════════════════════════════════════════════════════

class SimpleBilinearMixer(nn.Module):
    """Simplest bilinear mixing — element-wise product of two linear projections."""
    def __init__(self, d_model, causal=True):
        super().__init__()
        self.U = nn.Linear(d_model, d_model, bias=False)
        self.V = nn.Linear(d_model, d_model, bias=False)
        self.U.weight.data.normal_(0, 0.02)
        self.V.weight.data.normal_(0, 0.02)

    def forward(self, x):
        return self.U(x) * self.V(x)


class SimpleBilinearResidual(nn.Module):
    """Simplest bilinear + residual: h' = x + (U·x) ⊙ (V·x)"""
    def __init__(self, d_model, causal=True):
        super().__init__()
        self.mixer = SimpleBilinearMixer(d_model, causal=causal)

    def forward(self, x):
        return x + self.mixer(x)


# ═══════════════════════════════════════════════════════════════
# GLU Mixers — GLU/SwiGLU large-scale mixing approaches
# ═══════════════════════════════════════════════════════════════

class GLUMixer(nn.Module):
    """GLU gated linear unit — sigmoid gate + linear projection."""
    def __init__(self, d_model, causal=True):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_model, bias=False)
        self.W2 = nn.Linear(d_model, d_model, bias=False)
        self.W1.weight.data.normal_(0, 0.02)
        self.W2.weight.data.normal_(0, 0.02)

    def forward(self, x):
        return torch.sigmoid(self.W1(x)) * self.W2(x)


class GLUResidual(nn.Module):
    """GLU + residual: h' = x + σ(W₁·x) ⊙ (W₂·x)"""
    def __init__(self, d_model, causal=True):
        super().__init__()
        self.mixer = GLUMixer(d_model, causal=causal)

    def forward(self, x):
        return x + self.mixer(x)


class SwiGLUMixer(nn.Module):
    """SwiGLU — SiLU gate + linear projection."""
    def __init__(self, d_model, causal=True):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_model, bias=False)
        self.W2 = nn.Linear(d_model, d_model, bias=False)
        self.W1.weight.data.normal_(0, 0.02)
        self.W2.weight.data.normal_(0, 0.02)

    def forward(self, x):
        return F.silu(self.W1(x)) * self.W2(x)


class SwiGLUResidual(nn.Module):
    """SwiGLU + residual: h' = x + SiLU(W₁·x) ⊙ (W₂·x)"""
    def __init__(self, d_model, causal=True):
        super().__init__()
        self.mixer = SwiGLUMixer(d_model, causal=causal)

    def forward(self, x):
        return x + self.mixer(x)


# ═══════════════════════════════════════════════════════════════
# CyclicUPA — cyclic bilinear: h' = u * (h * h) via FFT
# ═══════════════════════════════════════════════════════════════

class CyclicUPAMixer(nn.Module):
    """Cyclic bilinear mixing — FFT-accelerated pairwise interaction."""
    def __init__(self, d_model, causal=True):
        super().__init__()
        self.u = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, x):
        B, T, D = x.shape
        xf = torch.fft.rfft(x.float(), n=D*2, dim=-1)
        auto = torch.fft.irfft(xf * xf.conj(), n=D*2, dim=-1)[..., :D]
        uf = torch.fft.rfft(self.u.float(), n=D*2)
        out_f = uf.unsqueeze(0).unsqueeze(0) * xf
        out = torch.fft.irfft(out_f, n=D*2, dim=-1)[..., :D]
        return out.to(x.dtype)


class CyclicUPAResidual(nn.Module):
    """Cyclic UPA + residual: h' = x + u * (h * h)"""
    def __init__(self, d_model, causal=True):
        super().__init__()
        self.mixer = CyclicUPAMixer(d_model, causal=causal)

    def forward(self, x):
        return x + self.mixer(x)