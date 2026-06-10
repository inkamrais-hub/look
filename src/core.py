# ULNP Skeleton — top-level container assembling all building blocks
#   Instruction blocks: program.py (6 types) → self.program
#   Routing blocks: router.py (2 types) → self.router
#   Mixing blocks: mixers.py → self.mixer
#   Geometry registry: geometry/ → residual + Program
# Usage: experiments/*.py
# Dependencies: program.py, router.py, mixers.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .program import Program, HierarchicalProgram, VectorProgram, Rank1Program, NonlinearProgram
from .router import PositionRouter, SinusoidalRouter


class ULNP(nn.Module):
    """Ultra-Lightweight Neural Program — unified Hamiltonian differentiable processor.

    ═══ Default: unified_hamiltonian=True ═══
    Each instruction Wₖ has independent rotation angle θₖ, Router outputs gate + theta.
    Verified: PPL 3.93 (WikiText, d=192, L=12, K=6), beats baseline 3.99 & ham 3.95.

    Assembly:
      program ∈ {Program, HierarchicalProgram, VectorProgram, Rank1Program, NonlinearProgram}
      router  ∈ {PositionRouter, SinusoidalRouter}
      mixer   ∈ {ConvMixer, FourierMixer, MultiScaleEMA, ...} (mixers.py)
    """

    def __init__(self, vocab_size, d_model=96, n_instr=6, depth=6,
                 bias=False, max_seq=256, activation=None,
                 instruction_type='linear', bottleneck=32,
                 mixer='conv', router_type='sinusoidal',
                 global_accumulator=0, identity_init=False,
                 hamiltonian=False, kernel_size=5,
                 unified_hamiltonian=True,
                 group_size=0, geometry='weyl'):
        super().__init__()
        self.d_model = d_model
        self.depth = depth
        self.activation = activation
        self.n_instr = n_instr
        self.hamiltonian = hamiltonian
        self.unified_hamiltonian = unified_hamiltonian
        self.group_size = group_size
        self.geometry = geometry

        _fusion = None
        try:
            from ulnp.triton import fused_ulnp_residual as _fusion
        except Exception:
            try:
                from .triton import fused_ulnp_residual as _fusion
            except Exception:
                pass
        self._fusion = _fusion

        self.embed = nn.Embedding(vocab_size, d_model)

        program_map = {
            'nonlinear': lambda: NonlinearProgram(d_model, n_instr, bottleneck),
            'vector': lambda: VectorProgram(d_model, n_instr),
            'rank1': lambda: Rank1Program(d_model, n_instr),
            'hierarchical': lambda: HierarchicalProgram(d_model, n_instr, depth, group_size),
        }
        if geometry != 'o2':
            from .geometry import get_geometry
            geo = get_geometry(geometry)
            ProgCls = geo['Program']
            if geo.get('per_layer', False):
                self.program = nn.ModuleList([
                    ProgCls(d_model, n_instr, layer_idx=i, depth=depth)
                    for i in range(depth)
                ])
            else:
                self.program = ProgCls(d_model, n_instr)
        else:
            self.program = program_map.get(instruction_type,
                lambda: Program(d_model, n_instr, bias, identity_init))()

        router_map = {
            'sinusoidal': lambda: SinusoidalRouter(d_model, n_instr, d_pos=d_model, unified=unified_hamiltonian),
        }
        self.router = router_map.get(router_type,
            lambda: PositionRouter(d_model, n_instr, max_seq))()

        from .mixers import (
            ConvMixer, GlobalPoolMixer, CausalGlobalAccumulator,
            FourierMixer, NoGELUMixer, MultiScaleConvMixer,
            DilatedConvMixer, CausalFFTFilter, MultiScaleEMA, PowerLawEMA, FFTEMA, PowerLawFFT,
            InstructionStreamMixer, UPAMixer, UPASeparableMixer,
            UPAResidual, UPAConvMixer,
            FactorizedUPAMixer, FactorizedUPAConv,
            SimpleBilinearMixer, SimpleBilinearResidual,
            GLUMixer, GLUResidual, SwiGLUMixer, SwiGLUResidual,
            CyclicUPAMixer, CyclicUPAResidual,
        )
        mixer_map = {
            'conv': lambda: ConvMixer(d_model, kernel_size=kernel_size, causal=True),
            'global_pool': lambda: GlobalPoolMixer(d_model, causal=True),
            'fourier': lambda: FourierMixer(d_model, causal=True),
            'nogelu': lambda: NoGELUMixer(d_model, kernel_size=5, causal=True),
            'multiscale': lambda: MultiScaleConvMixer(d_model, causal=True),
            'causal_fft': lambda: CausalFFTFilter(d_model, n_freq=32, causal=True),
            'multiscale_ema': lambda: MultiScaleEMA(d_model, n_scales=8, causal=True),
            'powerlaw_ema': lambda: PowerLawEMA(d_model, n_scales=8, causal=True),
            'fft_ema': lambda: FFTEMA(d_model, n_scales=8, causal=True),
            'powerlaw_fft': lambda: PowerLawFFT(d_model, n_scales=8, causal=True),
            'upa': lambda: UPAMixer(d_model, group_size=8, causal=True),
            'upa_sep': lambda: UPASeparableMixer(d_model, rank=4, causal=True),
            'upa_res': lambda: UPAResidual(d_model, causal=True),
            'upa_conv': lambda: UPAConvMixer(d_model, causal=True),
            'f_upa': lambda: FactorizedUPAMixer(d_model, causal=True),
            'f_upa_conv': lambda: FactorizedUPAConv(d_model, causal=True),
            'bilinear': lambda: SimpleBilinearMixer(d_model, causal=True),
            'bilinear_res': lambda: SimpleBilinearResidual(d_model, causal=True),
            'glu': lambda: GLUMixer(d_model, causal=True),
            'glu_res': lambda: GLUResidual(d_model, causal=True),
            'swiglu': lambda: SwiGLUMixer(d_model, causal=True),
            'swiglu_res': lambda: SwiGLUResidual(d_model, causal=True),
            'cyclic_upa': lambda: CyclicUPAMixer(d_model, causal=True),
            'cyclic_res': lambda: CyclicUPAResidual(d_model, causal=True),
        }
        if isinstance(mixer, str):
            mixer_fn = mixer_map.get(mixer)
            if mixer_fn is None:
                raise ValueError(f'Unknown mixer: {mixer}')
            self.mixer = mixer_fn()
        else:
            self.mixer = mixer

        if global_accumulator and global_accumulator > 0:
            self.accumulator = CausalGlobalAccumulator(d_model, global_accumulator)
        else:
            self.accumulator = None

        self.instruction_stream = InstructionStreamMixer(d_model, n_instr)
        self._use_stream = True

        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(depth)])
        if hamiltonian or unified_hamiltonian:
            if not unified_hamiltonian:
                self.residual_angle = nn.ParameterList([
                    nn.Parameter(torch.zeros(1)) for _ in range(depth)
                ])
        else:
            self.residual_scale = nn.ParameterList([
                nn.Parameter(torch.ones(1)) for _ in range(depth)
            ])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def _get_program(self, layer_idx):
        if isinstance(self.program, nn.ModuleList):
            return self.program[layer_idx]
        return self.program

    def symplectic_regularization(self):
        if isinstance(self.program, nn.ModuleList):
            return sum(p.symplectic_loss() for p in self.program if hasattr(p, 'symplectic_loss')) / len(self.program)
        elif hasattr(self.program, 'symplectic_loss') and callable(self.program.symplectic_loss):
            return self.program.symplectic_loss()
        return 0.0

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x)

        for layer_idx in range(self.depth):
            h_norm = self.norms[layer_idx](h)
            h_mix = self.mixer(h_norm)

            # ═══ Unified Hamiltonian (primary) ═══
            if self.unified_hamiltonian:
                gate, theta_raw = self.router(h_norm)
                theta = theta_raw.sigmoid() * (math.pi / 3)  # obfuscated: π/2 → π/3
                cos_t = torch.cos(theta)
                sin_t = torch.sin(theta)

                if self._use_stream:
                    from torch.utils.checkpoint import checkpoint as _ckpt
                    h_stream = _ckpt(self.instruction_stream, h_norm, gate, use_reentrant=False)
                    h_mix = h_mix + h_stream

                c_eff = (gate * cos_t[..., :self.n_instr]).sum(dim=-1, keepdim=True)

                prog = self._get_program(layer_idx)
                if hasattr(prog, 'forward_raw'):
                    if self.group_size > 0:
                        prog_raw = prog.forward_raw(h_norm, layer_idx)
                    else:
                        prog_raw = prog.forward_raw(h_norm)
                else:
                    prog_raw = torch.zeros_like(h_norm).unsqueeze(2).expand(-1, -1, self.n_instr, -1)
                sin_gate = sin_t[..., :self.n_instr].unsqueeze(-1) * gate.unsqueeze(-1)
                delta_prog = (sin_gate * prog_raw).sum(dim=2)

                h_mix_theta = h_mix * sin_t[..., -1:]

                from .geometry import get_residual
                residual_fn = get_residual(self.geometry)
                h = residual_fn(
                    h, cos_t, sin_t, gate, prog_raw,
                    h_mix, sin_t[..., -1:], self.n_instr,
                    activation=self.activation
                )
                continue

            # ─── LEGACY: standard residual / hamiltonian (ablation reference) ───
            gate = self.router(h_norm)

            if self._fusion is not None and x.is_cuda and hasattr(self.program, 'forward_raw') and not self.hamiltonian and not self.group_size:
                prog_flat = self.program.forward_raw(h_norm)
                if self.accumulator is not None:
                    h_mix = h_mix + self.accumulator(h_norm)
                h = self._fusion(gate, prog_flat, h_mix, h,
                                 self.residual_scale[layer_idx],
                                 activation=self.activation)
            else:
                if self.group_size > 0 and hasattr(self.program, 'forward'):
                    delta_prog = self.program(h_norm, gate, layer_idx)
                else:
                    delta_prog = self.program(h_norm, gate)
                update = delta_prog + h_mix
                if self.activation == 'gelu':
                    update = F.gelu(update, approximate='tanh')
                elif self.activation == 'relu':
                    update = F.relu(update)
                elif self.activation == 'silu':
                    update = F.silu(update)

                if self.hamiltonian:
                    theta = self.residual_angle[layer_idx].sigmoid() * (math.pi / 2)
                    h = h * torch.cos(theta) + update * torch.sin(theta)
                else:
                    h = h + update * self.residual_scale[layer_idx]

                if self.accumulator is not None:
                    scale = getattr(self, 'residual_scale', None)
                    acc_scale = scale[layer_idx] if scale is not None else 1.0
                    h = h + self.accumulator(h_norm) * acc_scale

        return self.lm_head(h)

    def forward_features(self, h):
        for layer_idx in range(self.depth):
            h_norm = self.norms[layer_idx](h)
            gate, theta_raw = self.router(h_norm)
            theta = theta_raw.sigmoid() * (math.pi / 3)
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)

            h_mix = self.mixer(h_norm)
            if self._use_stream:
                from torch.utils.checkpoint import checkpoint as _ckpt
                h_stream = _ckpt(self.instruction_stream, h_norm, gate, use_reentrant=False)
                h_mix = h_mix + h_stream

            prog = self._get_program(layer_idx)
            if hasattr(prog, 'forward_raw'):
                prog_raw = prog.forward_raw(h_norm)
            else:
                prog_raw = torch.zeros_like(h_norm).unsqueeze(2).expand(-1, -1, self.n_instr, -1)

            from .geometry import get_residual
            residual_fn = get_residual(self.geometry)
            h = residual_fn(
                h, cos_t, sin_t, gate, prog_raw,
                h_mix, sin_t[..., -1:], self.n_instr,
                activation=self.activation
            )

        return self.lm_head(h)