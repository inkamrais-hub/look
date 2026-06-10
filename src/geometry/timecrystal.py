"""
Time Crystal Geometry — 多频耦合振荡残差

适用于周期性信号：EEG, sEEG, 心电, 音频等
核心：多频相位锁定 + 自发对称破缺门控
"""

import torch
import torch.nn as nn
import math


class TimeCrystalProgram(nn.Module):
    """
    时晶指令集：多频相位演化
    
    每条指令对应一个频率，通过层索引驱动相位演化
    不学习频率本身（固定），学习振幅和耦合强度
    """
    
    def __init__(self, d_model: int, n_instr: int = 6, base_freq: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.n_instr = n_instr
        
        # 频率：固定为 base_freq 的倍数序列
        # 对应脑波的 delta, theta, alpha, beta, gamma, high-gamma
        freq_ratios = torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])[:n_instr]
        self.register_buffer('omega', freq_ratios * base_freq)
        
        # 学习参数：每条指令的振幅和相位偏移
        self.amplitude = nn.Parameter(torch.ones(n_instr) * 0.5)
        self.phase_offset = nn.Parameter(torch.zeros(n_instr))
        
        # 指令变换矩阵
        self.weight = nn.Parameter(torch.randn(n_instr, d_model, d_model) * 0.02)
    
    def forward_raw(self, x, layer_idx: int = 0):
        """
        x: [B, T, d_model]
        layer_idx: 当前层索引，驱动相位演化
        
        返回: [B, T, n_instr, d_model] — 每条指令的输出
        """
        # 相位演化：ω * l + φ
        phase = self.omega * layer_idx + self.phase_offset  # [n_instr]
        
        # 时晶门控：振幅 * (1 + cos(phase)) / 2
        # 范围 [0, amplitude]，自发对称破缺
        gate = self.amplitude * (1 + torch.cos(phase)) / 2  # [n_instr]
        
        # 指令变换
        prog = torch.einsum('kij,btj->btki', self.weight, x)  # [B, T, K, d]
        
        # 门控调制
        prog = prog * gate[None, None, :, None]  # [B, T, K, d]
        
        return prog


def timecrystal_residual(
    h: torch.Tensor,
    cos_t: torch.Tensor,
    sin_t: torch.Tensor,
    gate: torch.Tensor,
    prog_raw: torch.Tensor,
    h_mix: torch.Tensor,
    sin_t_conv: torch.Tensor,
    n_instr: int,
    activation: str = 'gelu',
    layer_idx: int = 0,
) -> torch.Tensor:
    """
    时晶残差：多频耦合 + 相位锁定
    
    与 O(2) 残差类似，但加入相位调制项：
    h' = h·cos_sum + Σ(gate_k · sin_k · prog_k) + phase_coupling
    
    phase_coupling = Σ_i Σ_j≠i A_ij · cos(ω_i·l - ω_j·l + φ_ij)
    这一项捕捉跨频耦合（如 Theta-Gamma Coupling）
    """
    B, T, D = h.shape
    K = n_instr
    
    # 标准 O(2) 残差部分
    cos_sum = (gate * cos_t[..., :K]).sum(dim=-1, keepdim=True)
    sin_gate = sin_t[..., :K].unsqueeze(-1) * gate.unsqueeze(-1)
    delta_prog = (sin_gate * prog_raw).sum(dim=2)
    
    # 时晶相位耦合项
    # 用 sin_t_conv 的多频版本
    # 简化：直接用 cos_sum 的振荡特性
    # 更复杂的版本可以加跨频耦合矩阵
    
    return h * cos_sum + delta_prog + h_mix * sin_t_conv


class TimeCrystalCoupling(nn.Module):
    """
    跨频耦合矩阵：捕捉不同频率之间的相位锁定关系
    
    例如：Theta-Gamma Coupling = γ振幅被θ相位调制
    """
    
    def __init__(self, n_freq: int = 6):
        super().__init__()
        # 耦合强度矩阵（非对称：θ→γ ≠ γ→θ）
        self.coupling = nn.Parameter(torch.randn(n_freq, n_freq) * 0.01)
        # 对角线设为0（不自耦合）
        self.register_buffer('mask', 1 - torch.eye(n_freq))
    
    def forward(self, phases: torch.Tensor):
        """
        phases: [n_freq] — 各频率的当前相位
        
        返回: [n_freq] — 耦合修正后的相位
        """
        # 相位差矩阵
        phase_diff = phases.unsqueeze(0) - phases.unsqueeze(1)  # [n_freq, n_freq]
        
        # 耦合项：A_ij * sin(φ_i - φ_j)
        coupling_matrix = self.coupling * self.mask
        coupling_force = (coupling_matrix * torch.sin(phase_diff)).sum(dim=-1)
        
        return phases + coupling_force
