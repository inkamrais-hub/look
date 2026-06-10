"""
SL(n) Geometry — Special Linear Group

体积保持的线性变换，det = 1
特征：保体积但可以拉伸/剪切，比 O(2) 更灵活但比一般线性更有约束
"""

import torch
import torch.nn as nn
import math


class SLnProgram(nn.Module):
    """
    SL(n) 指令集：det = 1 的线性变换
    
    实现方式：用 Lie 代数 sl(n)（迹为零的矩阵）指数映射
    sl(n) = {X : tr(X) = 0}
    SL(n) = {exp(X) : X ∈ sl(n)}
    
    exp(X) 自动满足 det = 1（因为 tr(X) = 0 → det(exp(X)) = exp(tr(X)) = 1）
    """
    
    def __init__(self, d_model: int, n_instr: int = 6):
        super().__init__()
        self.d_model = d_model
        self.n_instr = n_instr
        
        # Lie 代数参数（迹为零的矩阵）
        # 用 d×d 矩阵，然后减去对角线均值确保 tr=0
        self.weight = nn.Parameter(torch.randn(n_instr, d_model, d_model) * 0.02)
    
    def forward_raw(self, x):
        """
        x: [B, T, d_model]
        返回: [B, T, n_instr, d_model]
        """
        # 强制 tr(W) = 0 → sl(n)
        w = self.weight - self.weight.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True) \
            * torch.eye(self.d_model, device=x.device).unsqueeze(0)
        
        # 矩阵指数 → SL(n) 元素
        # 小矩阵可以精确计算，大矩阵用 Padé 近似
        if self.d_model <= 64:
            w_exp = torch.matrix_exp(w)  # [K, d, d]
        else:
            # 一阶近似：exp(X) ≈ I + X（当 ||X|| 小时）
            w_exp = torch.eye(self.d_model, device=x.device).unsqueeze(0) + w
        
        return torch.einsum('kij,btj->btki', w_exp, x)
    
    def det_loss(self):
        """正则化：det(exp(X)) = exp(tr(X)) → 确保 tr(X) ≈ 0"""
        tr = self.weight.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        return tr.pow(2).mean()


def sln_residual(
    h: torch.Tensor,
    cos_t: torch.Tensor,
    sin_t: torch.Tensor,
    gate: torch.Tensor,
    prog_raw: torch.Tensor,
    h_mix: torch.Tensor,
    sin_t_conv: torch.Tensor,
    n_instr: int,
    activation: str = 'gelu',
) -> torch.Tensor:
    """
    SL(n) 残差：体积保持 + 剪切/拉伸
    
    与 O(2) 类似，但 prog 代表的是 SL(n) 变换而非旋转
    h' = h + Σ gate_k · prog_k(h)
    
    不用 cos/sin 旋转，直接用门控加权的 SL(n) 变换
    """
    B, T, D = h.shape
    K = n_instr
    
    # 门控加权的 SL(n) 变换
    # prog_raw: [B, T, K, D]
    # gate: [B, T, K]
    delta = (gate.unsqueeze(-1) * prog_raw).sum(dim=2)  # [B, T, D]
    
    # 残差更新（加性，保持体积的直觉：det=1 的变换不改变"体积"）
    return h + delta + h_mix * sin_t_conv


class SLnResidual(nn.Module):
    """SL(n) 残差层（可选的独立实现）"""
    
    def __init__(self, d_model, n_instr=6):
        super().__init__()
        self.program = SLnProgram(d_model, n_instr)
        self.gate_proj = nn.Linear(d_model, n_instr)
        self.mix_proj = nn.Linear(d_model, d_model)
    
    def forward(self, h, layer_idx=0):
        prog = self.program.forward_raw(h)
        gate = torch.sigmoid(self.gate_proj(h))
        h_mix = self.mix_proj(h)
        
        delta = (gate.unsqueeze(-1) * prog).sum(dim=2)
        return h + delta + h_mix
