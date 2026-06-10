# 作用: CO(n) 保角残差 — 旋转 × 标量，保方向不保幅度
# 调用: core.py → geometry/__init__.py → get_residual('co')
# 依赖: torch
# 不变量: 方向 (角度), 不保 ||h|| (幅度可变)

import math
import torch
import torch.nn.functional as F


def co_residual(h, cos_t, sin_t, gate, prog_raw, h_mix, sin_t_conv, n_instr, activation='gelu'):
    """CO(n) 保角残差: O(2) 旋转 × 可学习标量缩放.

    CO(n) = O(2) × R⁺:
      - 旋转: 保持方向 (h/||h|| 在 (h, update) 平面旋转)
      - 缩放: λ = 1 + α·tanh(·), 可放大或缩小幅度

    与 O(2) 的区别:
      O(2):      ∥h∥² 严格不变 — 刚性
      CO(n):     方向不变, 幅度 λ ∈ (0, 2) — 柔性

    适用场景: token 置信度估计 — 某些 token 需要"更确定"(幅度大)
    或"更不确定"(幅度小), 但不丢掉语义方向.
    """
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

    # O(2) 旋转
    h_rotated = h * cos_sum + update

    # 保角缩放: λ = sigmoid(sin_t_conv) + 0.5 ∈ (0.5, 1.5)
    lambda_scale = torch.sigmoid(sin_t_conv) + 0.5

    return h_rotated * lambda_scale
