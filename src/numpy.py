"""τ normalization reference — pure NumPy, zero dependencies.

CANONICAL formula (matching CUDA kernel):
    a_i = softplus(s_i)^τ / Σ softplus(s_j)^τ     [SP]

Variants:
  [SP]    stau_norm(s, τ)          — CANONICAL (matches CUDA v15a kernel)
  [SP-M]  stau_norm_maxstab(s, τ)  — max-stabilized (shifts scores BEFORE softplus)
  [SM]    softmax_norm(s, T)       — baseline softmax
  [τ*-SP] tau_star_closed_sp(s)    — τ* = Cov(s, log(sp(s))) / Var(log(sp(s)))

IMPORTANT: [SP] and [SP-M] are NOT equivalent.
  [SP]:  a_i ∝ softplus(s_i)^τ          — everything passes through softplus first
  [SP-M]: a_i ∝ softplus(s_i - max(s))^τ — scores shifted THEN softplus → spikier

The CUDA kernel (fused_attention.py) implements [SP], not [SP-M].
Use [SP] as the CANONICAL math reference.

This file is used as a unit-test oracle for validating CUDA implementations.
No other modules depend on it at runtime.
"""

# ============================================================
# 文件角色: s^τ 归一化的纯 NumPy 参考实现 (数学真相源)
# 调用顺序: 独立 → 被 test_tau_star_sp.py 调用验证
# 数据流向: scores [N] → _softplus → pow(τ) → normalize → probs [N]
# 实现: [SP] CANONICAL + [SP-M] variant + [SM] baseline
# ============================================================

import numpy as np

EPS = 1e-8


def _softplus(x):
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


# ═══════════════════════════════════════════════════════════════
# [SP]  CANONICAL — a_i = softplus(s_i)^τ / Σ
# ═══════════════════════════════════════════════════════════════

def stau_norm(scores, tau, eps=EPS):
    """[SP] CANONICAL: a_i = softplus(s_i)^τ / Σ_j softplus(s_j)^τ

    Matches CUDA v15a kernel (fused_attention.py L132-133):
        sp = softplus_stable(dot)
        val = tau_h * logf(sp + eps)   → log-space softmax ≡ softplus(s)^τ / Σ

    Implementation note:
        Σ is computed as sum(powered) then normalized.
        For numerical stability, uses max-stabilized log-space internally.

    Args:
        scores: (N,) pre-softmax attention scores
        tau:    scalar ∈ (1, ∞),  1.0=wide  20.0=telephoto
        eps:    floor for numerical stability

    Returns:
        probs:  (N,) Σ_i probs_i = 1.0
    """
    phi = _softplus(scores) + eps
    log_phi = np.log(phi)
    log_scaled = tau * log_phi
    max_val = log_scaled.max()
    exps = np.exp(log_scaled - max_val)
    total = exps.sum()
    if total <= 0:
        return np.ones_like(exps) / len(exps)
    return exps / total


stau_norm_softplus = stau_norm  # alias


# ═══════════════════════════════════════════════════════════════
# [SP-M]  max-stabilized — shifts scores BEFORE softplus (spikier)
# ═══════════════════════════════════════════════════════════════

def stau_norm_maxstab(scores, tau, eps=EPS):
    """[SP-M] a_i = softplus(s_i - max(s))^τ / Σ

    DIFFERENT from [SP]: subtracting max BEFORE softplus compresses
    non-maximal scores much more aggressively.

    Compare:
        s = [-1, 0, 5]
        [SP]:   softplus(-1)^τ, softplus(0)^τ, softplus(5)^τ
        [SP-M]: softplus(-6)^τ, softplus(-5)^τ, softplus(0)^τ   (shifted by +5)

    [SP-M] is NOT what the CUDA kernel computes.
    The CUDA kernel uses [SP] + log-space max for numerical stability.
    """
    s_stable = scores - scores.max()
    phi = _softplus(s_stable) + eps
    powered = np.power(phi, tau)
    total = powered.sum()
    if total <= 0:
        return np.ones_like(powered) / len(powered)
    return powered / total


stau_norm_sp_maxstab = stau_norm_maxstab  # backward-compat alias


# ═══════════════════════════════════════════════════════════════
# [SM]  baseline softmax
# ═══════════════════════════════════════════════════════════════

def softmax_norm(scores, temperature=1.0):
    """[SM] a_i = exp((s_i - max(s)) / T) / Σ"""
    shifted = scores - scores.max()
    exps = np.exp(shifted / max(temperature, 0.01))
    total = exps.sum()
    if total <= 0:
        return np.ones_like(exps) / len(exps)
    return exps / total


# ═══════════════════════════════════════════════════════════════
# 分析工具
# ═══════════════════════════════════════════════════════════════

def entropy(probs):
    mask = probs > 0
    return -np.sum(probs[mask] * np.log2(probs[mask]))


def effective_n(probs):
    return 1.0 / np.maximum(np.sum(probs ** 2), EPS)


# ═══════════════════════════════════════════════════════════════
# [τ*-SP]  closed-form τ* = Cov(s, log(sp(s))) / Var(log(sp(s)))
# ═══════════════════════════════════════════════════════════════

def tau_star_closed_sp(scores, eps=EPS):
    """[τ*-SP] τ* = Cov(s, log(sp(s))) / Var(log(sp(s)))

    Closed-form from s^τ ↔ softmax equivalence theorem.
    Assumes s ≈ τ·log(sp(s)) + C, OLS slope = covariance ratio.
    1D or 2D input (2D auto-flattened).

    Numerical implementation:
        Uses Pearson covariance and Bessel-corrected variance.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    phi = _softplus(s) + eps
    log_phi = np.log(phi)
    var = np.var(log_phi, ddof=1)
    if var < eps:
        return 1.0
    s_mean = s.mean()
    lp_mean = log_phi.mean()
    cov = np.mean((s - s_mean) * (log_phi - lp_mean))
    return cov / var


# legacy, for reference only
def _tau_star_closed_clamp(scores, eps=EPS):
    """[τ*-CLAMP] LEGACY — τ* on clamp(s,ε) stats. Do not use for softplus."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    s_c = np.clip(s, eps, None)
    log_s = np.log(s_c)
    var = np.var(log_s, ddof=1)
    if var < eps:
        return 1.0
    s_mean = s.mean()
    ls_mean = log_s.mean()
    cov = np.mean((s - s_mean) * (log_s - ls_mean))
    return cov / var


# ═══════════════════════════════════════════════════════════════
#  demonstration
# ═══════════════════════════════════════════════════════════════

def demonstrate():
    scores = np.array([100.0, 10.0, 1.0])
    print("=" * 60)
    print("  s^τ FOCUS-KNOB DEMONSTRATION  [SP] CANONICAL")
    print("=" * 60)
    print(f"  Scores: {scores}")
    print()
    header = f"  {'tau':>5}  {'probs':>30}  {'entropy(bits)':>15}  {'eff_n':>7}"
    sep = f"  {'-'*5}  {'-'*30}  {'-'*15}  {'-'*7}"
    print(header)
    print(sep)

    for tau in [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]:
        p = stau_norm(scores, tau)
        ent = entropy(p)
        eff = effective_n(p)
        pstr = "  ".join(f"{x:.4f}" for x in p)
        print(f"  {tau:5.1f}  [{pstr}]  {ent:15.3f}  {eff:7.2f}")

    print()
    sp = _softplus(scores)
    t_sp = tau_star_closed_sp(scores)
    print(f"  [τ*-SP]   tau* = {t_sp:.4f}  (softplus)")
    print(f"  softplus(s) = {sp}")
    print()
    print("  [SP]    CANONICAL — matches CUDA kernel, softplus(s)^τ")
    print("  [SP-M]  max-stab  — shifts BEFORE softplus (spikier)")


if __name__ == '__main__':
    demonstrate()