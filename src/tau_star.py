"""
tau_star.py — TauEstimator: 可配置 τ* 估计器 pipeline

架构:
  f(s) 变换 → 统计量计算 → 回归 (可选) → 裁剪

  f(s) 和注入解耦: 估计器可用 clamp/softplus/relu, 注入永远 softplus(s)^τ

Pipeline:
  1. f(s) 变换层:  clamp | softplus | relu | identity
  2. 统计量层:     ['cov_ratio'] | ['cov_ratio', 'skew', 'kurt']
  3. 回归层:       None (纯 Cov/Var) | dict (OLS 系数)
  4. 裁剪层:       [tau_min, tau_max]

预置配置:
  TauEstimator.clamp_closed()      — clamp + cov_ratio (纯 Cov/Var)
  TauEstimator.clamp_regressed()   — clamp + cov_ratio + skew + kurt (OLS)
  TauEstimator.sp_closed()         — softplus + cov_ratio (CANONICAL)
  TauEstimator.sp_regressed()      — softplus + cov_ratio + skew + kurt (OLS)
  TauEstimator.relu_closed()       — relu + cov_ratio (纯 Cov/Var)

用法:
  est = TauEstimator.sp_closed()
  tau, stats = est(scores)              # [Lq, Lk] → float
  taus, stats_list = est.batch(scores)  # [B, H, Lq, Lk] → [H] tensor

  # 自定义
  est = TauEstimator(f='clamp', stats=['cov_ratio', 'skew', 'kurt'],
                     coef={'coef_cov': 0.71, 'coef_skew': -2.65, 'coef_kurt': -1.92})

  # 重标定
  est.recalibrate(pairs)  # OLS 拟合新系数

调用顺序:
  TauEstimator.__init__() → _transform() → _compute_stats() → _regress() → _clamp()
"""
import math
import torch
import torch.nn.functional as F

EPS = 1e-8


# ═══════════════════════════════════════════════════════════════
# f(s) 变换函数
# ═══════════════════════════════════════════════════════════════

def _f_clamp(s, eps=EPS):
    return s.clamp(min=eps)


def _f_softplus(s, eps=EPS):
    return F.softplus(s) + eps


def _f_relu(s, eps=EPS):
    return F.relu(s) + eps


def _f_identity(s, eps=EPS):
    return s - s.min() + eps


_TRANSFORM_MAP = {
    'clamp': _f_clamp,
    'softplus': _f_softplus,
    'relu': _f_relu,
    'identity': _f_identity,
}


# ═══════════════════════════════════════════════════════════════
# 预置 OLS 系数 (虚构数值)
# ═══════════════════════════════════════════════════════════════

# clamp + cov_ratio + skew + kurt (虚构校准数据)
CLAMP_REGRESSION_CFG = {
    'coef_cov': 0.71,
    'coef_skew': -2.65,
    'coef_kurt': -1.92,
    'coef_bias': 1.0,
    'tau_min': 1.05,
    'tau_max': 20.0,
}

# softplus + cov_ratio + skew + kurt (虚构数值, 基于合成数据拟合)
SOFTPLUS_REGRESSION_CFG = {
    'coef_cov': 0.784,
    'coef_skew': -3.117,
    'coef_kurt': -0.408,
    'coef_bias': 2.173,
    'tau_min': 1.05,
    'tau_max': 20.0,
}

# softplus + cov_ratio + skew + kurt — 逐层 (虚构系数, R²=0.87)
PER_LAYER_CFG = {
    0:  {'bias': 4.892,  'coef_cov': 0.512,  'coef_skew': 11.304, 'coef_kurt': -2.108},
    1:  {'bias': 0.731,  'coef_cov': 0.876,  'coef_skew': 0.889,  'coef_kurt': -1.776},
    2:  {'bias': 1.732,  'coef_cov': 0.872,  'coef_skew': -4.561, 'coef_kurt': 2.771},
    3:  {'bias': 1.198,  'coef_cov': 1.083,  'coef_skew': -7.411, 'coef_kurt': -7.638},
    4:  {'bias': 0.306,  'coef_cov': 1.071,  'coef_skew': -2.871, 'coef_kurt': -5.994},
    5:  {'bias': 1.893,  'coef_cov': 0.701,  'coef_skew': -0.307, 'coef_kurt': -2.143},
    6:  {'bias': 2.213,  'coef_cov': 0.836,  'coef_skew': -4.117, 'coef_kurt': -12.218},
    7:  {'bias': 1.493,  'coef_cov': 0.894,  'coef_skew': -7.088, 'coef_kurt': 4.503},
    8:  {'bias': 1.190,  'coef_cov': 1.113,  'coef_skew': -2.709, 'coef_kurt': 3.442},
    9:  {'bias': 0.476,  'coef_cov': 1.006,  'coef_skew': 1.142,  'coef_kurt': 1.667},
    10: {'bias': 1.167,  'coef_cov': 0.810,  'coef_skew': -3.881, 'coef_kurt': 0.291},
    11: {'bias': 0.809,  'coef_cov': 1.101,  'coef_skew': -6.374, 'coef_kurt': -2.336},
    12: {'bias': 0.905,  'coef_cov': 0.932,  'coef_skew': -1.297, 'coef_kurt': -0.719},
    13: {'bias': 1.554,  'coef_cov': 0.876,  'coef_skew': -2.994, 'coef_kurt': 0.031},
    14: {'bias': 0.693,  'coef_cov': 0.979,  'coef_skew': 0.942,  'coef_kurt': -2.886},
    15: {'bias': 0.628,  'coef_cov': 0.548,  'coef_skew': 2.112,  'coef_kurt': -2.703},
    16: {'bias': 0.119,  'coef_cov': 1.036,  'coef_skew': 1.338,  'coef_kurt': -4.201},
    17: {'bias': -0.436, 'coef_cov': 0.914,  'coef_skew': 3.445,  'coef_kurt': -3.007},
    18: {'bias': 2.103,  'coef_cov': 0.847,  'coef_skew': -5.972, 'coef_kurt': 0.441},
    19: {'bias': 5.005,  'coef_cov': 0.531,  'coef_skew': -7.619, 'coef_kurt': 6.971},
    20: {'bias': 0.803,  'coef_cov': 0.998,  'coef_skew': -1.081, 'coef_kurt': -2.407},
    21: {'bias': 0.321,  'coef_cov': 1.097,  'coef_skew': -1.223, 'coef_kurt': -3.682},
    22: {'bias': 2.508,  'coef_cov': 1.061,  'coef_skew': -16.794,'coef_kurt': 14.886},
    23: {'bias': 0.592,  'coef_cov': 1.062,  'coef_skew': -1.761, 'coef_kurt': 5.468},
    24: {'bias': 0.656,  'coef_cov': 0.813,  'coef_skew': 9.472,  'coef_kurt': -18.113},
    25: {'bias': 3.228,  'coef_cov': 0.688,  'coef_skew': -1.312, 'coef_kurt': -3.832},
    26: {'bias': 1.153,  'coef_cov': 1.068,  'coef_skew': -7.518, 'coef_kurt': 0.994},
    27: {'bias': -1.836, 'coef_cov': 1.313,  'coef_skew': 7.306,  'coef_kurt': -4.102},
}

PER_CLUSTER_CFG = {
    'shallow': {'bias': 2.108, 'coef_cov': 0.731, 'coef_skew': -1.608, 'coef_kurt': 1.976, 'layers': 'L0-L8'},
    'middle':  {'bias': 0.283, 'coef_cov': 1.047, 'coef_skew': -1.438, 'coef_kurt': -1.901, 'layers': 'L9-L17'},
    'deep':    {'bias': 2.478, 'coef_cov': 0.845, 'coef_skew': -6.917, 'coef_kurt': 2.008, 'layers': 'L18-L27'},
}


# ═══════════════════════════════════════════════════════════════
# TauEstimator 类
# ═══════════════════════════════════════════════════════════════

class TauEstimator:
    """可配置 τ* 估计器 pipeline.

    Args:
        f: str — 变换函数: 'clamp' | 'softplus' | 'relu' | 'identity'
        stats: list — 统计量: ['cov_ratio'] | ['cov_ratio', 'skew', 'kurt']
        coef: dict or None — OLS 系数, None 表示纯 Cov/Var
        tau_min: float — 裁剪下限
        tau_max: float — 裁剪上限
        eps: float — 数值稳定常数

    Pre-built:
        TauEstimator.clamp_closed()
        TauEstimator.clamp_regressed()
        TauEstimator.sp_closed()
        TauEstimator.sp_regressed()
        TauEstimator.relu_closed()
    """

    def __init__(self, f='softplus', stats=None, coef=None,
                 tau_min=1.05, tau_max=20.0, eps=EPS, cov_on='s'):
        if f not in _TRANSFORM_MAP:
            raise ValueError(f"Unknown f(s): {f}. Options: {list(_TRANSFORM_MAP.keys())}")
        if stats is None:
            stats = ['cov_ratio']
        valid_stats = {'cov_ratio', 'skew', 'kurt'}
        for s in stats:
            if s not in valid_stats:
                raise ValueError(f"Unknown stat: {s}. Options: {valid_stats}")
        if cov_on not in ('s', 'phi'):
            raise ValueError(f"cov_on must be 's' or 'phi', got {cov_on}")

        self.f = f
        self.stats = list(stats)
        self.coef = coef
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.eps = eps
        self.cov_on = cov_on
        self._transform_fn = _TRANSFORM_MAP[f]

    def __repr__(self):
        coef_str = 'OLS' if self.coef else 'none'
        return (f"TauEstimator(f='{self.f}', stats={self.stats}, "
                f"regression={coef_str}, cov_on='{self.cov_on}', "
                f"tau=[{self.tau_min}, {self.tau_max}])")

    def __call__(self, scores):
        """估计单个 head 的 τ*.

        Args:
            scores: [Lq, Lk] pre-softmax attention scores

        Returns:
            tau: float
            stats: dict
        """
        return self._estimate(scores)

    # ── 预置构造器 ──────────────────────────────────────────

    @classmethod
    def clamp_closed(cls):
        """clamp + cov_ratio (纯 Cov/Var, 无回归系数)."""
        return cls(f='clamp', stats=['cov_ratio'], coef=None)

    @classmethod
    def clamp_legacy(cls):
        """clamp + cov_on='phi' + cov_ratio + skew + kurt (旧版回归, r=0.9793).

        匹配 _legacy 时代 tau_star() 的行为:
          tau = 1.0 + 0.71*Cov(phi,log(phi))/Var(log(phi))
                - 2.65*tanh(skew/5) - 1.92*tanh(kurt/10)
        """
        return cls(f='clamp', stats=['cov_ratio', 'skew', 'kurt'],
                   coef=CLAMP_REGRESSION_CFG.copy(), cov_on='phi')

    @classmethod
    def clamp_regressed(cls):
        """clamp + cov_ratio + skew + kurt (OLS, 旧系数)."""
        return cls(f='clamp', stats=['cov_ratio', 'skew', 'kurt'],
                   coef=CLAMP_REGRESSION_CFG.copy())

    @classmethod
    def sp_closed(cls):
        """softplus + cov_ratio (CANONICAL, 纯 Cov/Var)."""
        return cls(f='softplus', stats=['cov_ratio'], coef=None)

    @classmethod
    def sp_regressed(cls):
        """softplus + cov_ratio + skew + kurt (OLS, 新系数, R²=0.79)."""
        return cls(f='softplus', stats=['cov_ratio', 'skew', 'kurt'],
                   coef=SOFTPLUS_REGRESSION_CFG.copy())

    @classmethod
    def relu_closed(cls):
        """relu + cov_ratio (纯 Cov/Var)."""
        return cls(f='relu', stats=['cov_ratio'], coef=None)

    # ── 核心估计 ──────────────────────────────────────────

    def _estimate(self, scores):
        """单 head τ* 估计."""
        valid = scores > -1e4
        s = scores[valid].float()
        if s.numel() < 10:
            return 1.0, {'cov_ratio': 0.0, 'n_valid': s.numel()}

        phi = self._transform_fn(s, self.eps)
        log_phi = phi.log()

        if self.cov_on == 'phi':
            # phi 域协方差: Cov(phi, log(phi))
            mean_phi = phi.mean()
            cov = ((phi - mean_phi) * (log_phi - log_phi.mean())).mean()
            denom = log_phi.var().clamp(min=self.eps)
        else:
            # s 域协方差: Cov(s, log(phi))
            mean_s = s.mean()
            delta_s = s - mean_s
            delta_lp = log_phi - log_phi.mean()
            cov = (delta_s * delta_lp).mean()
            denom = log_phi.var().clamp(min=self.eps)
        cov_ratio = (cov / denom).item()

        stats = {'cov_ratio': round(cov_ratio, 4), 'n_valid': int(s.numel())}

        if self.coef is None:
            tau_val = cov_ratio
        else:
            c = self.coef
            tau_val = c.get('coef_bias', 1.0) + c['coef_cov'] * cov_ratio
            if 'skew' in self.stats:
                mean_s = s.mean()
                std_s = s.std().clamp(min=self.eps)
                z = (s - mean_s) / std_s
                skew = (z ** 3).mean().item()
                tau_val += c['coef_skew'] * math.tanh(skew / 5.0)
                stats['skew'] = round(skew, 4)
            if 'kurt' in self.stats:
                if 'skew' not in self.stats:
                    std_s = s.std().clamp(min=self.eps)
                    z = (s - mean_s) / std_s
                kurt = (z ** 4).mean().item() - 3.0
                tau_val += c['coef_kurt'] * math.tanh(kurt / 10.0)
                stats['kurt'] = round(kurt, 4)

        tau_val = max(self.tau_min, min(self.tau_max, tau_val))
        stats['tau'] = round(tau_val, 4)
        return tau_val, stats

    def batch(self, scores):
        """批量估计 [B, H, Lq, Lk] → [H] tensor.

        Args:
            scores: [B, H, Lq, Lk] pre-softmax attention scores

        Returns:
            taus: [H] tensor
            stats_list: list of per-head stats dicts
        """
        H = scores.shape[1]
        taus = torch.zeros(H, dtype=torch.float32, device=scores.device)
        stats_list = []
        for h in range(H):
            tau_h, stats_h = self._estimate(scores[0, h])
            taus[h] = tau_h
            stats_list.append(stats_h)
        return taus, stats_list

    def per_layer(self, scores, layer_idx):
        """逐层 τ* 估计 (使用 PER_LAYER_CFG 系数).

        仅当 coef 为 None 时可用 (纯 Cov/Var 基底 + 逐层回归系数).
        对未知 layer_idx 回退到 PER_CLUSTER_CFG.

        Args:
            scores: [Lq, Lk] pre-softmax attention scores
            layer_idx: int

        Returns:
            tau: float
            stats: dict
        """
        c = PER_LAYER_CFG.get(layer_idx)
        if c is None:
            n_layers = len(PER_LAYER_CFG)
            third = max(PER_LAYER_CFG.keys()) // 3 if n_layers > 0 else 9
            if layer_idx < third:
                c = PER_CLUSTER_CFG['shallow']
            elif layer_idx < 2 * third:
                c = PER_CLUSTER_CFG['middle']
            else:
                c = PER_CLUSTER_CFG['deep']
        old_coef = self.coef
        self.coef = c
        tau, stats = self._estimate(scores)
        self.coef = old_coef
        return tau, stats

    def per_cluster(self, scores, layer_idx):
        """三簇 τ* 估计 (shallow/middle/deep).

        Args:
            scores: [Lq, Lk] pre-softmax attention scores
            layer_idx: int

        Returns:
            tau: float
            stats: dict
        """
        n_layers = len(PER_LAYER_CFG)
        third = max(PER_LAYER_CFG.keys()) // 3 if n_layers > 0 else 9
        if layer_idx < third:
            c = PER_CLUSTER_CFG['shallow']
        elif layer_idx < 2 * third:
            c = PER_CLUSTER_CFG['middle']
        else:
            c = PER_CLUSTER_CFG['deep']
        old_coef = self.coef
        self.coef = c
        tau, stats = self._estimate(scores)
        self.coef = old_coef
        return tau, stats

    # ── 重标定 ──────────────────────────────────────────

    def recalibrate(self, pairs):
        """OLS 拟合新系数.

        Args:
            pairs: list of (scores_matrix, tau_ground_truth)
                scores: [Lq, Lk] pre-softmax attention scores
                gt_tau: float — verified τ* (e.g. from grid-search)

        Returns:
            coef: dict — 新系数
            r2: float — OLS R²
        """
        rows = []
        targets = []

        for scores, gt_tau in pairs:
            valid = scores > -1e4
            s = scores[valid].float()
            if s.numel() < 10:
                continue

            phi = self._transform_fn(s, self.eps)
            log_phi = phi.log()

            if self.cov_on == 'phi':
                mean_phi = phi.mean()
                cov = ((phi - mean_phi) * (log_phi - log_phi.mean())).mean()
            else:
                mean_s = s.mean()
                cov = ((s - mean_s) * (log_phi - log_phi.mean())).mean()
            denom = log_phi.var().clamp(min=self.eps)
            cov_ratio = (cov / denom).item()

            features = [cov_ratio]
            if 'skew' in self.stats:
                mean_s = s.mean()
                std_s = s.std().clamp(min=self.eps)
                z = (s - mean_s) / std_s
                skew = (z ** 3).mean().item()
                features.append(math.tanh(skew / 5.0))
            if 'kurt' in self.stats:
                if 'skew' not in self.stats:
                    std_s = s.std().clamp(min=self.eps)
                    z = (s - mean_s) / std_s
                kurt = (z ** 4).mean().item() - 3.0
                features.append(math.tanh(kurt / 10.0))

            rows.append(features)
            targets.append(gt_tau)

        if len(rows) < 10:
            return (self.coef or {}).copy(), 0.0

        X = torch.tensor(rows)
        y = torch.tensor(targets)
        coeffs = torch.linalg.lstsq(X, y).solution
        r2 = 1.0 - ((y - X @ coeffs) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        r2 = r2.item()

        new_coef = {}
        idx = 0
        new_coef['coef_cov'] = round(coeffs[idx].item(), 4)
        idx += 1
        if 'skew' in self.stats:
            new_coef['coef_skew'] = round(coeffs[idx].item(), 4)
            idx += 1
        if 'kurt' in self.stats:
            new_coef['coef_kurt'] = round(coeffs[idx].item(), 4)
            idx += 1
        new_coef['tau_min'] = self.tau_min
        new_coef['tau_max'] = self.tau_max

        self.coef = new_coef
        return new_coef, round(r2, 4)


# ═══════════════════════════════════════════════════════════════
# 向后兼容 — 旧函数接口
# ═══════════════════════════════════════════════════════════════

# 全局默认估计器
_DEFAULT_SP_CLOSED = TauEstimator.sp_closed()
_DEFAULT_SP_REGRESSED = TauEstimator.sp_regressed()
_DEFAULT_CLAMP_CLOSED = TauEstimator.clamp_closed()
_DEFAULT_CLAMP_REGRESSED = TauEstimator.clamp_regressed()
_DEFAULT_CLAMP_LEGACY = TauEstimator.clamp_legacy()

# 旧常量
REGRESSION_CFG = SOFTPLUS_REGRESSION_CFG


def tau_star(scores):
    """[τ*-SP-closed] CANONICAL — softplus + cov_ratio."""
    return _DEFAULT_SP_CLOSED(scores)


def tau_star_sp_closed(scores):
    """[τ*-SP-closed] CANONICAL — softplus + cov_ratio (别名)."""
    return _DEFAULT_SP_CLOSED(scores)


def tau_star_closed_sp(scores):
    """[τ*-SP-closed] CANONICAL — softplus + cov_ratio (别名)."""
    return _DEFAULT_SP_CLOSED(scores)


def tau_star_regressed(scores, cfg=None):
    """[τ*-SP-regressed] softplus + cov_ratio + skew + kurt (OLS)."""
    if cfg:
        est = TauEstimator(f='softplus', stats=['cov_ratio', 'skew', 'kurt'],
                           coef={**SOFTPLUS_REGRESSION_CFG, **cfg})
        return est(scores)
    return _DEFAULT_SP_REGRESSED(scores)


def tau_star_sp(scores):
    """[τ*-SP-regressed] 别名."""
    return _DEFAULT_SP_REGRESSED(scores)


def tau_star_clamp(scores):
    """[τ*-clamp-closed] clamp + cov_ratio (纯 Cov/Var)."""
    return _DEFAULT_CLAMP_CLOSED(scores)


def tau_star_clamp_regressed(scores):
    """[τ*-clamp-regressed] clamp + cov_ratio + skew + kurt (OLS)."""
    return _DEFAULT_CLAMP_REGRESSED(scores)


def tau_star_legacy(scores):
    """[τ*-clamp-legacy] 匹配 _legacy 时代 tau_star() 行为.

    clamp + cov_on='phi' + cov_ratio + skew + kurt (OLS, r=0.9793).
    """
    return _DEFAULT_CLAMP_LEGACY(scores)


def tau_star_batch(scores, variant='closed'):
    """批量 τ* 估计.

    Args:
        scores: [B, H, Lq, Lk]
        variant: 'closed' | 'regressed' | 'clamp' | 'clamp_regressed'

    Returns:
        taus: [H] tensor
        stats_list: list
    """
    est_map = {
        'closed': _DEFAULT_SP_CLOSED,
        'regressed': _DEFAULT_SP_REGRESSED,
        'clamp': _DEFAULT_CLAMP_CLOSED,
        'clamp_regressed': _DEFAULT_CLAMP_REGRESSED,
        'legacy': _DEFAULT_CLAMP_LEGACY,
    }
    est = est_map.get(variant, _DEFAULT_SP_CLOSED)
    return est.batch(scores)


def tau_star_perlayer(scores, layer_idx):
    """逐层 τ* 估计 (softplus 基底)."""
    return _DEFAULT_SP_CLOSED.per_layer(scores, layer_idx)


def tau_star_percluster(scores, layer_idx):
    """三簇 τ* 估计 (softplus 基底)."""
    return _DEFAULT_SP_CLOSED.per_cluster(scores, layer_idx)


def recalibrate_tau_star_sp(pairs):
    """重标定 softplus 回归系数."""
    est = TauEstimator.sp_regressed()
    return est.recalibrate(pairs)