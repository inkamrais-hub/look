# τ-softplus 测试摘要

## 核心测试套件: test_tau_star_sp.py

> 位置: `F:\τ\τ-softplus\tests\test_tau_star_sp.py`

### 测试内容（14 个测试函数, 37 个 check 点）

| 编号 | 测试 | 说明 |
|:---:|:-----|:-----|
| 1 | `test_norm_variants` | [SP] CANONICAL, [SP-M] max-stabilized, [SM] softmax 归一化验证 |
| 2 | `test_tau_star_closed` | τ* 闭式解 Cov(s,log(sp))/Var(log(sp)) |
| 3 | `test_tau_star_regressed` | τ* 回归公式 (Cov+Skew+Kurt) |
| 4 | `test_closed_vs_regressed` | 闭式 vs 回归的一致性 |
| 5 | `test_tau_star_batch_fn` | 批量 τ* 估计 |
| 6 | `test_numpy_parity` | NumPy vs PyTorch 数值等价性 |
| 7 | `test_roundtrip` | τ* → stau_norm → 验证 |
| 8 | `test_tau_estimator_presets` | TauEstimator 预置构造器 |
| 9 | `test_tau_estimator_custom` | 自定义参数估计器 |
| 10 | `test_tau_estimator_batch` | 批量估计 |
| 11 | `test_tau_estimator_perlayer` | 逐层/逐簇估计 |
| 12 | `test_tau_estimator_recalibrate` | OLS 重标定 |
| 13 | `test_backward_compat` | 向后兼容函数接口 |
| 14 | `test_transform_consistency` | 各 f(s) 变换的一致性 |

### 运行方法

```bash
cd F:\τ\τ-softplus
D:\python\python.exe tests\test_tau_star_sp.py
```

### 辅助测试

| 测试 | 位置 | 说明 |
|:----|:-----|:-----|
| `test_autocal.py` | `stau/tests/` | TauAutoCal 自动化校准 (3 tests) |
| `test_native_ops.py` | `stau/tests/` | CUDA 原生算子测试 |
| `demo_bandit.py` | `stau/` | 老虎机验证 demo |
| `demo_inject.py` | `stau/` | 注入生成 demo |