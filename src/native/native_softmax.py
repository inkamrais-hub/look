"""
native_softmax.py — fused τ-softplus normalization CUDA kernel. Replaces v12-v14 suite.
Single kernel for log_softplus + tau*log + softmax. Supports fp16 + bf16 dispatch.

文件角色: s^τ = softplus(s)^τ 融合归一化 CUDA JIT kernel
调用链:
  stau_softmax_fused(scores, tau) → StauSoftmaxFusedFn.apply() → CUDA fwd/bwd
  autograd 兼容: 可直接用于 nn.Module.forward 中
数据流:
  scores [N, D] fp16/bf16
    → CUDA fwd: softplus_stable → log → tau*log → exp-max → warp-reduce rcp
    → attn [N, D] (归一化权重) + sp_buf [N, D] (softplus 值, 存 ctx)
    → CUDA bwd: d_P → d_s = tau * P*(d_P - ΣP*d_P) * sigmoid'(s)
    → d_scores [N, D] + d_tau scalar

实现:
  [SP] CANONICAL — softplus(s)^τ / Σ (匹配 numpy.stau_norm)
  无 mask, 无 dropout — 纯数学融合归一化
"""

import torch
from torch.utils.cpp_extension import load_inline

EPS = 1e-8
BLOCK_DIM = 128

CPP_SRC = r'''
#define CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
#include <torch/extension.h>

void tausp_fwd(torch::Tensor scores, torch::Tensor attn, torch::Tensor sp_buf,
               double tau, double eps);
void tausp_bwd(torch::Tensor grad_out, torch::Tensor attn, torch::Tensor sp_buf,
               torch::Tensor d_scores, torch::Tensor d_tau_row,
               double tau, double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tausp_fwd", &tausp_fwd, "");
    m.def("tausp_bwd", &tausp_bwd, "");
}
'''

CUDA_SRC = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>

#define BLOCK 128
#define WARP 32

__device__ __forceinline__ float softplus_stable(float x) {
    return fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
}

__device__ __forceinline__ void warp_reduce_max(float* val) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        *val = fmaxf(*val, __shfl_xor_sync(0xffffffff, *val, off));
}

__device__ __forceinline__ void warp_reduce_sum(float* val) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        *val += __shfl_xor_sync(0xffffffff, *val, off);
}

template<typename T> __device__ __forceinline__ float to_f32(T v) { return (float)v; }
template<> __device__ __forceinline__ float to_f32<__half>(__half v) { return __half2float(v); }
template<> __device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template<typename T> __device__ __forceinline__ T from_f32(float v) { return (T)v; }
template<> __device__ __forceinline__ __half from_f32<__half>(float v) { return __float2half_rn(v); }
template<> __device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

template<typename T> __device__ __forceinline__ float load_float(const T* p, int i) {
    return to_f32<T>(p[i]);
}
template<typename T> __device__ __forceinline__ void store_float(T* p, int i, float v) {
    p[i] = from_f32<T>(v);
}

template<typename scalar_t>
__global__ void fwd_kernel(
    const scalar_t* __restrict__ scores,
    scalar_t* __restrict__ attn,
    scalar_t* __restrict__ sp_buf,
    int rows, int cols, float tau, float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    const int N_WARPS = BLOCK / WARP;

    __shared__ float sm_red[N_WARPS];

    const scalar_t* s_row = scores + row * cols;
    scalar_t* sp_row = sp_buf + row * cols;
    scalar_t* a_row = attn + row * cols;

    float max_val = -1e30f;

    for (int i = tid; i < cols; i += BLOCK) {
        float s = to_f32<scalar_t>(s_row[i]);
        float sp = softplus_stable(s) + eps;
        store_float<scalar_t>(sp_row, i, sp);
        float logit = tau * logf(sp);
        max_val = fmaxf(max_val, logit);
    }

    warp_reduce_max(&max_val);
    if (lane == 0) sm_red[warp_id] = max_val;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < N_WARPS) ? sm_red[lane] : -1e30f;
        warp_reduce_max(&val);
        if (lane == 0) sm_red[0] = val;
    }
    __syncthreads();
    max_val = sm_red[0];

    float sum_exp = 0.0f;

    for (int i = tid; i < cols; i += BLOCK) {
        float sp = to_f32<scalar_t>(sp_row[i]);
        float logit = tau * logf(sp);
        float e = expf(logit - max_val);
        store_float<scalar_t>(a_row, i, e);
        sum_exp += e;
    }

    warp_reduce_sum(&sum_exp);
    if (lane == 0) sm_red[warp_id] = sum_exp;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < N_WARPS) ? sm_red[lane] : 0.0f;
        warp_reduce_sum(&val);
        if (lane == 0) sm_red[0] = val;
    }
    __syncthreads();
    float inv_sum = 1.0f / (sm_red[0] + eps);

    for (int i = tid; i < cols; i += BLOCK) {
        float e = to_f32<scalar_t>(a_row[i]);
        store_float<scalar_t>(a_row, i, e * inv_sum);
    }
}

template<typename scalar_t>
__global__ void bwd_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ attn,
    const scalar_t* __restrict__ sp_buf,
    scalar_t* __restrict__ d_scores,
    float* __restrict__ d_tau_row,
    int rows, int cols, float tau, float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    const int N_WARPS = BLOCK / WARP;

    __shared__ float sm_red[N_WARPS];

    const scalar_t* go_row = grad_out + row * cols;
    const scalar_t* a_row = attn + row * cols;
    const scalar_t* sp_row = sp_buf + row * cols;
    scalar_t* ds_row = d_scores + row * cols;

    float grad_dot = 0.0f;
    float log_dot = 0.0f;

    for (int i = tid; i < cols; i += BLOCK) {
        float go = to_f32<scalar_t>(go_row[i]);
        float p = to_f32<scalar_t>(a_row[i]);
        float sp = to_f32<scalar_t>(sp_row[i]);
        grad_dot += p * go;
        log_dot += p * logf(sp);
    }

    warp_reduce_sum(&grad_dot);
    if (lane == 0) sm_red[warp_id] = grad_dot;
    __syncthreads();

    if (warp_id == 0) {
        float vg = (lane < N_WARPS) ? sm_red[lane] : 0.0f;
        warp_reduce_sum(&vg);
        if (lane == 0) sm_red[0] = vg;
    }
    __syncthreads();
    grad_dot = sm_red[0];

    warp_reduce_sum(&log_dot);
    if (lane == 0) sm_red[warp_id] = log_dot;
    __syncthreads();

    if (warp_id == 0) {
        float vl = (lane < N_WARPS) ? sm_red[lane] : 0.0f;
        warp_reduce_sum(&vl);
        if (lane == 0) sm_red[0] = vl;
    }
    __syncthreads();
    log_dot = sm_red[0];

    float local_d_tau = 0.0f;

    for (int i = tid; i < cols; i += BLOCK) {
        float go = to_f32<scalar_t>(go_row[i]);
        float p = to_f32<scalar_t>(a_row[i]);
        float sp = to_f32<scalar_t>(sp_row[i]);

        float sm_grad = p * (go - grad_dot);
        float sp_grad = (1.0f - expf(-sp)) / (sp + eps);
        float ds = tau * sm_grad * sp_grad;
        store_float<scalar_t>(ds_row, i, ds);

        local_d_tau += go * p * (logf(sp) - log_dot);
    }

    warp_reduce_sum(&local_d_tau);
    if (lane == 0) sm_red[warp_id] = local_d_tau;
    __syncthreads();

    if (warp_id == 0) {
        float tau_sum = (lane < N_WARPS) ? sm_red[lane] : 0.0f;
        warp_reduce_sum(&tau_sum);
        if (lane == 0) d_tau_row[row] = tau_sum;
    }
}

template<typename scalar_t>
void launch_fwd(torch::Tensor scores, torch::Tensor attn, torch::Tensor sp_buf,
                double tau, double eps) {
    int rows = scores.size(0);
    int cols = scores.size(1);
    fwd_kernel<scalar_t><<<rows, BLOCK>>>(
        reinterpret_cast<const scalar_t*>(scores.data_ptr()),
        reinterpret_cast<scalar_t*>(attn.data_ptr()),
        reinterpret_cast<scalar_t*>(sp_buf.data_ptr()),
        rows, cols, (float)tau, (float)eps);
}

template<typename scalar_t>
void launch_bwd(torch::Tensor grad_out, torch::Tensor attn, torch::Tensor sp_buf,
                torch::Tensor d_scores, torch::Tensor d_tau_row,
                double tau, double eps) {
    int rows = attn.size(0);
    int cols = attn.size(1);
    bwd_kernel<scalar_t><<<rows, BLOCK>>>(
        reinterpret_cast<const scalar_t*>(grad_out.data_ptr()),
        reinterpret_cast<const scalar_t*>(attn.data_ptr()),
        reinterpret_cast<const scalar_t*>(sp_buf.data_ptr()),
        reinterpret_cast<scalar_t*>(d_scores.data_ptr()),
        d_tau_row.data_ptr<float>(),
        rows, cols, (float)tau, (float)eps);
}

void tausp_fwd(torch::Tensor scores, torch::Tensor attn, torch::Tensor sp_buf,
               double tau, double eps) {
    auto dtype = scores.scalar_type();
    if (dtype == c10::kHalf) {
        launch_fwd<__half>(scores, attn, sp_buf, tau, eps);
    } else if (dtype == c10::kBFloat16) {
        launch_fwd<__nv_bfloat16>(scores, attn, sp_buf, tau, eps);
    } else {
        throw std::runtime_error("tausp_fwd: only fp16 and bf16 supported");
    }
}

void tausp_bwd(torch::Tensor grad_out, torch::Tensor attn, torch::Tensor sp_buf,
               torch::Tensor d_scores, torch::Tensor d_tau_row,
               double tau, double eps) {
    auto dtype = attn.scalar_type();
    if (dtype == c10::kHalf) {
        launch_bwd<__half>(grad_out, attn, sp_buf, d_scores, d_tau_row, tau, eps);
    } else if (dtype == c10::kBFloat16) {
        launch_bwd<__nv_bfloat16>(grad_out, attn, sp_buf, d_scores, d_tau_row, tau, eps);
    } else {
        throw std::runtime_error("tausp_bwd: only fp16 and bf16 supported");
    }
}
'''

_cached = None

def _load():
    global _cached
    if _cached is None:
        import platform
        is_win = platform.system() == 'Windows'
        _cached = load_inline(
            name='s_tau_sp_fused_native',
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            verbose=False,
            extra_cflags=['/Zc:preprocessor'] if is_win else [],
            extra_cuda_cflags=['-Xcompiler', '/Zc:preprocessor', '--use_fast_math'] if is_win else ['--use_fast_math'],
        )
    return _cached


class StauSoftmaxFusedFn(torch.autograd.Function):
    """Fused τ-softplus normalization autograd Function.

    forward:  scores [N, D] → CUDA fwd → attn [N, D] (归一化权重)
             saves sp_buf for backward
    backward: grad_out [N, D] → CUDA bwd → d_scores [N, D], d_tau scalar
    """

    @staticmethod
    def forward(ctx, scores, tau, eps=EPS):
        if scores.dim() != 2:
            raise ValueError(f"stau_softmax_fused expects 2D input [N, D], got shape {scores.shape}")

        rows, cols = scores.shape
        dtype = scores.dtype
        scores = scores.contiguous()

        attn = torch.empty_like(scores)
        sp_buf = torch.empty_like(scores)

        _load().tausp_fwd(scores, attn, sp_buf, float(tau), float(eps))

        ctx.eps = eps
        ctx.rows = rows
        ctx.cols = cols
        ctx.tau_is_tensor = isinstance(tau, torch.Tensor)
        ctx.save_for_backward(attn, sp_buf, torch.tensor(float(tau), device=scores.device))
        return attn

    @staticmethod
    def backward(ctx, grad_out):
        attn, sp_buf, tau_tensor = ctx.saved_tensors
        tau_val = float(tau_tensor.item())
        eps = ctx.eps
        rows = ctx.rows

        grad_out = grad_out.contiguous()

        d_scores = torch.empty_like(attn)
        d_tau_row = torch.empty(rows, dtype=torch.float32, device=attn.device)

        _load().tausp_bwd(grad_out, attn, sp_buf, d_scores, d_tau_row, tau_val, float(eps))

        d_tau = d_tau_row.sum() if ctx.tau_is_tensor else None
        return d_scores, d_tau, None


def stau_softmax_fused(scores, tau, eps=EPS):
    """Exported τ-softplus fused normalization.

    Auto-dispatches between fp16 and bf16 CUDA kernels based on scores.dtype.

    Args:
        scores: [N, D] pre-softmax attention scores, fp16 or bf16
        tau:    scalar ∈ (1, ∞),  1.0=wide  20.0=telephoto
        eps:    numerical stability floor

    Returns:
        attn: [N, D] normalized τ-softplus weights (Σ = 1 per row)
    """
    if scores.device.type != 'cuda':
        raise RuntimeError("stau_softmax_fused requires CUDA tensor")
    if scores.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(f"stau_softmax_fused requires fp16 or bf16, got {scores.dtype}")

    return StauSoftmaxFusedFn.apply(scores, tau, eps)


_AVAILABLE = True


if __name__ == "__main__":
    import torch.nn.functional as F

    print("=" * 70)
    print("  native_softmax.py — τ-softplus fused CUDA kernel 验证")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA 不可用")
        exit(0)

    DEVICE = 'cuda'
    DTYPE = torch.float16
    TAU = 4.0
    EPS_TEST = 1e-8
    N_ROWS = 8
    D_COLS = 1024

    print(f"  device={DEVICE}, dtype={DTYPE}, tau={TAU}, shape=[{N_ROWS}, {D_COLS}]")
    print()

    torch.manual_seed(42)
    scores = torch.randn(N_ROWS, D_COLS, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def ref_tau_softplus(s, tau, eps=EPS_TEST):
        sp = F.softplus(s.float()) + eps
        log_sp = sp.log()
        logits = tau * log_sp
        w = F.softmax(logits, dim=-1)
        return w

    ref_attn = ref_tau_softplus(scores, TAU)
    print(f"  [ref]   PyTorch eager:    shape={ref_attn.shape}, sum≈{ref_attn.sum(-1).mean().item():.6f}")

    cuda_attn = stau_softmax_fused(scores, TAU, eps=EPS_TEST)
    print(f"  [cuda]  fused kernel:     shape={cuda_attn.shape}, sum≈{cuda_attn.sum(-1).mean().item():.6f}")

    diff = (cuda_attn.float() - ref_attn.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"  [fwd]   差异: max={max_diff:.6e}, mean={mean_diff:.6e}")

    fwd_ok = max_diff < 0.02
    print(f"  [fwd]   {'PASS' if fwd_ok else 'FAIL'} (threshold 0.02)")
    print()

    scores_f32 = scores.detach().clone().float().requires_grad_(True)
    ref_out = ref_tau_softplus(scores_f32, TAU)
    grad_out_f32 = torch.randn(N_ROWS, D_COLS, dtype=torch.float32, device=DEVICE) / D_COLS
    ref_out.backward(grad_out_f32)
    ref_grad = scores_f32.grad.clone()

    cuda_out = stau_softmax_fused(scores, TAU, eps=EPS_TEST)
    grad_out = torch.randn(N_ROWS, D_COLS, dtype=DTYPE, device=DEVICE) / D_COLS
    cuda_out.backward(grad_out)
    cuda_grad = scores.grad.clone()

    grad_diff = (cuda_grad.float() - ref_grad).abs()
    grad_max_diff = grad_diff.max().item()
    grad_mean_diff = grad_diff.mean().item()
    print(f"  [bwd]   梯度差异: max={grad_max_diff:.6e}, mean={grad_mean_diff:.6e}")

    grad_ok = grad_max_diff < 0.05
    print(f"  [bwd]   {'PASS' if grad_ok else 'FAIL'} (threshold 0.05)")
    print()

    if fwd_ok and grad_ok:
        print("  [VERDICT] ALL TESTS PASSED")
    else:
        print("  [VERDICT] TESTS FAILED — need kernel check")

    bf16_available = torch.cuda.is_bf16_supported()
    if bf16_available:
        print()
        print(f"  [bf16]  {'bf16 可用' if bf16_available else 'bf16 不可用'}")
        if bf16_available:
            scores_bf16 = torch.randn(2, 256, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
            try:
                bf16_attn = stau_softmax_fused(scores_bf16, TAU, eps=EPS_TEST)
                ref_bf16 = ref_tau_softplus(scores_bf16, TAU)
                bf16_diff = (bf16_attn.float() - ref_bf16.float()).abs().max().item()
                print(f"  [bf16]  forward diff: max={bf16_diff:.6e}")
                bf16_ok = bf16_diff < 0.05
                print(f"  [bf16]  {'PASS' if bf16_ok else 'FAIL'}")

                scores_bf16.grad = None
                bf16_attn.backward(torch.randn(2, 256, dtype=torch.bfloat16, device=DEVICE) / 256)
                print(f"  [bf16]  backward: grad shape={scores_bf16.grad.shape}")
            except Exception as e:
                print(f"  [bf16]  ERROR: {e}")
    else:
        print()
        print("  [bf16]  本 GPU 不支持 bf16 (SM < 80)")

    print()
    print("=" * 70)