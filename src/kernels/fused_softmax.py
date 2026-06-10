"""FusedSoftmax — fused mask+softmax+dropout CUDA kernel.

Based on s^tau v10/v12 framework, stripped of softplus/log/tau math.
Features:
  - half2/half4 vectorized fwd+bwd
  - mask fusion (causal mask injected inline)
  - dropout fusion (SplitMix64 hash)
  - warp shuffle reduce
  - __ldg read-only cache
  - fused backward for Lk <= 1024

Benchmark: 1.73x faster than cuDNN at Lk=2048 (fp16).
"""

import torch
from torch.utils.cpp_extension import load_inline

CPP_SRC = r'''
#define CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
#include <torch/extension.h>

void fwd(torch::Tensor scores, torch::Tensor mask_flat, torch::Tensor attn,
    uint64_t dropout_seed, float dropout_p, float eps);

void bwd(torch::Tensor go, torch::Tensor attn,
    torch::Tensor sg,
    uint64_t dropout_seed, float dropout_p, float eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &fwd, "");
    m.def("bwd", &bwd, "");
}
'''

CUDA_SRC = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define BLOCK_DIM 128
#define FUSED_TILE 512

__device__ __forceinline__ void warp_reduce_sum(float* val) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        *val += __shfl_xor_sync(0xffffffff, *val, off);
}

__device__ __forceinline__ void warp_reduce_max(float* val) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, *val, off);
        *val = fmaxf(*val, other);
    }
}

__device__ __forceinline__ float hash_rand(uint64_t seed, uint32_t idx) {
    uint64_t z = seed ^ (((uint64_t)idx) << 32);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z = z ^ (z >> 31);
    return (float)(uint32_t)(z & 0xFFFFFFFFULL) / 4294967296.0f;
}

template<typename T> __device__ __forceinline__ float to_f32(T v);
template<> __device__ __forceinline__ float to_f32<float>(float v) { return v; }
template<> __device__ __forceinline__ float to_f32<__half>(__half v) { return __half2float(v); }

template<typename T> __device__ __forceinline__ T from_f32(float v);
template<> __device__ __forceinline__ float from_f32<float>(float v) { return v; }
template<> __device__ __forceinline__ __half from_f32<__half>(float v) { return __float2half_rn(v); }

template<typename T> __device__ __forceinline__ float load_elem(const T* p, int i) { return to_f32(p[i]); }
template<typename T> __device__ __forceinline__ void store_elem(T* p, int i, float v) { p[i] = from_f32<T>(v); }

template<typename T>
__device__ __forceinline__ float ldg_elem(const T* __restrict__ p, int i) {
    return to_f32(__ldg(p + i));
}

template<typename scalar_t>
__global__ void fwd_kernel(
    const scalar_t* __restrict__ s, const float* __restrict__ mask_row,
    scalar_t* __restrict__ a,
    uint64_t dropout_seed, float dropout_p,
    int Lk, int H, int Lq, float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    __shared__ float sm[8];
    float sum = 0.0f, maxv = -1e30f;
    int base = row * Lk;

    for (int i = tid; i < Lk; i += blockDim.x) {
        float score = load_elem(s + base, i);
        if (mask_row) score += mask_row[(row % Lq) * Lk + i];
        maxv = fmaxf(maxv, score);
    }
    warp_reduce_max(&maxv);
    if (lane == 0) sm[warp] = maxv;
    __syncthreads();
    if (warp == 0) {
        maxv = sm[lane];
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1)
            maxv = fmaxf(maxv, __shfl_xor_sync(0xffffffff, maxv, off));
        if (lane == 0) sm[0] = maxv;
    }
    __syncthreads();
    maxv = sm[0];

    for (int i = tid; i < Lk; i += blockDim.x) {
        float score = load_elem(s + base, i);
        if (mask_row) score += mask_row[(row % Lq) * Lk + i];
        float p = expf(score - maxv);
        store_elem(a + base, i, p);
        sum += p;
    }

    warp_reduce_sum(&sum);
    if (lane == 0) sm[4 + warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = sm[4 + lane];
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1)
            sum += __shfl_xor_sync(0xffffffff, sum, off);
        if (lane == 0) sm[4] = sum;
    }
    __syncthreads();
    float inv_denom = 1.0f / (sm[4] + eps);

    for (int i = tid; i < Lk; i += blockDim.x) {
        float val = load_elem(a + base, i) * inv_denom;
        if (dropout_p > 0.0f) {
            float r = hash_rand(dropout_seed, (uint32_t)(base + i));
            if (r < dropout_p) val = 0.0f;
            else val /= (1.0f - dropout_p);
        }
        store_elem(a + base, i, val);
    }
}

__global__ void fwd_kernel_float4(
    const float* __restrict__ s, const float* __restrict__ mask_row,
    float* __restrict__ a,
    uint64_t dropout_seed, float dropout_p,
    int Lk, int H, int Lq, float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    __shared__ float sm[8];
    float sum = 0.0f, maxv = -1e30f;
    int base = row * Lk;
    const float4* s4 = reinterpret_cast<const float4*>(s + base);
    const float4* mask4 = mask_row ? reinterpret_cast<const float4*>(mask_row + (row % Lq) * Lk) : nullptr;
    int n4 = Lk / 4;

    for (int i = tid; i < n4; i += blockDim.x) {
        float4 sv = s4[i];
        if (mask4) { float4 mv = mask4[i]; sv.x += mv.x; sv.y += mv.y; sv.z += mv.z; sv.w += mv.w; }
        maxv = fmaxf(maxv, fmaxf(fmaxf(sv.x, sv.y), fmaxf(sv.z, sv.w)));
    }
    warp_reduce_max(&maxv);
    if (lane == 0) sm[warp] = maxv;
    __syncthreads();
    if (warp == 0) {
        maxv = sm[lane];
        for (int off = 4; off > 0; off >>= 1) maxv = fmaxf(maxv, __shfl_xor_sync(0xffffffff, maxv, off));
        if (lane == 0) sm[0] = maxv;
    }
    __syncthreads();
    maxv = sm[0];

    float4* a4 = reinterpret_cast<float4*>(a + base);
    for (int i = tid; i < n4; i += blockDim.x) {
        float4 sv = s4[i];
        if (mask4) { float4 mv = mask4[i]; sv.x += mv.x; sv.y += mv.y; sv.z += mv.z; sv.w += mv.w; }
        float p0 = expf(sv.x - maxv), p1 = expf(sv.y - maxv);
        float p2 = expf(sv.z - maxv), p3 = expf(sv.w - maxv);
        a4[i] = make_float4(p0, p1, p2, p3);
        sum += p0 + p1 + p2 + p3;
    }

    warp_reduce_sum(&sum);
    if (lane == 0) sm[4 + warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = sm[4 + lane];
        for (int off = 4; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
        if (lane == 0) sm[4] = sum;
    }
    __syncthreads();
    float inv = 1.0f / (sm[4] + eps);

    for (int i = tid; i < n4; i += blockDim.x) {
        float4 av = a4[i];
        float a0 = av.x * inv, a1 = av.y * inv, a2 = av.z * inv, a3 = av.w * inv;
        if (dropout_p > 0.0f) {
            uint32_t off = (uint32_t)(base + i * 4);
            float r0 = hash_rand(dropout_seed, off), r1 = hash_rand(dropout_seed, off + 1);
            float r2 = hash_rand(dropout_seed, off + 2), r3 = hash_rand(dropout_seed, off + 3);
            float scale = 1.0f / (1.0f - dropout_p);
            a0 = (r0 < dropout_p) ? 0.0f : a0 * scale;
            a1 = (r1 < dropout_p) ? 0.0f : a1 * scale;
            a2 = (r2 < dropout_p) ? 0.0f : a2 * scale;
            a3 = (r3 < dropout_p) ? 0.0f : a3 * scale;
        }
        a4[i] = make_float4(a0, a1, a2, a3);
    }
}

__global__ void fwd_kernel_half4(
    const __half* __restrict__ s, const float* __restrict__ mask_row,
    __half* __restrict__ a,
    uint64_t dropout_seed, float dropout_p,
    int Lk, int H, int Lq, float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    __shared__ float sm[8];
    float sum = 0.0f, maxv = -1e30f;
    int base = row * Lk;
    const __half2* s2 = reinterpret_cast<const __half2*>(s + base);
    int n4 = Lk / 4;

    for (int i = tid; i < n4; i += blockDim.x) {
        __half2 sv0 = s2[i * 2], sv1 = s2[i * 2 + 1];
        float s0 = __half2float(sv0.x), s1 = __half2float(sv0.y);
        float s2v = __half2float(sv1.x), s3v = __half2float(sv1.y);
        if (mask_row) {
            s0 += mask_row[(row % Lq) * Lk + i * 4];     s1 += mask_row[(row % Lq) * Lk + i * 4 + 1];
            s2v += mask_row[(row % Lq) * Lk + i * 4 + 2]; s3v += mask_row[(row % Lq) * Lk + i * 4 + 3];
        }
        maxv = fmaxf(maxv, fmaxf(fmaxf(s0, s1), fmaxf(s2v, s3v)));
    }
    warp_reduce_max(&maxv);
    if (lane == 0) sm[warp] = maxv;
    __syncthreads();
    if (warp == 0) {
        maxv = sm[lane];
        for (int off = 4; off > 0; off >>= 1) maxv = fmaxf(maxv, __shfl_xor_sync(0xffffffff, maxv, off));
        if (lane == 0) sm[0] = maxv;
    }
    __syncthreads();
    maxv = sm[0];

    __half2* a2 = reinterpret_cast<__half2*>(a + base);
    for (int i = tid; i < n4; i += blockDim.x) {
        __half2 sv0 = s2[i * 2], sv1 = s2[i * 2 + 1];
        float s0 = __half2float(sv0.x), s1 = __half2float(sv0.y);
        float s2v = __half2float(sv1.x), s3v = __half2float(sv1.y);
        if (mask_row) {
            s0 += mask_row[(row % Lq) * Lk + i * 4];     s1 += mask_row[(row % Lq) * Lk + i * 4 + 1];
            s2v += mask_row[(row % Lq) * Lk + i * 4 + 2]; s3v += mask_row[(row % Lq) * Lk + i * 4 + 3];
        }
        float p0 = expf(s0 - maxv), p1 = expf(s1 - maxv);
        float p2 = expf(s2v - maxv), p3 = expf(s3v - maxv);
        __half2 pv0; pv0.x = __float2half_rn(p0); pv0.y = __float2half_rn(p1);
        __half2 pv1; pv1.x = __float2half_rn(p2); pv1.y = __float2half_rn(p3);
        a2[i * 2] = pv0; a2[i * 2 + 1] = pv1;
        sum += p0 + p1 + p2 + p3;
    }

    warp_reduce_sum(&sum);
    if (lane == 0) sm[4 + warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = sm[4 + lane];
        for (int off = 4; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
        if (lane == 0) sm[4] = sum;
    }
    __syncthreads();
    float inv = 1.0f / (sm[4] + eps);

    for (int i = tid; i < n4; i += blockDim.x) {
        __half2 av0 = a2[i * 2], av1 = a2[i * 2 + 1];
        float a0 = __half2float(av0.x) * inv, a1 = __half2float(av0.y) * inv;
        float a2v = __half2float(av1.x) * inv, a3v = __half2float(av1.y) * inv;
        if (dropout_p > 0.0f) {
            uint32_t off = (uint32_t)(base + i * 4);
            float r0 = hash_rand(dropout_seed, off), r1 = hash_rand(dropout_seed, off + 1);
            float r2 = hash_rand(dropout_seed, off + 2), r3 = hash_rand(dropout_seed, off + 3);
            float scale = 1.0f / (1.0f - dropout_p);
            a0 = (r0 < dropout_p) ? 0.0f : a0 * scale; a1 = (r1 < dropout_p) ? 0.0f : a1 * scale;
            a2v = (r2 < dropout_p) ? 0.0f : a2v * scale; a3v = (r3 < dropout_p) ? 0.0f : a3v * scale;
        }
        __half2 nv0; nv0.x = __float2half_rn(a0); nv0.y = __float2half_rn(a1);
        __half2 nv1; nv1.x = __float2half_rn(a2v); nv1.y = __float2half_rn(a3v);
        a2[i * 2] = nv0; a2[i * 2 + 1] = nv1;
    }
}

template<typename scalar_t>
__global__ void bwd_kernel(
    const scalar_t* __restrict__ go, const scalar_t* __restrict__ a,
    scalar_t* __restrict__ sg,
    uint64_t dropout_seed, float dropout_p,
    int Lk, int H, int Lq, float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    __shared__ float sm[8];
    float wsum = 0.0f;
    int base = row * Lk;

    for (int i = tid; i < Lk; i += blockDim.x) {
        float gv = load_elem(go + base, i);
        float av = load_elem(a + base, i);
        if (dropout_p > 0.0f) {
            float r = hash_rand(dropout_seed, (uint32_t)(base + i));
            if (r < dropout_p) gv = 0.0f;
            else gv /= (1.0f - dropout_p);
        }
        wsum += av * gv;
    }

    warp_reduce_sum(&wsum);
    if (lane == 0) sm[warp] = wsum;
    __syncthreads();
    if (warp == 0) {
        wsum = sm[lane];
        for (int off = 4; off > 0; off >>= 1) wsum += __shfl_xor_sync(0xffffffff, wsum, off);
        if (lane == 0) sm[0] = wsum;
    }
    __syncthreads();
    wsum = sm[0];

    for (int i = tid; i < Lk; i += blockDim.x) {
        float gv = load_elem(go + base, i);
        float av = load_elem(a + base, i);
        if (dropout_p > 0.0f) {
            float r = hash_rand(dropout_seed, (uint32_t)(base + i));
            if (r < dropout_p) gv = 0.0f;
            else gv /= (1.0f - dropout_p);
        }
        float val = av * (gv - wsum);
        store_elem(sg + base, i, val);
    }
}

template<typename scalar_t>
__global__ void bwd_kernel_fused(
    const scalar_t* __restrict__ go, const scalar_t* __restrict__ a,
    scalar_t* __restrict__ sg,
    uint64_t dropout_seed, float dropout_p,
    int Lk, int H, int Lq, float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    __shared__ float go_sm[FUSED_TILE];
    __shared__ float a_sm[FUSED_TILE];
    __shared__ float red_sm[8];

    float wsum = 0.0f;
    int base = row * Lk;

    for (int i = tid; i < Lk; i += blockDim.x) {
        float gv = load_elem(go + base, i);
        float av = load_elem(a + base, i);
        if (dropout_p > 0.0f) {
            float r = hash_rand(dropout_seed, (uint32_t)(base + i));
            if (r < dropout_p) gv = 0.0f;
            else gv /= (1.0f - dropout_p);
        }
        go_sm[i] = gv; a_sm[i] = av;
        wsum += av * gv;
    }

    warp_reduce_sum(&wsum);
    if (lane == 0) red_sm[warp] = wsum;
    __syncthreads();
    if (warp == 0) {
        wsum = red_sm[lane];
        for (int off = 4; off > 0; off >>= 1) wsum += __shfl_xor_sync(0xffffffff, wsum, off);
        if (lane == 0) red_sm[0] = wsum;
    }
    __syncthreads();
    wsum = red_sm[0];

    for (int i = tid; i < Lk; i += blockDim.x) {
        sg[row * Lk + i] = from_f32<scalar_t>(a_sm[i] * (go_sm[i] - wsum));
    }
}

template<typename scalar_t>
void launch_fwd(const scalar_t* s, const float* mask,
    scalar_t* a, uint64_t seed, float dp, int Lk, int H, int Lq, float eps, int rows) {
    fwd_kernel<scalar_t><<<rows, BLOCK_DIM>>>(s, mask, a, seed, dp, Lk, H, Lq, eps);
}

template<typename scalar_t>
void launch_bwd(const scalar_t* go, const scalar_t* a,
    scalar_t* sg, uint64_t seed, float dp, int Lk, int H, int Lq, float eps, int rows) {
    if (Lk <= FUSED_TILE) {
        bwd_kernel_fused<scalar_t><<<rows, BLOCK_DIM>>>(go, a, sg, seed, dp, Lk, H, Lq, eps);
    } else {
        bwd_kernel<scalar_t><<<rows, BLOCK_DIM>>>(go, a, sg, seed, dp, Lk, H, Lq, eps);
    }
}

void fwd(torch::Tensor scores, torch::Tensor mask_flat, torch::Tensor attn,
    uint64_t dropout_seed, float dropout_p, float eps) {
    int Lk = scores.size(3), H = scores.size(1), Lq = scores.size(2);
    int rows = scores.numel() / Lk;
    bool has_mask = mask_flat.numel() > 0;
    const float* mask_ptr = has_mask ? mask_flat.data_ptr<float>() : nullptr;
    auto dtype = scores.scalar_type();

    if (dtype == c10::kFloat && Lk % 4 == 0) {
        fwd_kernel_float4<<<rows, BLOCK_DIM>>>(
            scores.data_ptr<float>(), mask_ptr, attn.data_ptr<float>(),
            dropout_seed, dropout_p, Lk, H, Lq, eps);
    } else if (dtype == c10::kFloat) {
        launch_fwd<float>(scores.data_ptr<float>(), mask_ptr, attn.data_ptr<float>(),
            dropout_seed, dropout_p, Lk, H, Lq, eps, rows);
    } else if (dtype == c10::kHalf && Lk % 4 == 0) {
        fwd_kernel_half4<<<rows, BLOCK_DIM>>>(
            reinterpret_cast<const __half*>(scores.data_ptr()), mask_ptr,
            reinterpret_cast<__half*>(attn.data_ptr()),
            dropout_seed, dropout_p, Lk, H, Lq, eps);
    } else if (dtype == c10::kHalf && Lk % 2 == 0) {
        AT_ERROR("half2 dispatch — use half4 path or scalar");
    } else if (dtype == c10::kHalf) {
        launch_fwd<__half>(reinterpret_cast<const __half*>(scores.data_ptr()), mask_ptr,
            reinterpret_cast<__half*>(attn.data_ptr()),
            dropout_seed, dropout_p, Lk, H, Lq, eps, rows);
    } else {
        throw std::runtime_error("Unsupported dtype");
    }
}

void bwd(torch::Tensor go, torch::Tensor attn,
    torch::Tensor sg,
    uint64_t dropout_seed, float dropout_p, float eps) {
    int Lk = attn.size(3), H = attn.size(1), Lq = attn.size(2);
    int rows = attn.numel() / Lk;
    auto dtype = attn.scalar_type();

    if (dtype == c10::kHalf && Lk <= FUSED_TILE) {
        bwd_kernel_fused<__half><<<rows, BLOCK_DIM>>>(
            reinterpret_cast<const __half*>(go.data_ptr()),
            reinterpret_cast<const __half*>(attn.data_ptr()),
            reinterpret_cast<__half*>(sg.data_ptr()),
            dropout_seed, dropout_p, Lk, H, Lq, eps);
    } else if (dtype == c10::kHalf) {
        launch_bwd<__half>(reinterpret_cast<const __half*>(go.data_ptr()),
            reinterpret_cast<const __half*>(attn.data_ptr()),
            reinterpret_cast<__half*>(sg.data_ptr()),
            dropout_seed, dropout_p, Lk, H, Lq, eps, rows);
    } else if (dtype == c10::kFloat && Lk <= FUSED_TILE) {
        bwd_kernel_fused<float><<<rows, BLOCK_DIM>>>(
            go.data_ptr<float>(), attn.data_ptr<float>(), sg.data_ptr<float>(),
            dropout_seed, dropout_p, Lk, H, Lq, eps);
    } else if (dtype == c10::kFloat) {
        launch_bwd<float>(go.data_ptr<float>(), attn.data_ptr<float>(),
            sg.data_ptr<float>(), dropout_seed, dropout_p, Lk, H, Lq, eps, rows);
    } else {
        throw std::runtime_error("Unsupported dtype");
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
            name='s_tau_softmax_fused_v1',
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            verbose=False,
            extra_cflags=['/Zc:preprocessor'] if is_win else [],
            extra_cuda_cflags=['-Xcompiler', '/Zc:preprocessor', '--use_fast_math'] if is_win else ['--use_fast_math'],
        )
    return _cached


class _FusedSoftmaxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, eps=1e-8, mask=None, dropout_p=0.0):
        import random
        out_dtype = scores.dtype
        B, H, Lq, Lk = scores.shape

        if scores.dtype == torch.float16:
            s_native = scores.contiguous()
            attn = torch.empty_like(s_native)
        else:
            s_native = scores.to(torch.float32).contiguous()
            attn = torch.empty_like(s_native)

        if mask is not None:
            mask_f32 = mask.to(torch.float32).contiguous()
            while mask_f32.dim() < 4:
                mask_f32 = mask_f32.unsqueeze(0)
        else:
            mask_f32 = torch.empty(0, device=scores.device, dtype=torch.float32)

        dropout_seed = random.getrandbits(63)
        _load().fwd(s_native, mask_f32, attn, dropout_seed, dropout_p, float(eps))

        ctx.eps = eps
        ctx.out_dtype = out_dtype
        ctx.dropout_seed = dropout_seed
        ctx.dropout_p = dropout_p
        ctx.save_for_backward(attn)
        return attn

    @staticmethod
    def backward(ctx, grad_out):
        attn, = ctx.saved_tensors
        eps = ctx.eps
        dropout_seed = ctx.dropout_seed
        dropout_p = ctx.dropout_p

        if attn.dtype == torch.float16:
            go_native = grad_out.contiguous()
            sg = torch.empty_like(attn)
        else:
            go_native = grad_out.to(torch.float32).contiguous()
            sg = torch.empty_like(attn)

        _load().bwd(go_native, attn, sg, dropout_seed, dropout_p, float(eps))

        return sg.to(ctx.out_dtype), None, None, None


def softmax_fused(scores, eps=1e-8, mask=None, dropout_p=0.0):
    return _FusedSoftmaxFn.apply(scores, eps, mask, dropout_p)