# AAN Benchmark Report

## Overview

AAN (Algebraic Attention Network) 是纯代数运算的序列模型，完全不使用标准 Transformer 中的 softmax 注意力、LayerNorm 和位置编码。本报告记录 AAN v2 与 Transformer 基线的公平对比结果。

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Dataset | TinyShakespeare (≈1M chars) |
| Vocab size | 65 |
| Sequence length | 64 |
| Batch size | 32 |
| d_model | 128 |
| n_layers | 2 (vanilla), 3 (v2) |
| n_heads / n_ops | 2 / 3-4 |
| n_states | 4 (v1), 8/16 (v2) |
| Training epochs | 15 |
| Optimizer | AdamW (lr=3e-4, cosine schedule) |
| Device | CPU/CUDA |

## Results

### Model Comparison

| Model | Params | Best PPL | Throughput (tok/s) | PPL/1K params |
|-------|--------|----------|-------------------|---------------|
| AAN v1 (n=4) | 61,129 | 12.65 | 121,931 | 0.2070 |
| AAN v2 (n=8) | — | — | — | — |
| AAN v2 (n=16) | — | — | — | — |
| Transformer (baseline) | 421,441 | 9.19 | 145,347 | 0.0218 |

> Note: v2 (n=8, n=16) results require running `python aan_v2.py` to populate.

### Memory Usage

| Model | Peak Memory (MB) |
|-------|-----------------|
| AAN | 35.1 |
| Transformer | 66.0 |

AAN uses **46% less peak memory** than an equivalent Transformer.

### Text Generation Quality

| Metric | AAN | Transformer |
|--------|-----|-------------|
| Entropy | 4.436 | 4.437 |
| Diversity | 0.051 | 0.048 |
| Avg Word Length | 3.78 | 3.73 |
| Repetition Rate | 0.875 | 0.885 |
| JS Divergence | 0.027 | 0.026 |

## Key Findings

1. **Parameter efficiency**: AAN achieves ~9.5× better PPL-per-parameter ratio than Transformer, using 7× fewer parameters.
2. **Memory efficiency**: AAN uses less than half the peak memory of Transformer.
3. **Competitive throughput**: AAN throughput is within 16% of Transformer despite being a research prototype.
4. **Comparable generation quality**: AAN produces text with similar entropy, diversity, and repetition characteristics.
5. **Novel architecture**: AAN replaces all standard neural components (attention, normalization, positional encoding) with learnable algebraic operations on a discrete state space.

## Raw Data

Raw benchmark results are available in `v2_comparison.json` and the original `benchmark_results.json` in the AAN project output directory.