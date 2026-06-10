"""
TauAutoCal — one-stop τ* auto-calibration and injection.

Measurement:
  Forward-pass calibration prompts → capture per-head attention scores →
  compute τ*_raw from Cov(s, log(softplus(s)))/Var(log(softplus(s))) statistical moments.

Auto-calibration:
  k = max(1.5, mean(τ_raw) - 0.8)
  τ_eff = clamp(1.0 + (τ_raw - 1.0) / k, 1.2, 3.0)

  Centre τ_eff at ~2.0, preserve per-head ordering, safe for generation.

Injection:
  Drop-in replacement of eager_attention_forward with max-stabilized s^τ.
  Auto-detects Qwen3 / Qwen3.5 / Qwen2 / Llama / GPT-2 / MiniMind.

Usage:
    from stau.autocal import TauAutoCal

    autocal = TauAutoCal(model, tokenizer)
    tau_map = autocal.calibrate()           # {layer_idx: tensor[H]}

    with autocal.apply(model):
        output = model.generate(**inputs)

    autocal.restore()  # or auto via context manager exit

Validated on:  Qwen3-0.6B (k=5.7), Qwen3.5-0.8B (k=4.9),
               Llama-3.2-1B (k=1.0), GPT-2 (k=1.0), MiniMind-3 (k=4.8)
"""
import math
import importlib
import torch
import torch.nn.functional as F
from contextlib import contextmanager

from .tau_star import TauEstimator

EPS = 1e-8
DEFAULT_CAL_TEXTS = [
    "人工智能的发展历程可以追溯到上世纪",
    "机器学习是人工智能的一个重要分支",
    "深度学习通过多层神经网络来",
    "在自然语言处理领域，模型能力",
]

MODEL_FAMILIES = {
    "qwen3":       "transformers.models.qwen3.modeling_qwen3",
    "qwen3_5":     "transformers.models.qwen3_5.modeling_qwen3_5",
    "qwen3_5_text":"transformers.models.qwen3_5.modeling_qwen3_5",
    "qwen2":       "transformers.models.qwen2.modeling_qwen2",
    "gpt2":        "transformers.models.gpt2.modeling_gpt2",
    "llama":       "transformers.models.llama.modeling_llama",
}


def _has_gqa(mod):
    return hasattr(mod, 'repeat_kv')


def _tensor_to(t, device, dtype):
    if t is None:
        return None
    return t.to(device=device, dtype=dtype)


def _tau_star_from_scores_sp(s_val, eps=EPS):
    """[τ*-SP] Regressed τ* formula on softplus statistics (recalibrated 2026-05-15).

    Coefficients from TauEstimator.sp_regressed() (R²=0.79 on Qwen3-0.6B 448 heads).
    """
    est = TauEstimator.sp_regressed()
    return est(s_val)


class TauAutoCal:
    def __init__(self, model, tokenizer, calibration_texts=None, device=None):
        self._model = model
        self._tokenizer = tokenizer
        self._cal_texts = calibration_texts or DEFAULT_CAL_TEXTS

        if device is None:
            device = next(model.parameters()).device
        self._device = device

        mt = getattr(model.config, 'model_type', '')
        if mt not in MODEL_FAMILIES:
            raise ValueError(
                f"Unsupported model type '{mt}'. "
                f"Supported: {list(MODEL_FAMILIES.keys())}")
        self._model_type = mt
        self._model_module = importlib.import_module(MODEL_FAMILIES[mt])

        self._n_layers = model.config.num_hidden_layers
        self._n_heads  = model.config.num_attention_heads
        self._tau_raw  = None
        self._tau_eff  = None
        self._k_value  = None
        self._stats    = None
        self._orig_eager = None

    def measure(self):
        """Phase 1: compute per-head τ*_raw and statistical moments."""
        mm = self._model_module
        orig = mm.eager_attention_forward
        captured = {}

        def patched(module, query, key, value, attention_mask, scaling,
                    dropout=0.0, **kwargs):
            if _has_gqa(mm):
                k = mm.repeat_kv(key, module.num_key_value_groups)
                v = mm.repeat_kv(value, module.num_key_value_groups)
            else:
                k, v = key, value
            if scaling is None:
                scaling = query.shape[-1] ** 0.5
            scores = torch.matmul(query, k.transpose(2, 3)) * scaling
            if attention_mask is not None:
                scores = scores + attention_mask
            li = module.layer_idx
            if li not in captured:
                captured[li] = []
            captured[li].append(scores.detach().cpu().clone())
            w = F.softmax(scores.float(), dim=-1).to(query.dtype)
            out = w @ v
            return out.transpose(1, 2).contiguous(), None

        mm.eager_attention_forward = patched
        for text in self._cal_texts:
            inputs = self._tokenizer(text, return_tensors='pt').to(self._device)
            with torch.no_grad():
                self._model(**inputs)
            torch.cuda.empty_cache()
        mm.eager_attention_forward = orig

        tau_raw = {}
        all_stats = []
        for li in sorted(captured.keys()):
            H = captured[li][0].shape[1]
            taus = torch.zeros(H, dtype=torch.float32)
            for h in range(H):
                s_vals = []
                for scores in captured[li]:
                    s = scores[0, h]
                    valid = s > -1e4
                    if valid.any():
                        s_vals.append(s[valid].float())
                if not s_vals:
                    taus[h] = 1.0
                    all_stats.append({'layer': li, 'head': h,
                                      'tau_raw': 1.0, 'cov_ratio': 0.0,
                                      'skew': 0.0, 'kurt': 0.0})
                    continue
                s_val = torch.cat(s_vals, dim=0)
                tau_val, st = _tau_star_from_scores_sp(s_val)
                taus[h] = tau_val
                all_stats.append({'layer': li, 'head': h,
                                  'tau_raw': round(tau_val, 4),
                                  'cov_ratio': st['cov_ratio'],
                                  'skew': st['skew'],
                                  'kurt': st['kurt']})
            tau_raw[li] = taus

        self._tau_raw = tau_raw
        self._stats = all_stats
        return self

    def calibrate(self, tau_clamp_lo=1.2, tau_clamp_hi=3.0):
        """Phase 2: auto k → τ_eff, return per-layer tau map."""
        if self._tau_raw is None:
            self.measure()

        all_raw = torch.cat([self._tau_raw[li]
                             for li in sorted(self._tau_raw)])
        raw_mean = float(all_raw.mean())
        self._k_value = max(1.5, raw_mean - 0.8)

        tau_eff = {}
        for li in sorted(self._tau_raw):
            tau_eff[li] = (1.0 + (self._tau_raw[li] - 1.0) /
                           self._k_value).clamp(tau_clamp_lo, tau_clamp_hi)

        self._tau_eff = tau_eff
        return tau_eff

    def inject(self, model=None):
        """Phase 3: patch eager_attention_forward with τ_eff."""
        if model is None:
            model = self._model
        if self._tau_eff is None:
            self.calibrate()

        mm = self._model_module
        self._orig_eager = mm.eager_attention_forward

        tau_eff = self._tau_eff
        n_heads = self._n_heads

        def patched(module, query, key, value, attention_mask, scaling,
                    dropout=0.0, **kwargs):
            if _has_gqa(mm):
                k = mm.repeat_kv(key, module.num_key_value_groups)
                v = mm.repeat_kv(value, module.num_key_value_groups)
            else:
                k, v = key, value
            if scaling is None:
                scaling = query.shape[-1] ** 0.5
            scores = torch.matmul(query, k.transpose(2, 3)) * scaling
            if attention_mask is not None:
                scores = scores + attention_mask

            tau = tau_eff.get(module.layer_idx,
                              torch.ones(n_heads, device=scores.device))
            s_stable = scores - scores.max(dim=-1, keepdim=True).values
            sp = F.softplus(s_stable.float()) + EPS
            tau_b = tau.view(1, -1, 1, 1).to(sp.device).to(sp.dtype)
            powered = sp.pow(tau_b)
            w = (powered / (powered.sum(dim=-1, keepdim=True) + EPS))
            w = w.to(query.dtype)
            out = w @ v
            return out.transpose(1, 2).contiguous(), None

        mm.eager_attention_forward = patched
        return self

    def restore(self):
        if self._orig_eager is not None:
            self._model_module.eager_attention_forward = self._orig_eager
            self._orig_eager = None

    @contextmanager
    def apply(self, model=None):
        self.inject(model)
        try:
            yield
        finally:
            self.restore()

    def generate(self, prompts, model=None, tokenizer=None, **gen_kwargs):
        """Convenience: inject, generate, restore, return texts."""
        if tokenizer is None:
            tokenizer = self._tokenizer

        defaults = {'max_new_tokens': 80, 'temperature': 0.7,
                    'top_k': 40, 'do_sample': True}
        defaults.update(gen_kwargs)

        with self.apply(model):
            results = []
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors='pt').to(self._device)
                ilen = inputs['input_ids'].shape[1]
                with torch.no_grad():
                    out = self._model.generate(
                        **inputs,
                        pad_token_id=tokenizer.eos_token_id,
                        **defaults)
                text = tokenizer.decode(out[0, ilen:],
                                        skip_special_tokens=True)
                results.append(text)
                torch.cuda.empty_cache()
        return results

    @property
    def tau_raw(self):
        return self._tau_raw

    @property
    def tau_eff(self):
        return self._tau_eff

    @property
    def k(self):
        return self._k_value

    @property
    def stats(self):
        return self._stats

    def summary(self):
        """Return summary dict for serialization."""
        if self._tau_raw is None:
            return {'status': 'not measured'}
        all_raw = torch.cat([self._tau_raw[li]
                             for li in sorted(self._tau_raw)])
        all_eff = torch.cat([self._tau_eff[li]
                             for li in sorted(self._tau_eff)]) \
                  if self._tau_eff else None
        return {
            'model_type': self._model_type,
            'n_layers': self._n_layers,
            'n_heads': self._n_heads,
            'total_heads': int(all_raw.numel()),
            'tau_raw': {
                'mean': float(all_raw.mean()),
                'std': float(all_raw.std()),
                'min': float(all_raw.min()),
                'max': float(all_raw.max()),
            },
            'auto_k': float(self._k_value) if self._k_value else None,
            'tau_eff': {
                'mean': float(all_eff.mean()),
                'min': float(all_eff.min()),
                'max': float(all_eff.max()),
            } if all_eff is not None else None,
        }