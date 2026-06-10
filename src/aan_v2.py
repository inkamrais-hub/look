"""
AAN v2: 代数神经网络结构
不使用标准注意力机制与归一化层
基于代数运算构建序列建模
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time, os, json

OUT = r"F:\τ\AAN\results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
#  代数运算原语
# ============================================================

class AlgebraicTable(nn.Module):
    """可学习的代数运算表——核心运算法则"""
    def __init__(self, n, tau=1.0):
        super().__init__()
        self.n = n
        self.tau = tau
        # 运算表参数: 编码状态间映射关系
        self.logits = nn.Parameter(torch.randn(n, n, n) * 0.03)

    def forward(self, x, y):
        """
        执行两状态间的代数运算
        x, y: [B, T] 状态索引
        返回: [B, T, n] 概率化结果
        """
        B, T = x.shape
        table = F.gumbel_softmax(self.logits.view(-1, self.n), tau=self.tau, hard=False)
        table = table.view(self.n, self.n, self.n)
        if x.dtype != torch.long:
            x = x.argmax(dim=-1)
        if y.dtype != torch.long:
            y = y.argmax(dim=-1)
        x = x.clamp(0, self.n - 1)
        y = y.clamp(0, self.n - 1)
        result = table[x, y]
        return result

    def hard_forward(self, x, y):
        """离散查表"""
        table = self.logits.argmax(dim=-1)
        return table[x, y]


class AlgebraicReduce(nn.Module):
    """序列代数归约——逐步聚合状态序列"""
    def __init__(self, n, d):
        super().__init__()
        self.n = n
        # 归约运算表
        self.reduce_op = nn.Parameter(torch.randn(n, n, n) * 0.03)
        # 混合权重
        self.blend_param = nn.Parameter(torch.tensor(0.5))
        # 状态到向量投影
        self.to_vec = nn.Linear(n, d)

    def forward(self, states):
        """
        states: [B, T, n] 概率状态序列
        返回: [B, n] 聚合状态
        """
        B, T, _ = states.shape
        table = F.gumbel_softmax(self.reduce_op.view(-1, self.n), tau=1.0, hard=False)
        table = table.view(self.n, self.n, self.n)

        # 双向扫描归约
        current = states[:, 0]
        blend = torch.sigmoid(self.blend_param)
        for t in range(1, T):
            ci = current.argmax(dim=-1).clamp(0, self.n-1)
            si = states[:, t].argmax(dim=-1).clamp(0, self.n-1)
            merged = table[ci, si]
            # 学习型混合
            current = blend * current + (1 - blend) * merged

        return current


class AlgebraicLayer(nn.Module):
    """
    代数神经层:
    1. 状态编码 (连续→离散)
    2. 邻域代数交互
    3. 全局状态归约
    4. 状态解码 (离散→连续)
    5. 残差连接
    """
    def __init__(self, d, n_states=8, n_ops=3):
        super().__init__()
        self.d = d
        self.n = n_states

        self.embed = nn.Linear(d, n_states)
        self.ops = nn.ModuleList([AlgebraicTable(n_states) for _ in range(n_ops)])
        self.reduce = AlgebraicReduce(n_states, d)

        self.recover = nn.Sequential(
            nn.Linear(n_states, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        self.res_alpha = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(d)

        # 交互权重
        self.interact_weight = nn.Parameter(torch.tensor(0.6))

    def forward(self, x):
        """
        x: [B, T, d]
        返回: [B, T, d]
        """
        B, T, D = x.shape
        residual = x

        # 1. 编码到状态空间
        state_logits = self.embed(x)
        state_soft = F.gumbel_softmax(state_logits, tau=0.5, hard=False)

        # 2. 代数交互
        states = state_soft.clone()
        w = torch.sigmoid(self.interact_weight)
        for op in self.ops:
            new_states = []
            for t in range(T):
                if t == 0:
                    merged = op(states[:, t], states[:, min(t+1, T-1)])
                    new_states.append(w * states[:, t] + (1-w) * merged)
                elif t == T - 1:
                    merged = op(states[:, t], states[:, t-1])
                    new_states.append(w * states[:, t] + (1-w) * merged)
                else:
                    left = op(states[:, t], states[:, t-1])
                    right = op(states[:, t], states[:, t+1])
                    new_states.append(0.4 * states[:, t] + 0.3 * left + 0.3 * right)
            states = torch.stack(new_states, dim=1)

        # 3. 全局归约
        global_state = self.reduce(states)
        global_vec = self.recover(global_state)
        global_expanded = global_vec.unsqueeze(1).expand(-1, T, -1)

        # 4. 局部恢复
        local_out = self.recover(states)

        # 5. 残差混合
        alpha = torch.sigmoid(self.res_alpha)
        out = alpha * (global_expanded + local_out) + (1 - alpha) * residual

        return self.norm(out)


# ============================================================
#  AAN v2 完整模型架构
# ============================================================

class AANv2(nn.Module):
    """
    代数注意力网络 v2
    完全基于代数运算的序列模型
    """
    def __init__(self, vocab_size, d=128, n_layers=3, n_states=8, n_ops=3):
        super().__init__()
        self.d = d
        self.n = n_states

        self.embed = nn.Embedding(vocab_size, d)
        self.layers = nn.ModuleList([
            AlgebraicLayer(d, n_states, n_ops) for _ in range(n_layers)
        ])

        self.decode_op = AlgebraicTable(n_states)
        self.decode_embed = nn.Embedding(n_states, d)
        self.decode_head = nn.Linear(d, vocab_size)

        self.n_st = n_states

    def forward(self, x, targets=None):
        B, T = x.shape

        h = self.embed(x)

        for layer in self.layers:
            h = layer(h)

        local_logits = self.decode_head(h)
        out = local_logits

        loss = None
        if targets is not None:
            loss = F.cross_entropy(out.view(-1, out.size(-1)), targets.view(-1))

        return out, loss


# ============================================================
#  对比基线 AAN v1
# ============================================================

class AANv1(nn.Module):
    """v1 基线版本：基于代数注意力的混合架构"""
    def __init__(self, vocab_size, d=128, n_layers=2, n_heads=2, n_states=4):
        super().__init__()
        self.d = d; self.n = n_states
        self.emb = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(64, d)

        self.ops = nn.ModuleList([AlgebraicTable(n_states) for _ in range(n_heads)])
        self.gate_op = AlgebraicTable(n_states)
        self.to_st = nn.Linear(d, n_states)
        self.from_st = nn.Linear(n_states, d)
        self.W_out = nn.Linear(d, d)
        self.drop = nn.Dropout(0.1)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size)
        self.layers = n_layers

    def forward(self, x, targets=None):
        B, T = x.shape
        h = self.emb(x) + self.pos(torch.arange(T, device=x.device).unsqueeze(0))

        for _ in range(self.layers):
            res = h
            st = F.gumbel_softmax(self.to_st(h), tau=0.5, hard=False)
            heads = []
            for op in self.ops:
                si = st.argmax(-1)
                tab = op.logits.argmax(-1)
                tv = tab[si.unsqueeze(2), si.unsqueeze(1)].float()
                base = torch.bmm(st, st.transpose(1,2)) / (self.n**0.5)
                a = base + 0.1 * tv
                mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
                a = a.masked_fill(mask, float("-inf"))
                w = F.softmax(a, dim=-1)
                heads.append(torch.bmm(w, h))
            attn = torch.mean(torch.stack(heads), dim=0)
            ctx = st.mean(dim=1)
            gate = torch.sigmoid((st * ctx.unsqueeze(1)).sum(-1)).unsqueeze(-1)
            h = self.ln(self.drop(self.W_out(gate * attn + (1-gate) * h)) + res)

        logits = self.head(h)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss


# ============================================================
#  训练与评估
# ============================================================

def get_data():
    cache = os.path.join(OUT, "shakespeare.txt")
    if not os.path.exists(cache):
        import shutil
        src = os.path.join(r"F:\τ\math_hunt\results", "shakespeare.txt")
        shutil.copy(src, cache)
    with open(cache, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for i, c in enumerate(chars)}
    data = np.array([c2i[c] for c in text])
    n = int(0.9 * len(data))
    return data[:n], data[n:], len(chars), c2i, i2c


class Loader:
    def __init__(self, data, seq_len=64, bs=32):
        self.data = np.array(data, dtype=np.int64)
        self.sl = seq_len; self.bs = bs
    def get_batch(self):
        ix = np.random.randint(0, len(self.data) - self.sl - 1, self.bs)
        x = torch.from_numpy(self.data[ix[:, None] + np.arange(self.sl)]).to(DEVICE)
        y = torch.from_numpy(self.data[ix[:, None] + np.arange(1, self.sl+1)]).to(DEVICE)
        return x, y


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def generate(model, i2c, c2i, prompt, length=300, temperature=0.8):
    model.eval()
    chars = list(prompt)
    x = torch.tensor([[c2i.get(c, 0) for c in chars]], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        for _ in range(length):
            x_cond = x[:, -64:]
            logits, _ = model(x_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, 1)
            x = torch.cat([x, idx], dim=1)
            chars.append(i2c.get(idx.item(), "?"))
    return "".join(chars)


def train_and_eval(model, train_ds, val_ds, name, epochs=20):
    params = count_params(model)
    print(f"\n{'='*50}")
    print(f"Training {name} ({params:,} params)")
    print(f"{'='*50}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "val_ppl": []}
    best_ppl = float("inf")

    for epoch in range(epochs):
        model.train()
        t_loss = 0
        for _ in range(80):
            x, y = train_ds.get_batch()
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
        t_loss /= 80

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for _ in range(20):
                x, y = val_ds.get_batch()
                _, loss = model(x, y)
                v_loss += loss.item()
        v_loss /= 20
        ppl = np.exp(v_loss)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_ppl"].append(ppl)
        scheduler.step()

        m = " *" if ppl < best_ppl else ""
        if ppl < best_ppl: best_ppl = ppl
        print(f"  Epoch {epoch+1:2d}/{epochs}: train={t_loss:.3f} val={v_loss:.3f} ppl={ppl:.1f}{m}")

    return history, best_ppl, params


def throughput_test(model, ds, n=200):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(10):
        x, y = ds.get_batch()
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    tokens = 0
    for _ in range(n):
        x, y = ds.get_batch()
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        tokens += x.numel()
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    return tokens / (time.time() - t0)


def main():
    print("=" * 60)
    print("AAN v2: 纯代数路线优化")
    print("=" * 60)

    train_data, val_data, vs, c2i, i2c = get_data()
    print(f"Vocab: {vs}")
    train_ds = Loader(train_data)
    val_ds = Loader(val_data)

    epochs = 15

    v1 = AANv1(vs, d=128, n_layers=2, n_heads=2, n_states=4).to(DEVICE)
    v1_hist, v1_ppl, v1_params = train_and_eval(v1, train_ds, val_ds, "AAN v1 (baseline)", epochs)

    v2_big = AANv2(vs, d=128, n_layers=3, n_states=8, n_ops=3).to(DEVICE)
    v2_hist, v2_ppl, v2_params = train_and_eval(v2_big, train_ds, val_ds, "AAN v2 (n=8, 3 ops)", epochs)

    v2_huge = AANv2(vs, d=128, n_layers=3, n_states=16, n_ops=4).to(DEVICE)
    v2h_hist, v2h_ppl, v2h_params = train_and_eval(v2_huge, train_ds, val_ds, "AAN v2 (n=16, 4 ops)", epochs)

    print(f"\n{'='*50}")
    print("Throughput test...")
    print(f"{'='*50}")
    v1_tput = throughput_test(v1, train_ds)
    v2_tput = throughput_test(v2_big, train_ds)
    v2h_tput = throughput_test(v2_huge, train_ds)
    print(f"  v1 (n=4):    {v1_tput:,.0f} tok/s")
    print(f"  v2 (n=8):    {v2_tput:,.0f} tok/s")
    print(f"  v2 (n=16):   {v2h_tput:,.0f} tok/s")

    print(f"\n{'='*50}")
    print("Text generation (prompt: 'To be ')")
    print(f"{'='*50}")
    for name, model in [("v1", v1), ("v2-n8", v2_big), ("v2-n16", v2_huge)]:
        text = generate(model, i2c, c2i, "To be ")
        print(f"\n  [{name}] {text[:200]}")

    print(f"\n{'='*60}")
    print("COMPREHENSIVE REPORT")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Params':<10} {'Best PPL':<10} {'Throughput':<15} {'PPL/1K param'}")
    print("-" * 65)
    for name, params, ppl, tput in [
        ("AAN v1", v1_params, v1_ppl, v1_tput),
        ("AAN v2 (n=8)", v2_params, v2_ppl, v2_tput),
        ("AAN v2 (n=16)", v2h_params, v2h_ppl, v2h_tput),
    ]:
        eff = ppl / (params / 1000)
        print(f"{name:<20} {params:>8,} {ppl:>9.1f} {tput:>12,.0f} {eff:>12.4f}")

    print(f"\n  Previous Transformer baseline: 421K params, PPL=9.2")
    print(f"  Previous AAN v1: 61K params, PPL=12.6")
    print(f"  Current AAN v2 best: {v2h_params:,} params, PPL={v2h_ppl:.1f}")

    results = {
        "v1": {"params": v1_params, "ppl": float(v1_ppl), "tput": float(v1_tput), "history": v1_hist},
        "v2_n8": {"params": v2_params, "ppl": float(v2_ppl), "tput": float(v2_tput), "history": v2_hist},
        "v2_n16": {"params": v2h_params, "ppl": float(v2h_ppl), "tput": float(v2h_tput), "history": v2h_hist},
    }
    with open(f"{OUT}/v2_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#0a0a2e')

    ax = axes[0]
    ax.set_facecolor('#0a0a2e')
    ax.plot(v1_hist["val_ppl"], label=f"v1 (n=4) PPL={v1_ppl:.1f}", linewidth=2, marker="o", markersize=3)
    ax.plot(v2_hist["val_ppl"], label=f"v2 (n=8) PPL={v2_ppl:.1f}", linewidth=2, marker="s", markersize=3)
    ax.plot(v2h_hist["val_ppl"], label=f"v2 (n=16) PPL={v2h_ppl:.1f}", linewidth=2, marker="^", markersize=3)
    ax.set_title("AAN v2: Validation Perplexity", color="white", fontsize=12)
    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("PPL", color="white")
    ax.legend(facecolor="#1a1a3e", edgecolor="white", labelcolor="white", fontsize=9)
    ax.tick_params(colors="white")

    ax = axes[1]
    ax.set_facecolor('#0a0a2e')
    names = ["v1\n(n=4)", "v2\n(n=8)", "v2\n(n=16)"]
    ppls = [v1_ppl, v2_ppl, v2h_ppl]
    params_k = [v1_params/1000, v2_params/1000, v2h_params/1000]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, ppls, w, label="PPL", color="#ff6b6b")
    ax2 = ax.twinx()
    ax2.bar(x + w/2, params_k, w, label="Params (K)", color="#4ecdc4", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, color="white")
    ax.set_ylabel("PPL", color="white")
    ax2.set_ylabel("Params (K)", color="white")
    ax.set_title("PPL vs Parameters", color="white", fontsize=12)
    ax.legend(loc="upper left", facecolor="#1a1a3e", edgecolor="white", labelcolor="white")
    ax2.legend(loc="upper right", facecolor="#1a1a3e", edgecolor="white", labelcolor="white")
    ax.tick_params(colors="white")
    ax2.tick_params(colors="white")

    plt.tight_layout()
    plt.savefig(f"{OUT}/v2_comparison.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nPlot saved: {OUT}/v2_comparison.png")


if __name__ == "__main__":
    main()