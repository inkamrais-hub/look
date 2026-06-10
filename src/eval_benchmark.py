"""
AAN vs Transformer: 公平对比评测
测试: 困惑度、吞吐量、参数效率、生成质量
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time, os, json, urllib.request
from collections import Counter

OUT = r"F:\τ\AAN\results"
os.makedirs(OUT, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ============================================================
#  数据
# ============================================================
def get_data():
    cache = os.path.join(OUT, "shakespeare.txt")
    if not os.path.exists(cache):
        cache2 = os.path.join(r"F:\τ\math_hunt\results", "shakespeare.txt")
        if os.path.exists(cache2):
            import shutil; shutil.copy(cache2, cache)
        else:
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            urllib.request.urlretrieve(url, cache)
    with open(cache, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for i, c in enumerate(chars)}
    data = np.array([c2i[c] for c in text])
    n = int(0.9 * len(data))
    return data[:n], data[n:], vocab_size, c2i, i2c


class Loader:
    def __init__(self, data, seq_len, batch_size):
        self.data = data; self.seq_len = seq_len; self.bs = batch_size
    def get_batch(self):
        ix = np.random.randint(0, len(self.data) - self.seq_len - 1, self.bs)
        x = torch.tensor([self.data[i:i+self.seq_len] for i in ix], dtype=torch.long)
        y = torch.tensor([self.data[i+1:i+self.seq_len+1] for i in ix], dtype=torch.long)
        return x.to(DEVICE), y.to(DEVICE)


# ============================================================
#  AAN模型
# ============================================================
class AlgebraicCayley(nn.Module):
    def __init__(self, n, tau=1.0):
        super().__init__()
        self.n = n; self.tau = tau
        self.logits = nn.Parameter(torch.randn(n, n, n) * 0.1)
    def forward(self):
        return F.gumbel_softmax(self.logits.view(-1, self.n), tau=self.tau, hard=False).view(self.n, self.n, self.n)
    def hard(self):
        return self.logits.argmax(dim=-1)


class AANLayer(nn.Module):
    def __init__(self, n_st, d, n_heads=2, dropout=0.1):
        super().__init__()
        self.n = n_st; self.d = d
        self.attn_ops = nn.ModuleList([AlgebraicCayley(n_st) for _ in range(n_heads)])
        self.gate_op = AlgebraicCayley(n_st)
        self.to_st = nn.Linear(d, n_st)
        self.from_st = nn.Linear(n_st, d)
        self.W_out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        B, T, D = x.shape
        res = x
        st_prob = F.gumbel_softmax(self.to_st(x), tau=0.5, hard=False)

        heads = []
        for h in range(len(self.attn_ops)):
            tab = self.attn_ops[h].forward()
            Q = st_prob
            base = torch.bmm(Q, Q.transpose(1,2)) / (self.n**0.5)
            si = st_prob.argmax(-1)
            tab_hard = self.attn_ops[h].hard()
            tv = tab_hard[si.unsqueeze(2), si.unsqueeze(1)].float()
            a = base + 0.1 * tv
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            a = a.masked_fill(mask, float("-inf"))
            w = F.softmax(a, dim=-1)
            heads.append(torch.bmm(w, x))
        attn = torch.mean(torch.stack(heads), dim=0)

        ctx = st_prob.mean(dim=1)
        gate = torch.sigmoid((st_prob * ctx.unsqueeze(1)).sum(-1)).unsqueeze(-1)
        out = gate * attn + (1 - gate) * x
        out = self.ln(self.drop(self.W_out(out)) + res)
        return out


class AAN(nn.Module):
    def __init__(self, vs, d=128, L=2, nh=2, ns=4, sl=64):
        super().__init__()
        self.emb = nn.Embedding(vs, d)
        self.pos = nn.Embedding(sl, d)
        self.layers = nn.ModuleList([AANLayer(ns, d, nh) for _ in range(L)])
        self.head = nn.Linear(d, vs)

    def forward(self, x, targets=None):
        B, T = x.shape
        h = self.emb(x) + self.pos(torch.arange(T, device=x.device).unsqueeze(0))
        for l in self.layers: h = l(h)
        logits = self.head(h)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss


# ============================================================
#  Transformer模型
# ============================================================
class TBlock(nn.Module):
    def __init__(self, d, nh, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nh, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        T = x.size(1)
        m = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=m)
        return x + self.drop(h) + self.ff(self.ln2(x))


class TransformerLM(nn.Module):
    def __init__(self, vs, d=128, L=2, nh=2, sl=64):
        super().__init__()
        self.emb = nn.Embedding(vs, d)
        self.pos = nn.Embedding(sl, d)
        self.layers = nn.ModuleList([TBlock(d, nh) for _ in range(L)])
        self.head = nn.Linear(d, vs)

    def forward(self, x, targets=None):
        B, T = x.shape
        h = self.emb(x) + self.pos(torch.arange(T, device=x.device).unsqueeze(0))
        for l in self.layers: h = l(h)
        logits = self.head(h)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss


# ============================================================
#  评测函数
# ============================================================
def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def measure_throughput(model, loader, n_batches=200):
    """测量吞吐量 (tokens/sec)"""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # warmup
    for _ in range(10):
        x, y = loader.get_batch()
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 测量
    torch.cuda.synchronize() if DEVICE.type == "cuda" else None
    t0 = time.time()
    total_tokens = 0
    for _ in range(n_batches):
        x, y = loader.get_batch()
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_tokens += x.numel()
    torch.cuda.synchronize() if DEVICE.type == "cuda" else None
    elapsed = time.time() - t0
    return total_tokens / elapsed, elapsed


def train_and_eval(model, train_loader, val_loader, vocab_size, name, epochs=15):
    """训练并评测"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "val_ppl": []}
    best_ppl = float("inf")

    print(f"\n{'='*50}")
    print(f"Training {name} ({count_params(model):,} params)")
    print(f"{'='*50}")

    for epoch in range(epochs):
        model.train()
        t_loss = 0
        for _ in range(80):
            x, y = train_loader.get_batch()
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
        t_loss /= 80

        model.eval()
        with torch.no_grad():
            v_loss = 0
            for _ in range(20):
                x, y = val_loader.get_batch()
                _, loss = model(x, y)
                v_loss += loss.item()
            v_loss /= 20

        ppl = np.exp(v_loss)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_ppl"].append(ppl)
        scheduler.step()

        marker = " *" if ppl < best_ppl else ""
        if ppl < best_ppl: best_ppl = ppl
        print(f"  Epoch {epoch+1:2d}/{epochs}: train={t_loss:.3f} val={v_loss:.3f} ppl={ppl:.1f}{marker}")

    return history, best_ppl


def generate(model, i2c, c2i, prompt, length=300, temperature=0.8):
    """生成文本"""
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


def compute_metrics(generated, reference):
    """计算生成质量指标"""
    # 1. 字符级困惑度
    chars = list(generated)
    counter = Counter(chars)
    total = len(chars)
    probs = [c/total for c in counter.values()]
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)

    # 2. 词汇多样性
    unique_chars = len(set(chars))
    diversity = unique_chars / total

    # 3. 平均词长
    words = generated.split()
    avg_word_len = np.mean([len(w) for w in words]) if words else 0

    # 4. 重复率 (n-gram重复)
    bigrams = [chars[i]+chars[i+1] for i in range(len(chars)-1)]
    bigram_counter = Counter(bigrams)
    repeated = sum(v for v in bigram_counter.values() if v > 1)
    repetition_rate = repeated / len(bigrams) if bigrams else 0

    # 5. 与参考文本的字符分布相似度
    ref_counter = Counter(list(reference))
    ref_total = len(reference)
    ref_dist = np.zeros(128)
    gen_dist = np.zeros(128)
    for c, v in counter.items():
        if ord(c) < 128: gen_dist[ord(c)] = v / total
    for c, v in ref_counter.items():
        if ord(c) < 128: ref_dist[ord(c)] = v / ref_total
    # Jensen-Shannon散度
    m = 0.5 * (gen_dist + ref_dist)
    js_div = 0.5 * np.sum(gen_dist * np.log(gen_dist / (m + 1e-10) + 1e-10)) + \
             0.5 * np.sum(ref_dist * np.log(ref_dist / (m + 1e-10) + 1e-10))

    return {
        "entropy": float(entropy),
        "diversity": float(diversity),
        "avg_word_len": float(avg_word_len),
        "repetition_rate": float(repetition_rate),
        "js_divergence": float(max(0, js_div)),
    }


# ============================================================
#  主评测
# ============================================================
def main():
    print("=" * 60)
    print("AAN vs Transformer: 公平对比评测")
    print("=" * 60)

    # 数据
    train_data, val_data, vocab_size, c2i, i2c = get_data()
    print(f"Vocab: {vocab_size}, Train: {len(train_data):,}, Val: {len(val_data):,}")

    train_loader = Loader(train_data, seq_len=64, batch_size=32)
    val_loader = Loader(val_data, seq_len=64, batch_size=32)

    # 模型 (公平对比: 相同d_model, n_layers, n_heads)
    d_model = 128
    n_layers = 2
    n_heads = 2
    n_states = 4
    epochs = 15

    aan = AAN(vocab_size, d_model, n_layers, n_heads, n_states).to(DEVICE)
    trans = TransformerLM(vocab_size, d_model, n_layers, n_heads).to(DEVICE)

    aan_params = count_params(aan)
    trans_params = count_params(trans)
    print(f"AAN params: {aan_params:,}")
    print(f"Transformer params: {trans_params:,}")

    # 1. 训练
    aan_hist, aan_best_ppl = train_and_eval(aan, train_loader, val_loader, vocab_size, "AAN", epochs)
    trans_hist, trans_best_ppl = train_and_eval(trans, train_loader, val_loader, vocab_size, "Transformer", epochs)

    # 2. 吞吐量
    print(f"\n{'='*50}")
    print("Measuring throughput...")
    print(f"{'='*50}")

    aan_tput, aan_time = measure_throughput(aan, train_loader, 200)
    trans_tput, trans_time = measure_throughput(trans, train_loader, 200)
    print(f"  AAN: {aan_tput:,.0f} tokens/sec ({aan_time:.1f}s for 200 batches)")
    print(f"  Transformer: {trans_tput:,.0f} tokens/sec ({trans_time:.1f}s for 200 batches)")

    # 3. 显存
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        x, y = train_loader.get_batch()
        _, loss = aan(x, y)
        loss.backward()
        aan_mem = torch.cuda.max_memory_allocated() / 1024**2
        torch.cuda.reset_peak_memory_stats()
        x, y = train_loader.get_batch()
        _, loss = trans(x, y)
        loss.backward()
        trans_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  AAN peak memory: {aan_mem:.1f} MB")
        print(f"  Transformer peak memory: {trans_mem:.1f} MB")
    else:
        aan_mem = 0; trans_mem = 0

    # 4. 生成质量
    print(f"\n{'='*50}")
    print("Generating text...")
    print(f"{'='*50}")

    prompts = ["To be ", "ROMEO: ", "What ", "The "]
    for prompt in prompts:
        aan_text = generate(aan, i2c, c2i, prompt)
        trans_text = generate(trans, i2c, c2i, prompt)
        print(f"\n  Prompt: '{prompt}'")
        print(f"  AAN:         {aan_text[:200]}")
        print(f"  Transformer: {trans_text[:200]}")

    # 5. 质量指标
    print(f"\n{'='*50}")
    print("Computing quality metrics...")
    print(f"{'='*50}")

    # 用训练集的一部分作为参考
    ref_text = "".join(i2c[c] for c in val_data[:5000])

    aan_gen = generate(aan, i2c, c2i, "To be ", length=1000)
    trans_gen = generate(trans, i2c, c2i, "To be ", length=1000)

    aan_metrics = compute_metrics(aan_gen, ref_text)
    trans_metrics = compute_metrics(trans_gen, ref_text)

    print(f"\n  {'Metric':<25} {'AAN':<15} {'Transformer':<15}")
    print("  " + "-" * 55)
    for k in aan_metrics:
        v1 = aan_metrics[k]
        v2 = trans_metrics[k]
        print(f"  {k:<25} {v1:<15.4f} {v2:<15.4f}")

    # 6. 综合报告
    print(f"\n{'='*60}")
    print("COMPREHENSIVE REPORT")
    print(f"{'='*60}")
    print()
    print(f"{'Metric':<30} {'AAN':<18} {'Transformer':<18} {'Ratio'}")
    print("-" * 80)
    print(f"{'Parameters':<30} {aan_params:>12,}   {trans_params:>12,}   {trans_params/aan_params:.1f}x")
    print(f"{'Best PPL':<30} {aan_best_ppl:>12.1f}   {trans_best_ppl:>12.1f}   {trans_best_ppl/aan_best_ppl:.2f}x")
    print(f"{'Throughput (tok/s)':<30} {aan_tput:>12,.0f}   {trans_tput:>12,.0f}   {trans_tput/aan_tput:.2f}x")
    if aan_mem > 0:
        print(f"{'Peak Memory (MB)':<30} {aan_mem:>12.1f}   {trans_mem:>12.1f}   {trans_mem/aan_mem:.2f}x")
    print(f"{'Text Entropy':<30} {aan_metrics['entropy']:>12.3f}   {trans_metrics['entropy']:>12.3f}   {trans_metrics['entropy']/aan_metrics['entropy']:.2f}x")
    print(f"{'Diversity':<30} {aan_metrics['diversity']:>12.4f}   {trans_metrics['diversity']:>12.4f}   {trans_metrics['diversity']/aan_metrics['diversity']:.2f}x")
    print(f"{'Repetition Rate':<30} {aan_metrics['repetition_rate']:>12.4f}   {trans_metrics['repetition_rate']:>12.4f}   {trans_metrics['repetition_rate']/(aan_metrics['repetition_rate']+1e-10):.2f}x")
    print(f"{'JS Divergence (lower=better)':<30} {aan_metrics['js_divergence']:>12.4f}   {trans_metrics['js_divergence']:>12.4f}   {trans_metrics['js_divergence']/(aan_metrics['js_divergence']+1e-10):.2f}x")

    # 效率比
    ppl_ratio = trans_best_ppl / aan_best_ppl
    param_ratio = trans_params / aan_params
    efficiency = ppl_ratio / param_ratio

    print()
    print(f"Parameter Efficiency (PPL per param):")
    print(f"  AAN:          {aan_best_ppl / aan_params * 1000:.4f} PPL per 1K params")
    print(f"  Transformer:  {trans_best_ppl / trans_params * 1000:.4f} PPL per 1K params")
    print(f"  Efficiency ratio: {efficiency:.2f}x")
    if efficiency > 1:
        print(f"  -> AAN is MORE parameter-efficient!")
    else:
        print(f"  -> Transformer is MORE parameter-efficient")

    # 保存
    results = {
        "aan_params": aan_params, "trans_params": trans_params,
        "aan_best_ppl": float(aan_best_ppl), "trans_best_ppl": float(trans_best_ppl),
        "aan_throughput": float(aan_tput), "trans_throughput": float(trans_tput),
        "aan_memory": float(aan_mem), "trans_memory": float(trans_mem),
        "aan_metrics": aan_metrics, "trans_metrics": trans_metrics,
        "param_efficiency": float(efficiency),
        "aan_history": aan_hist, "trans_history": trans_hist,
    }
    with open(f"{OUT}/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {OUT}/benchmark_results.json")

    # 画对比图
    plot_comparison(aan_hist, trans_hist)


def plot_comparison(aan_h, trans_h):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#0a0a2e')

    ax = axes[0]
    ax.set_facecolor('#0a0a2e')
    ax.plot(aan_h["train_loss"], label="AAN train", color="#ff6b6b", linewidth=2)
    ax.plot(aan_h["val_loss"], label="AAN val", color="#ff6b6b", linestyle="--", linewidth=2)
    ax.plot(trans_h["train_loss"], label="Trans train", color="#4ecdc4", linewidth=2)
    ax.plot(trans_h["val_loss"], label="Trans val", color="#4ecdc4", linestyle="--", linewidth=2)
    ax.set_title("Loss", color="white", fontsize=12)
    ax.legend(facecolor="#1a1a3e", edgecolor="white", labelcolor="white", fontsize=8)
    ax.tick_params(colors="white")

    ax = axes[1]
    ax.set_facecolor('#0a0a2e')
    ax.plot(aan_h["val_ppl"], label="AAN", color="#ff6b6b", linewidth=2, marker="o", markersize=4)
    ax.plot(trans_h["val_ppl"], label="Transformer", color="#4ecdc4", linewidth=2, marker="s", markersize=4)
    ax.set_title("Validation Perplexity", color="white", fontsize=12)
    ax.legend(facecolor="#1a1a3e", edgecolor="white", labelcolor="white", fontsize=8)
    ax.tick_params(colors="white")

    plt.tight_layout()
    plt.savefig(f"{OUT}/benchmark_plot.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Plot saved: {OUT}/benchmark_plot.png")


if __name__ == "__main__":
    main()