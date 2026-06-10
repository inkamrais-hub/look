"""
AAN v2 训练入口
快速启动脚本: 代数神经网络训练与对比评估
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, time, os, json, shutil

OUT = r"F:/τ/AAN/results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# 数据准备
src = os.path.join(r"F:/τ/math_hunt/results", "shakespeare.txt")
dst = os.path.join(OUT, "shakespeare.txt")
if not os.path.exists(dst) and os.path.exists(src): shutil.copy(src, dst)
with open(dst, "r", encoding="utf-8") as f: text = f.read()
chars = sorted(set(text)); c2i = {c:i for i,c in enumerate(chars)}; i2c = {i:c for i,c in enumerate(chars)}
vs = len(chars); data = np.array([c2i[c] for c in text], dtype=np.int64)
n = int(0.9*len(data)); tr, va = data[:n], data[n:]

class L:
    """数据加载器"""
    def __init__(s,d,sl=64,bs=32): s.d=d;s.sl=sl;s.bs=bs
    def batch(s):
        ix = np.random.randint(0,len(s.d)-s.sl-1,s.bs)
        x = torch.from_numpy(s.d[ix[:,None]+np.arange(s.sl)]).to(DEVICE)
        y = torch.from_numpy(s.d[ix[:,None]+np.arange(1,s.sl+1)]).to(DEVICE)
        return x,y

class CT(nn.Module):
    """代数运算表"""
    def __init__(s,n):
        super().__init__()
        s.n=n; s.L=nn.Parameter(torch.randn(n,n,n)*0.1)
    def forward(s,x,y):
        t=F.gumbel_softmax(s.L.view(-1,s.n),tau=1.0,hard=False).view(s.n,s.n,s.n)
        if x.dtype!=torch.long: x=x.argmax(-1)
        if y.dtype!=torch.long: y=y.argmax(-1)
        return t[x.clamp(0,s.n-1),y.clamp(0,s.n-1)]

class AAN1(nn.Module):
    """AAN v1 基线"""
    def __init__(s,vs,d=128,nl=2,nh=2,ns=4):
        super().__init__()
        s.emb=nn.Embedding(vs,d);s.pos=nn.Embedding(64,d)
        s.ops=nn.ModuleList([CT(ns) for _ in range(nh)])
        s.to_st=nn.Linear(d,ns);s.W=nn.Linear(d,d)
        s.drop=nn.Dropout(0.1);s.ln=nn.LayerNorm(d)
        s.head=nn.Linear(d,vs);s.nl=nl;s.ns=ns
    def forward(s,x,y=None):
        B,T=x.shape;h=s.emb(x)+s.pos(torch.arange(T,device=x.device).unsqueeze(0))
        for _ in range(s.nl):
            r=h;st=F.gumbel_softmax(s.to_st(h),tau=0.5,hard=False)
            heads=[]
            for op in s.ops:
                si=st.argmax(-1);tab=op.L.argmax(-1)
                tv=tab[si.unsqueeze(2),si.unsqueeze(1)].float()
                base=torch.bmm(st,st.transpose(1,2))/(s.ns**0.5)
                a=base+0.1*tv
                mask=torch.triu(torch.ones(T,T,device=x.device),diagonal=1).bool()
                a=a.masked_fill(mask,float("-inf"))
                heads.append(torch.bmm(F.softmax(a,dim=-1),h))
            attn=torch.mean(torch.stack(heads),dim=0)
            ctx=st.mean(dim=1)
            gate=torch.sigmoid((st*ctx.unsqueeze(1)).sum(-1)).unsqueeze(-1)
            h=s.ln(s.drop(s.W(gate*attn+(1-gate)*h))+r)
        logits=s.head(h)
        loss=F.cross_entropy(logits.view(-1,vs),y.view(-1)) if y is not None else None
        return logits,loss

class AAN2(nn.Module):
    """AAN v2 代数网络"""
    def __init__(s,vs,d=128,nl=3,ns=8,nops=3):
        super().__init__()
        s.emb=nn.Embedding(vs,d);s.ns=ns
        s.to_st=nn.Linear(d,ns)
        s.ops=nn.ModuleList([CT(ns) for _ in range(nops)])
        s.recover=nn.Sequential(nn.Linear(ns,d),nn.GELU(),nn.Linear(d,d))
        s.head=nn.Linear(d,vs);s.ln=nn.LayerNorm(d)
        s.alpha=nn.Parameter(torch.tensor(0.1));s.nl=nl
    def forward(s,x,y=None):
        B,T=x.shape;h=s.emb(x)
        for _ in range(s.nl):
            r=h;st=F.gumbel_softmax(s.to_st(h),tau=0.5,hard=False)
            for op in s.ops:
                ns_list=[]
                for t in range(T):
                    l=op(st[:,t],st[:,max(t-1,0)])
                    ri=op(st[:,t],st[:,min(t+1,T-1)])
                    ns_list.append(0.33*st[:,t]+0.33*l+0.34*ri)
                st=torch.stack(ns_list,dim=1)
            out=s.recover(st)
            a=torch.sigmoid(s.alpha)
            h=s.ln(a*out+(1-a)*r)
        logits=s.head(h)
        loss=F.cross_entropy(logits.view(-1,vs),y.view(-1)) if y is not None else None
        return logits,loss

class TBlock(nn.Module):
    """Transformer 块（对比基线）"""
    def __init__(s,d,nh):
        super().__init__()
        s.ln1=nn.LayerNorm(d);s.attn=nn.MultiheadAttention(d,nh,batch_first=True)
        s.ln2=nn.LayerNorm(d);s.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d),nn.Dropout(0.1))
        s.drop=nn.Dropout(0.1)
    def forward(s,x):
        T=x.size(1);m=torch.triu(torch.ones(T,T,device=x.device),diagonal=1).bool()
        h=s.ln1(x);h,_=s.attn(h,h,h,attn_mask=m)
        return x+s.drop(h)+s.ff(s.ln2(x))

class Trans(nn.Module):
    """Transformer 基线"""
    def __init__(s,vs,d=128,nl=2,nh=2):
        super().__init__()
        s.emb=nn.Embedding(vs,d);s.pos=nn.Embedding(64,d)
        s.layers=nn.ModuleList([TBlock(d,nh) for _ in range(nl)])
        s.head=nn.Linear(d,vs)
    def forward(s,x,y=None):
        B,T=x.shape;h=s.emb(x)+s.pos(torch.arange(T,device=x.device).unsqueeze(0))
        for l in s.layers:h=l(h)
        logits=s.head(h)
        loss=F.cross_entropy(logits.view(-1,vs),y.view(-1)) if y is not None else None
        return logits,loss

def train(m,ds,ds2,name,ep=15):
    p=sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"\n{'='*50}\nTraining {name} ({p:,} params)\n{'='*50}")
    opt=torch.optim.AdamW(m.parameters(),lr=3e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=ep)
    tl_h=[];vl_h=[];vp_h=[];bp=float("inf")
    for e in range(ep):
        m.train();tl=0
        for _ in range(80):
            x,y=ds.batch();_,loss=m(x,y)
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
            opt.step();tl+=loss.item()
        tl/=80;m.eval();vl=0
        with torch.no_grad():
            for _ in range(20):x,y=ds2.batch();_,loss=m(x,y);vl+=loss.item()
        vl/=20;ppl=np.exp(vl)
        tl_h.append(tl);vl_h.append(vl);vp_h.append(ppl)
        sched.step()
        mk=" *" if ppl<bp else ""
        if ppl<bp:bp=ppl
        print(f"  Epoch {e+1:2d}/{ep}: train={tl:.3f} val={vl:.3f} ppl={ppl:.1f}{mk}")
    return {"tl":tl_h,"vl":vl_h,"vp":vp_h},bp,p

def gen(m,prompt,length=200,temp=0.8):
    m.eval();ch=list(prompt)
    x=torch.tensor([[c2i.get(c,0) for c in ch]],dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        for _ in range(length):
            logits,_=m(x[:,-64:]);logits=logits[:,-1,:]/temp
            idx=torch.multinomial(F.softmax(logits,-1),1)
            x=torch.cat([x,idx],dim=1);ch.append(i2c.get(idx.item(),"?"))
    return "".join(ch)

# ============================================================
#  训练
# ============================================================
tr_l=L(tr);va_l=L(va)
v1=AAN1(vs).to(DEVICE);v1h,v1p,v1pa=train(v1,tr_l,va_l,"AAN v1")
v2=AAN2(vs,d=128,nl=3,ns=8,nops=3).to(DEVICE);v2h,v2p,v2pa=train(v2,tr_l,va_l,"AAN v2 (n=8)")
v3=AAN2(vs,d=128,nl=3,ns=16,nops=4).to(DEVICE);v3h,v3p,v3pa=train(v3,tr_l,va_l,"AAN v2 (n=16)")
th=Trans(vs).to(DEVICE);thh,thp,thpa=train(th,tr_l,va_l,"Transformer")

# 吞吐量
print(f"\n{'='*50}\nThroughput\n{'='*50}")
def tput(m,ds,n=100):
    m.train();o=torch.optim.AdamW(m.parameters(),lr=1e-3)
    for _ in range(5):x,y=ds.batch();_,l=m(x,y);o.zero_grad();l.backward();o.step()
    if DEVICE.type=="cuda":torch.cuda.synchronize()
    t=time.time();tk=0
    for _ in range(n):x,y=ds.batch();_,l=m(x,y);o.zero_grad();l.backward();o.step();tk+=x.numel()
    if DEVICE.type=="cuda":torch.cuda.synchronize()
    return tk/(time.time()-t)
v1t=tput(v1,tr_l);v2t=tput(v2,tr_l);v3t=tput(v3,tr_l);tht=tput(th,tr_l)
print(f"  v1:     {v1t:,.0f} tok/s")
print(f"  v2-n8:  {v2t:,.0f} tok/s")
print(f"  v2-n16: {v3t:,.0f} tok/s")
print(f"  Trans:  {tht:,.0f} tok/s")

print(f"\n{'='*50}\nGeneration\n{'='*50}")
for nm,m in [("v1",v1),("v2-n8",v2),("v2-n16",v3),("Trans",th)]:
    print(f"\n[{nm}] {gen(m,'To be ')}")

print(f"\n{'='*60}\nREPORT\n{'='*60}")
print(f"{'Model':<20} {'Params':<10} {'PPL':<8} {'tok/s':<10} {'PPL/1K'}")
print("-"*55)
for nm,p,pp,tp in [("AAN v1",v1pa,v1p,v1t),("AAN v2 n8",v2pa,v2p,v2t),("AAN v2 n16",v3pa,v3p,v3t),("Transformer",thpa,thp,tht)]:
    print(f"{nm:<20} {p:>8,} {pp:>7.1f} {tp:>9,.0f} {pp/(p/1000):>9.4f}")

res={"v1":{"p":v1pa,"ppl":float(v1p),"t":float(v1t)},"v2n8":{"p":v2pa,"ppl":float(v2p),"t":float(v2t)},"v2n16":{"p":v3pa,"ppl":float(v3p),"t":float(v3t)},"trans":{"p":thpa,"ppl":float(thp),"t":float(tht)}}
with open(os.path.join(OUT,"v2_final.json"),"w") as f:json.dump(res,f,indent=2)
print(f"\nSaved: {OUT}/v2_final.json")