# Minimal Labs — FlashAttention Backward

目標：用最小可跑的程式碼，把 §core_concepts 的數學變成「能對拍通過」的實作。建議用 fp32、CPU 即可。

---

## Lab 1：閉式 backward + 對拍（必做）

驗證 §1 閉式解正確。先用 autograd 拿 ground truth，再用我們的公式手算。

```python
import torch

def attn_forward(Q, K, V):
    d = Q.shape[-1]
    tau = 1.0 / (d ** 0.5)
    S = (Q @ K.transpose(-1, -2)) * tau          # (N, N)
    P = torch.softmax(S, dim=-1)
    O = P @ V
    L = torch.logsumexp(S, dim=-1)               # (N,) 前向只需存 O, L
    return O, L, (S, P)

def attn_backward_closed(Q, K, V, O, L, dO):
    d = Q.shape[-1]
    tau = 1.0 / (d ** 0.5)
    S = (Q @ K.transpose(-1, -2)) * tau
    P = torch.exp(S - L[..., None])              # 重算 P，不用前向存的 P
    dV = P.transpose(-1, -2) @ dO
    dP = dO @ V.transpose(-1, -2)
    D  = (dO * O).sum(-1, keepdim=True)          # 靈魂：rowsum(dO⊙O)
    dS = P * (dP - D)
    dQ = tau * (dS @ K)
    dK = tau * (dS.transpose(-1, -2) @ Q)
    return dQ, dK, dV

# --- 對拍 ---
torch.manual_seed(0)
N, d = 16, 8
Q = torch.randn(N, d, dtype=torch.float64, requires_grad=True)
K = torch.randn(N, d, dtype=torch.float64, requires_grad=True)
V = torch.randn(N, d, dtype=torch.float64, requires_grad=True)

O, L, _ = attn_forward(Q, K, V)
dO = torch.randn_like(O)
O.backward(dO)

dQ, dK, dV = attn_backward_closed(Q.detach(), K.detach(), V.detach(),
                                  O.detach(), L.detach(), dO)
print("dQ err", (dQ - Q.grad).abs().max().item())
print("dK err", (dK - K.grad).abs().max().item())
print("dV err", (dV - V.grad).abs().max().item())
# 三個都應 < 1e-9 (fp64)
```

**任務**：跑通並讓三個誤差都極小。改掉 `D` 那行（例如忘了 keepdim 或忘了乘 P）看看哪個梯度先壞。

---

## Lab 2：驗證「存 L 即可重算 P」

```python
O, L, (S, P_true) = attn_forward(Q, K, V)
P_recompute = torch.exp(S - L[..., None])
print("P recompute err", (P_recompute - P_true).abs().max().item())  # ~0
```

**任務**：把 `L` 換成 `S.exp().sum(-1).log()`（不減 max）並在大數值 S 下測試，觀察 overflow，理解為何用 logsumexp。

---

## Lab 3：Tiled backward（Python 迴圈版）

把 reference 拆成 block，模擬 kernel 的 SRAM 迴圈，驗證與 Lab 1 一致。

```python
def attn_backward_tiled(Q, K, V, O, L, dO, Br=4, Bc=4):
    N, d = Q.shape
    tau = 1.0 / (d ** 0.5)
    D = (dO * O).sum(-1)                          # (N,) 預先算好
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    for j in range(0, N, Bc):                     # 外層: K/V block
        Kj, Vj = K[j:j+Bc], V[j:j+Bc]
        dKj = torch.zeros_like(Kj)
        dVj = torch.zeros_like(Vj)
        for i in range(0, N, Br):                 # 內層: Q block
            Qi, dOi, Li, Di = Q[i:i+Br], dO[i:i+Br], L[i:i+Br], D[i:i+Br]
            Sij = tau * (Qi @ Kj.T)               # 重算 S block
            Pij = torch.exp(Sij - Li[:, None])    # 重算 P block
            dVj += Pij.T @ dOi
            dPij = dOi @ Vj.T
            dSij = Pij * (dPij - Di[:, None])
            dKj += tau * (dSij.T @ Qi)
            dQ[i:i+Br] += tau * (dSij @ Kj)       # dQ 跨 j 累加
        dK[j:j+Bc] = dKj
        dV[j:j+Bc] = dVj
    return dQ, dK, dV

dQt, dKt, dVt = attn_backward_tiled(Q.detach(), K.detach(), V.detach(),
                                    O.detach(), L.detach(), dO)
print("tiled dQ err", (dQt - Q.grad).abs().max().item())
print("tiled dK err", (dKt - K.grad).abs().max().item())
print("tiled dV err", (dVt - V.grad).abs().max().item())
```

**任務**：
1. 跑通三誤差極小。
2. 把 `Br, Bc` 改成各種值（含不整除 N 的情況，自己補 padding 或調 N）。
3. **故意把 `dQ[i:i+Br] +=` 改成 `=`**，看誤差爆掉，體會「dQ 必須 across-j 累加」。

---

## Lab 4（選讀）：包成 autograd.Function

```python
class FlashAttnRef(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        O, L, _ = attn_forward(Q, K, V)
        ctx.save_for_backward(Q, K, V, O, L)      # 注意：沒存 P！
        return O
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        return attn_backward_tiled(Q, K, V, O, L, dO)

# 用 gradcheck 驗證
Qg = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
Kg = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
Vg = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
print(torch.autograd.gradcheck(FlashAttnRef.apply, (Qg, Kg, Vg)))  # True
```

---

## Lab 5（進階選讀）：Triton kernel
- 參考官方 FlashAttention / Triton tutorial 的 `_bwd_kernel`。
- 重點觀察：(1) 迴圈順序 (2) dQ 的 atomic / 分離 kernel (3) `tl.load` 的 block 載入。
- 與 Lab 3 的 PyTorch tiled 對拍。

---

### 對拍門檻建議
| 精度 | 合格 max abs err |
|---|---|
| fp64 | < 1e-9 |
| fp32 | < 1e-4 |
| fp16/bf16 | < 1e-2（且 D 用 fp32 累加） |
