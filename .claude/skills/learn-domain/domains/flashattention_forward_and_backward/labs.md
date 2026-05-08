# Minimal Labs — 動手實作

每個 lab 都附 acceptance criteria，跑過才算過關。

---

## Lab 1：Online Softmax（NumPy）

**目標**：在不看到完整 row 的情況下，分塊計算 softmax，與標準 softmax 數值一致。

```python
import numpy as np

def online_softmax(x, block_size):
    """x: shape (N,)."""
    m = -np.inf
    l = 0.0
    o = 0.0  # 用於累加 sum exp（這裡先算 softmax 自身）
    for i in range(0, len(x), block_size):
        chunk = x[i:i+block_size]
        m_new = max(m, chunk.max())
        l = np.exp(m - m_new) * l + np.exp(chunk - m_new).sum()
        m = m_new
    # 第二 pass：用 m, l normalize
    return np.exp(x - m) / l

# 驗證
np.random.seed(0)
x = np.random.randn(1024) * 5
ref = np.exp(x - x.max()); ref /= ref.sum()
out = online_softmax(x, 64)
assert np.allclose(out, ref, atol=1e-6), "FAIL"
print("PASS")
```

**Acceptance**：與 reference 誤差 < 1e-6；改 block_size = 1, 7, 1024 都通過。

**Bonus**：把第二 pass 也合併進 streaming（即 attention 的 `O = P V` 式累加），這就是 FA 的本體。

---

## Lab 2：Tiled Attention Forward（PyTorch CPU/GPU）

**目標**：實作 FA-2 風格的 forward，外迴圈 Q，內迴圈 K/V。

```python
import torch
import torch.nn.functional as F

def flash_attn_forward(Q, K, V, Br=64, Bc=64):
    """Q, K, V: (N, d). 回傳 O 與 logsumexp L (給 backward)."""
    N, d = Q.shape
    scale = 1.0 / (d ** 0.5)
    O = torch.zeros_like(Q)
    L = torch.zeros(N, device=Q.device)
    for i in range(0, N, Br):
        Qi = Q[i:i+Br]                          # (Br, d)
        m_i = torch.full((Qi.shape[0],), float('-inf'), device=Q.device)
        l_i = torch.zeros(Qi.shape[0], device=Q.device)
        Oi = torch.zeros(Qi.shape[0], d, device=Q.device)
        for j in range(0, N, Bc):
            Kj = K[j:j+Bc]; Vj = V[j:j+Bc]      # (Bc, d)
            S = (Qi @ Kj.T) * scale             # (Br, Bc)
            m_new = torch.maximum(m_i, S.max(dim=-1).values)
            P = torch.exp(S - m_new[:, None])   # (Br, Bc)
            alpha = torch.exp(m_i - m_new)      # (Br,)
            l_i = alpha * l_i + P.sum(dim=-1)
            Oi = alpha[:, None] * Oi + P @ Vj
            m_i = m_new
        O[i:i+Br] = Oi / l_i[:, None]
        L[i:i+Br] = m_i + torch.log(l_i)        # logsumexp 給 backward
    return O, L

# 驗證
torch.manual_seed(0)
N, d = 256, 64
Q, K, V = [torch.randn(N, d) for _ in range(3)]
O_ref = F.scaled_dot_product_attention(Q.unsqueeze(0).unsqueeze(0),
                                       K.unsqueeze(0).unsqueeze(0),
                                       V.unsqueeze(0).unsqueeze(0)).squeeze()
O, L = flash_attn_forward(Q, K, V, Br=32, Bc=32)
assert torch.allclose(O, O_ref, atol=1e-5), (O - O_ref).abs().max()
print("PASS")
```

**Acceptance**：N=256, d=64 與 PyTorch SDPA 誤差 < 1e-5；改不同 Br, Bc 仍正確。

---

## Lab 3：Memory 曲線

把 Lab 2 包成 function，量測（用 `torch.cuda.max_memory_allocated()`）：
- 不同 N ∈ {512, 1024, 2048, 4096}
- 不同 (Br, Bc) ∈ {(32,32), (64,64), (128,128)}

**Acceptance**：畫出曲線，驗證 FA 的 peak memory 與 N 大致線性（標準 attention 是 N²）。

---

## Lab 4：Backward + gradcheck

把 Lab 2 包成 `torch.autograd.Function`，自寫 backward：

```python
class FlashAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, Br=32, Bc=32):
        O, L = flash_attn_forward(Q, K, V, Br, Bc)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.Br, ctx.Bc = Br, Bc
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        # TODO: 實作 tiled backward
        # 先算 D = rowsum(dO * O)，shape (N,)
        # 外迴圈 j (K, V)，內迴圈 i (Q)
        # 重算 P_ij = exp(S_ij - L_i)
        # dV_j += P_ij^T @ dO_i
        # dP_ij = dO_i @ V_j^T
        # dS_ij = P_ij * (dP_ij - D_i)
        # dQ_i += dS_ij @ K_j * scale     (atomic 累加)
        # dK_j += dS_ij^T @ Q_i * scale
        ...
        return dQ, dK, dV, None, None
```

**Acceptance**：
```python
torch.autograd.gradcheck(FlashAttn.apply, (Q.double().requires_grad_(),
                                            K.double().requires_grad_(),
                                            V.double().requires_grad_()))
```
回傳 True。

---

## Lab 5：Triton 實作 + benchmark

跑通 Triton 官方教學，並 benchmark：
```
N=2048, d=64, fp16
naive PyTorch:  ?? ms
SDPA (math):    ?? ms
SDPA (flash):   ?? ms
Your Triton:    ?? ms
```

**Acceptance**：你的 Triton 版本與 SDPA(flash) 同數量級（≤ 2x 慢以內就算成功，因為官方有許多微調）。

---

## Lab 6（選修）：Causal Mask

修改 Lab 2，加入 `is_causal=True` 路徑：
- 內迴圈中當 `j > i` 整個 block 跳過
- 當 `j == i` 對角區塊套用 lower-triangular mask

**Acceptance**：與 PyTorch `is_causal=True` 一致。

---

## Lab 7（選修，進階）：Logsumexp 數值穩定性測試

製造極端輸入：`Q*K^T` 範圍 [-100, 100]（fp16 直接溢位的等級），檢驗：
- naive softmax → 出現 NaN
- online softmax → 仍然正確

理解這就是 FA 內部必須先減 max 的物理原因。
