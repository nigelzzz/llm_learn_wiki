# Prerequisites（先備知識）

backward 的難度不在 coding，在**數學**。下面每一項都要能「手推」，不是「看得懂」。

## 1. Softmax 的 Jacobian（最重要）

設 `p = softmax(s)`，其中 `s, p ∈ ℝⁿ`。則

```
∂pᵢ/∂sⱼ = pᵢ (δᵢⱼ − pⱼ)
```

寫成矩陣（單列）：

```
J = diag(p) − p pᵀ
```

給定上游梯度 `dp`，則對 `s` 的梯度：

```
ds = J·dp = p ⊙ dp − p · (pᵀ dp)
        = p ⊙ (dp − (pᵀ dp))      # 標量 pᵀdp 廣播相減
```

**這個式子是整個 FlashAttention backward 的種子。** 後面的 `D = pᵀ dp` 就是這裡的 `pᵀ dp`。

> ✅ 自測：能不能在 30 秒內從 `pᵢ=e^{sᵢ}/Σe^{sₖ}` 推出 `∂pᵢ/∂sⱼ`？不行就先補。

## 2. 矩陣微分基本款

要會這三個（用到處都是）：

| 前向 | 反向（給 dY） |
|---|---|
| `Y = A B` | `dA = dY Bᵀ`,  `dB = Aᵀ dY` |
| `Y = A Bᵀ` | `dA = dY B`,  `dB = dYᵀ A` |
| `Y = softmax(X) (沿 row)` | 見上節，逐 row 套 Jacobian |

## 3. Attention 前向的精確定義

```
S = Q Kᵀ / √d        # (N, N)，N 個 query × N 個 key
P = softmax(S)        # 沿著 key 維（每個 row 加總為 1）
O = P V               # (N, d)
```

縮放因子 `1/√d` 記得帶著，backward 會出現在 dS → dQ/dK 那一步。

## 4. Online Softmax（前向的基礎，backward 也借用觀念）

逐 block 計算 softmax 而不需要看到整列，靠維護 running max `m` 與 running sum `ℓ`：

```
mᵢ = max(mᵢ₋₁, rowmax(Sᵢ))
ℓᵢ = e^{mᵢ₋₁−mᵢ} ℓᵢ₋₁ + rowsum(e^{Sᵢ−mᵢ})
```

前向結束時，把 `L = m + log(ℓ)`（即 logsumexp）存起來——**backward 重算 P 全靠它**。

## 5. 為什麼存 L 就能重算 P？

因為

```
Pᵢⱼ = e^{Sᵢⱼ − Lᵢ}
```

其中 `Lᵢ = logsumexp_j(Sᵢⱼ)`。只要 backward 時重算 `Sᵢⱼ = qᵢ·kⱼ/√d`，再減去存好的 `Lᵢ` 取 exp，就**精確還原** `P`，完全不需要前向存 N×N 的 P。這就是「recompute 而非 re-store」的數學基礎。

## 6. 工具面（labs 需要）

- PyTorch：`torch.autograd.grad`、`gradcheck`、手寫 `autograd.Function`。
- （進階／選讀）Triton 或 CUDA 的 tiling 心智模型：SRAM、block、`tl.load/tl.store`。
- 數值：float32 對拍、`atol/rtol` 的設定直覺。

## 自我檢核清單

- [ ] 能手推 softmax Jacobian
- [ ] 能手推 `Y=AB` 的 dA, dB
- [ ] 能寫出 S, P, O 三行前向
- [ ] 能說明 L 的定義與「重算 P」的關係
- [ ] 會用 torch.autograd.gradcheck
