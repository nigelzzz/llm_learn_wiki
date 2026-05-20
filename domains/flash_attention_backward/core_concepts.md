# Core Concepts — FlashAttention Backward 完整推導

> 全程目標：在不落地 N×N 矩陣到 HBM 的前提下，從 `dO` 算出 `dQ, dK, dV`。
> 策略：**tile 化 + 重算 P + online 累加**。

---

## 0. 符號與前向回顧

```
S = Q Kᵀ · τ          τ = 1/√d，scaling
P = softmax(S)         沿 key 維 (row-wise)
O = P V

前向只存:  O (N×d),  L (N,)   其中 Lᵢ = logsumexp_j Sᵢⱼ
backward 輸入:  dO (N×d)
```

---

## 1. 先求閉式梯度（不管 tiling）

把計算圖拆成 `O = P V`、`P = softmax(S)`、`S = QKᵀτ`，逐步反傳。

### (a) 從 O 回到 P 和 V

`O = P V`：

```
dV = Pᵀ dO          # (N, d)
dP = dO Vᵀ          # (N, N)
```

### (b) 從 P 穿過 softmax 回到 S

每一 row 套 softmax Jacobian（prerequisites §1）：

```
dSᵢ = Pᵢ ⊙ (dPᵢ − (Pᵢ · dPᵢ))
```

定義關鍵標量（每個 query row 一個）：

```
Dᵢ = Pᵢ · dPᵢ = Σⱼ Pᵢⱼ dPᵢⱼ
```

代入 `dP = dO Vᵀ`，可得一個**極漂亮的簡化**：

```
Dᵢ = Σⱼ Pᵢⱼ (dOᵢ · Vⱼ) = dOᵢ · (Σⱼ Pᵢⱼ Vⱼ) = dOᵢ · Oᵢ
```

也就是

```
D = rowsum(dO ⊙ O)      # (N,)  一個 cheap 的逐 row 內積！
```

**這是 FlashAttention backward 的靈魂。** 它把「需要整個 P 才能算的 `Pᵢ·dPᵢ`」變成「只要 `dO` 和 `O` 就能算的逐 row 內積」。`O` 是前向存好的，`dO` 是輸入——**不需要任何 N×N 矩陣**。

於是：

```
dS = P ⊙ (dP − D)        # D 沿 row 廣播
   = P ⊙ (dO Vᵀ − D)
```

### (c) 從 S 回到 Q 和 K

`S = QKᵀτ`：

```
dQ = τ · dS K            # (N, d)
dK = τ · dSᵀ Q           # (N, d)
```

### 閉式總結（reference 實作就照這個寫）

```
P  = exp(S − L)                 # 用存好的 L 重算，免存 P
dV = Pᵀ dO
dP = dO Vᵀ
D  = rowsum(dO ⊙ O)             # = rowsum(P ⊙ dP)，兩者等價
dS = P ⊙ (dP − D)
dQ = τ · dS K
dK = τ · dSᵀ Q
```

---

## 2. 為什麼可以「重算」而不是「重存」

- 前向沒存 `S`、`P`（N×N，太大）。
- 但存了 `L`（N，很小）和 `O`（N×d，本來就要輸出）。
- backward 進到某個 (query block i, key block j) 的 tile 時，**用該 block 的 q,k 即時算 `Sᵢⱼ = qᵢ·kⱼ·τ`**，再 `Pᵢⱼ = exp(Sᵢⱼ − Lᵢ)`。
- recompute 的 FLOPs 約等於再做一次 forward 的 QKᵀ，但**省下的是 HBM 頻寬**——而 attention 是 memory-bound，所以整體仍大幅加速。

> 口訣：**Compute is cheap, memory traffic is expensive.** 多算一遍 S 換不落地 N×N，划算。

---

## 3. Tiled Backward 演算法

把 Q 切成 `Tr` 個 row block（大小 Br），K/V 切成 `Tc` 個 col block（大小 Bc）。

### 3.1 經典實作的迴圈結構（外層跑 K/V block）

FlashAttention 原論文 backward 採 **outer loop over K/V blocks j，inner loop over Q blocks i**，因為 `dK_j, dV_j` 需要累加所有 i 的貢獻，放外層可常駐 SRAM：

```
預先算:  D = rowsum(dO ⊙ O)    # (N,)，一次算好存著

for j in 0..Tc-1:                       # 外層：每個 K/V block
    載入 Kⱼ, Vⱼ 到 SRAM
    初始化 dKⱼ = 0, dVⱼ = 0
    for i in 0..Tr-1:                   # 內層：每個 Q block
        載入 Qᵢ, dOᵢ, Lᵢ, Dᵢ
        # --- 重算 ---
        Sᵢⱼ = τ · Qᵢ Kⱼᵀ               # (Br, Bc)，在 SRAM
        Pᵢⱼ = exp(Sᵢⱼ − Lᵢ)            # 重算 P，免從 HBM 讀
        # --- 累加 dV, dP ---
        dVⱼ += Pᵢⱼᵀ dOᵢ
        dPᵢⱼ = dOᵢ Vⱼᵀ
        # --- dS 與 dK ---
        dSᵢⱼ = Pᵢⱼ ⊙ (dPᵢⱼ − Dᵢ)       # Dᵢ 沿 row 廣播
        dKⱼ += τ · dSᵢⱼᵀ Qᵢ
        # --- dQ 需要 across-j 累加，用 atomic 或第二輪 ---
        dQᵢ += τ · dSᵢⱼ Kⱼ             # 注意：跨 j 累加
    寫回 dKⱼ, dVⱼ 到 HBM
```

### 3.2 dQ 的累加難題

`dQᵢ = τ Σⱼ dSᵢⱼ Kⱼ`：每個 query block i 的 dQ 是**對所有 key block j 求和**。在「外層 j」結構下，dQ 會被不同 j 反覆更新，做法有二：

1. **atomic add** 到 HBM 的 dQ（簡單，但有原子衝突）。
2. **兩階段 / 換迴圈方向**：另設一個「外層 i、內層 j」的 kernel 專算 dQ（FlashAttention-2 的做法，平行度更好）。

> FlashAttention-2 的關鍵改良之一就是調整迴圈順序與工作分配，讓 dQ/dK/dV 的累加更 GPU-friendly、減少 atomics。

### 3.3 為什麼先算好 D 而不是 inline 算

`Dᵢ = dOᵢ·Oᵢ` 只跟 row i 有關、與 j 無關。**預先一次算好存成 (N,) 向量**，內層迴圈直接讀，避免每個 j 都重算，省 compute 也簡化 kernel。

---

## 4. 記憶體與複雜度

| 項目 | 標準 attention backward | FlashAttention backward |
|---|---|---|
| 額外 HBM（中間矩陣） | O(N²)（存 P 或 S） | O(N)（只存 L, D；O 本來就有） |
| HBM 讀寫量 | O(N²) | O(N²/M·d) 量級，M=SRAM tile |
| FLOPs | ~ baseline | baseline + 一次 recompute（常數倍） |
| 數值穩定 | 靠 softmax max-shift | 靠存好的 L（已含 max） |

結論：**用常數倍的多餘 FLOPs，換掉 O(N²) 的 HBM 流量**，對 memory-bound 的 attention 是淨贏。

---

## 5. 數值穩定性要點

- 重算 `Pᵢⱼ = exp(Sᵢⱼ − Lᵢ)`：因為 `Lᵢ ≥ rowmax(Sᵢ)`，指數內必為 ≤0，**不會 overflow**。這就是前向存 logsumexp（而非單純 sum）的原因。
- `D = rowsum(dO⊙O)` 在 fp16 下建議用 fp32 累加，避免精度流失。
- 對拍時用 fp32，`gradcheck` 才不會因精度誤判。

---

## 6. 一句話總結每個輸出怎麼來

- `dV = Σᵢ Pᵢⱼᵀ dOᵢ`：P 的轉置乘上游 → 哪些 value 被 attend 多就吃多少梯度。
- `dK = τ Σᵢ dSᵢⱼᵀ Qᵢ`：經 softmax 修正後的 score 梯度回灌到 key。
- `dQ = τ Σⱼ dSᵢⱼ Kⱼ`：同上回灌到 query（跨 key block 累加）。
- 靈魂簡化：`D = rowsum(dO⊙O)` 讓 softmax 反傳不需要 N×N。
