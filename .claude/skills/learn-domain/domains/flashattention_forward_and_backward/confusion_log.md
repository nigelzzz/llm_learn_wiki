# Common Misunderstandings — FlashAttention 易混淆點

> 這份檔案是「會被踩到的雷」清單，學習過程中遇到新的混淆，**直接 append 在文末**並標日期。

---

## M1. 「FlashAttention 減少了 FLOPs」 ❌

FLOPs 沒變。Forward 仍是 ~4 N² d。FA 改的是 **HBM 讀寫量**，不是計算量。

**正確說法**：FA 把 attention 從 memory-bound 變成更接近 compute-bound（因為 IO 大幅減少，現在主要時間花在 matmul 本身）。

---

## M2. 「Online softmax 是近似」 ❌

Online softmax 在無限精度下與標準 softmax **位元級等價**。它只是用「先看到的 max 來累積，後來看到更大時用 e^(m_old-m_new) 修正」。

**證明**：把遞推展開，`ℓ_final = Σ e^(x_i - m_final)`，正是標準分母。

---

## M3. 把 `m` 和 `ℓ` 搞混

| 符號 | 意義 |
|------|------|
| `m`  | running max（用來避免溢位） |
| `ℓ`  | running sum of exp（softmax 分母） |
| `L = m + log(ℓ)` | logsumexp（forward 的 attention bookkeeping，backward 用） |

`exp(x - L) = exp(x - m) / ℓ` 正好是 softmax 結果，這就是為什麼 backward 只需存 `L`。

---

## M4. 「Backward 也要存 P」 ❌

不需要。FA backward **重算** P：拿出 Q, K, V, L，重算 `S = QK^T/√d`，再 `P = exp(S - L)`。多花了 forward 的時間，但省了 N² memory。這是經典 **memory ↔ compute trade-off**。

---

## M5. v1 與 v2 外迴圈順序到底哪個快？

兩個都對；v2 改外迴圈為 Q 是因為：
- 每個 thread block 負責一個 Q block，可天然分散到多 SM。
- 不需要在最終才 normalize（O 可在迴圈內直接累加，最後一次除 ℓ 即可）。
- 減少了「每內迴圈都要 rescale 整個 O」的非 matmul FLOPs。

v1 是先想到的版本，v2 是工程優化。

---

## M6. SRAM 與 shared memory 是同一件事嗎？

GPU 上 "SRAM" 是泛稱，包含：
- shared memory（共享記憶體，per-block，~100 KB）
- register file（per-thread）
- L1 cache

FA paper 用 "SRAM" 主要指 shared memory + registers 的可控部分。實作中 Q, K, V tile 通常放 shared memory，部分中間量放 register。

---

## M7. 「causal mask 就是把 P 上三角設 0」（在 FA 中不該這樣寫）

直接乘 mask 浪費，FA 的做法：
- 整塊 j > i：根本跳過內迴圈那一步。
- 對角塊 j == i：在 SRAM 中把 `S_ij` 對應位置設 -∞（exp 後為 0）。
- 整塊 j < i：照常算。

這樣 causal 大概省一半 FLOPs。

---

## M8. 「為什麼不直接降 N（用 sliding window）就好？」

Sliding window 是 **改變了 attention 的數學語義**（變成局部 attention）；FA 是 **不改語義** 的全 attention 加速。它們是正交的——可以同時用。

---

## M9. fp16 / bf16 / fp32 混合在哪裡？

FA 內部約定：
- Q, K, V, O 在 HBM：fp16/bf16
- S, P 在 SRAM 計算時：fp32 累加（特別是 softmax 的 sum）
- Tensor Core 的 mma：input fp16, accumulate fp32

**踩雷**：自己實作時把 `ℓ` 用 fp16 → 數值不穩。

---

## M10. dQ 為什麼要 atomic？

FA-2 backward 外迴圈是 j（K, V block），內迴圈 i（Q block）：
- `dV_j` 只在外迴圈 j 那層累加 → 一個 thread block 負責，沒有衝突。
- `dK_j` 同上。
- `dQ_i` 卻會在不同 j 的 iteration 都被累加 → 跨 thread block 衝突 → 需要 `atomicAdd` 或第二次 pass。

實作上常見：第一次 pass 算 dK, dV；第二次 pass 換內外迴圈算 dQ。或全用 atomic（簡單但可能慢）。

---

## 你的個人 confusion log（持續更新）

> 格式：`[YYYY-MM-DD] 我以為 X，實際上 Y，原因是 Z。`

- [2026-05-07] _初次學習，預留位置_。
