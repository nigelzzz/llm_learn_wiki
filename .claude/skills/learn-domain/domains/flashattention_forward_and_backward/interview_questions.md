# Interview Questions — FlashAttention

題目分三層：**概念**、**推導**、**系統設計**。每題附「金句答法」(white-board ready) 與「容易踩雷」。

---

## A. 概念題（30 秒～1 分鐘）

### A1. 一句話解釋 FlashAttention 為什麼快？

**金句**：FlashAttention 是 IO-aware kernel；attention 在 GPU 上是 memory-bound，瓶頸是搬 N×N matrix 進出 HBM。它用 tiling + online softmax 把整個 attention 在 SRAM 內 fuse 成一次性 kernel，HBM IO 從 O(N²) 降到 O(N²·d/M)，數學上仍精確。

**容易踩雷**：說「它減少了 FLOPs」——錯，FLOPs 一樣。

### A2. FlashAttention 是不是近似？

**金句**：不是。它對 forward 與 backward 都是位元級（up to 浮點 reorder）等價於標準 attention。近似 attention（Linformer、Performer）是另一條路線。

### A3. FA-1 vs FA-2 最大差異？

**金句**：v1 外迴圈是 K, V，每個 K 塊都要 atomic 更新所有 Q 的 O；v2 外迴圈改為 Q，每個 Q block 完整算完才 normalize 並寫一次 O，減少非 matmul FLOPs 並提高 occupancy（每個 thread block 只負責一個 Q block，可在多 SM 並行）。

---

## B. 推導題（白板 5～10 分鐘）

### B1. 推導 online softmax 的遞推式

**白板步驟**：
1. 寫出標準穩定 softmax：`s_i = exp(x_i - m) / Σ exp(x_j - m)`，其中 `m = max(x)`。
2. 假設已看到部分 logits，維護 `(m_old, ℓ_old)`，新增一塊 `y`：
   - `m_new = max(m_old, max(y))`
   - `ℓ_new = e^(m_old - m_new) ℓ_old + Σ e^(y - m_new)`
3. 同理累積 `O = Σ s_i v_i`：`O_new = e^(m_old - m_new) O_old + Σ e^(y - m_new) v`。
4. 最後 `O / ℓ`。

**驗證**：展開歸納法或直接代入 `m, ℓ` 的最終值即得標準形式。

### B2. 為什麼 backward 只存 logsumexp `L`？

**金句**：Forward 我們有 `m_i, ℓ_i`，存 `L_i = m_i + log(ℓ_i)`。Backward 重算 `S_ij = Q_i K_j^T / √d`，立刻可得 `P_ij = exp(S_ij - L_i)`（因為 `exp(S - m - log ℓ) = exp(S-m)/ℓ`，正是 softmax 結果）。每行只多存 1 個 fp32 scalar，總 O(N)。

### B3. 推導 dQ, dK, dV

```
給定 O = P V, P = softmax(S), S = QK^T / √d
dV = P^T dO
dP = dO V^T
dS_ij = P_ij (dP_ij - Σ_k P_ik dP_ik)     [softmax Jacobian]
       = P_ij (dP_ij - D_i)    其中 D_i = rowsum(dO ⊙ O)
dQ = dS K / √d
dK = dS^T Q / √d
```

**為什麼 `Σ_k P_ik dP_ik = rowsum(dO ⊙ O)`**：
`Σ_k P_ik (dO V^T)_ik = Σ_k P_ik Σ_t dO_it V_kt = Σ_t dO_it Σ_k P_ik V_kt = Σ_t dO_it O_it`。

**容易踩雷**：忘掉 softmax Jacobian 的 `-Σ` 項；把 D_i 算錯。

---

## C. 系統題（10 分鐘+）

### C1. 給你 H100（SRAM 228KB / SM），N=8192, d=128, fp16，怎麼選 Br, Bc？

**思路**：
- 一個 SM 要放：`Q_i (Br×d) + K_j (Bc×d) + V_j (Bc×d) + S_ij (Br×Bc) + O_i (Br×d)` 在 fp32 累加。
- d=128 fp16 = 256 B/row。
- 嘗試 Br=Bc=64：64×128×2 ≈ 16KB per matrix；S 用 fp32 = 64×64×4 = 16KB。約共 80KB，可裝下並留 double buffering。
- 太大（Br=128, Bc=128）→ 爆 SRAM；太小（16）→ tensor core 利用率差。
- 最終常見配置：Br=128, Bc=64 with cp.async double buffer。

### C2. 怎麼把 FlashAttention 推廣到 multi-GPU sequence parallelism？

**金句**：Ring Attention。把 Q 固定在本地，K, V 分塊在 GPU 間 ring 傳遞；每收到一塊就執行 FA 的內迴圈一次，更新 `(m, ℓ, O)`。最後把 partial 結果 normalize。HBM IO 變成 NVLink IO，但結構一樣。

### C3. 為什麼 inference decode 階段（單個 query）需要 FlashDecoding 而不是直接用 FA？

**金句**：Decode 時 query 只有 1 個 token (`Br=1`)，外迴圈 Q 只剩一個 block——只用一個 SM，其他 SM 全閒。FlashDecoding 把 K, V 沿 sequence 維度切多塊分散到不同 SM 並行，每塊算 partial `(m, ℓ, O)`，最後再做一次 reduction（再次 online softmax 合併）。

### C4. 什麼情況下 FA 不會比 SDPA 快？

- 序列很短（N < 512）：HBM IO 不是瓶頸，啟動 kernel overhead 主導。
- 非標準 mask（任意 attention bias）：分支成本可能蓋過 IO 節省。
- 早期 fp32 訓練：FA 主要為 fp16/bf16 優化。

---

## D. 速答（warm-up）

| Q | A |
|---|---|
| FA 改變了 FLOPs 嗎？ | 沒有 |
| FA 是近似嗎？ | 否，精確 |
| Backward 重算什麼？ | P 矩陣 |
| Backward 額外存什麼？ | logsumexp L，每 row 一個 fp32 |
| v2 外迴圈是？ | Q |
| v1 外迴圈是？ | K |
| FA 在 SRAM 還是 HBM 算 S？ | SRAM |
| Memory 複雜度？ | O(N) |
