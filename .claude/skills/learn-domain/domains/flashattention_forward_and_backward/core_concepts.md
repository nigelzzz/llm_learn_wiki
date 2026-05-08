# Core Concepts — FlashAttention 第一性原理

## 1. 標準 Attention 的 IO 病灶

```
標準演算法 (HBM):
  S = Q K^T          # 寫入 N×N 到 HBM
  P = softmax(S)     # 讀回 N×N，寫回 N×N
  O = P V            # 讀回 N×N，寫 N×d
```
HBM 讀寫次數 ≈ O(N² + N·d)。當 N 大時，N² 主導。
**關鍵洞察**：所有中間量 `S, P` 都不需要保存——它們只是 `O` 的中間產物。

## 2. Tiling（分塊）：把 attention 變成可流式計算

把 Q 切成 `T_r` 個 row block（每塊 `B_r` 行），把 K, V 切成 `T_c` 個 col block（每塊 `B_c` 行）。

```
for i = 1 .. T_r:                # 外迴圈：Q row block (FA-2)
    Q_i  ← HBM (load to SRAM)
    O_i  ← 0,  m_i ← -inf,  ℓ_i ← 0
    for j = 1 .. T_c:            # 內迴圈：K, V col block
        K_j, V_j ← HBM (load)
        S_ij = Q_i K_j^T / √d    # (B_r, B_c) 在 SRAM
        # ↓ online softmax 更新
        m_new = max(m_i, rowmax(S_ij))
        P_ij  = exp(S_ij - m_new)
        ℓ_new = exp(m_i - m_new) · ℓ_i + rowsum(P_ij)
        O_i   = diag(exp(m_i - m_new)) · O_i + P_ij V_j
        m_i, ℓ_i = m_new, ℓ_new
    O_i = O_i / ℓ_i              # 最後一次 normalize
    寫回 O_i 到 HBM；存 L_i = m_i + log(ℓ_i)（給 backward）
```

**要點**：每個 `S_ij` 只活在 SRAM；不寫回 HBM；O 一次完成。

## 3. Online Softmax（最核心的數學）

問題：標準 softmax 需要看完整列才能算 max 與 sum。Tiling 一次只看一個 chunk，怎麼辦？

**遞推公式**（一個 row 視角，已看到的 logits 為 x，下一塊為 y）：

```
m_new   = max(m_old, max(y))
ℓ_new   = e^(m_old - m_new) · ℓ_old + Σ e^(y_k - m_new)
o_new   = e^(m_old - m_new) · o_old + Σ e^(y_k - m_new) · v_k
```

最終輸出：`O = o_final / ℓ_final`。

**為什麼正確**：把每個 exp 都用相同的 `m_new` 為基準重新標準化，所有舊的部分和乘上 `e^(m_old - m_new)` 來「修正基準」。展開後等於標準 softmax 表達式。

> 動手推一次：寫一個 row 共 4 個 logits，分兩塊各 2 個，用 online 算一次，再用標準公式驗證 — 看 lab 1。

## 4. Backward：只存 logsumexp `L`

Standard backward 需要 `P`（N×N），太大。FlashAttention 的觀察：
- Forward 結束後存 `L_i = m_i + log(ℓ_i)`（每行一個 scalar，總共 O(N)）。
- Backward 重算：`P_ij = exp(S_ij - L_i)`，數值上完全正確（因為 `L = m + log ℓ`，所以 `exp(s - L) = exp(s - m) / ℓ`）。

Backward 三條公式（標準推導，FA 重點是「分塊+重算」）：
```
dV_j = Σ_i P_ij^T dO_i
dP_ij = dO_i V_j^T
dS_ij = P_ij ⊙ (dP_ij - rowsum(dO_i ⊙ O_i))     # 這個 D_i = rowsum(dO_i ⊙ O_i) 預先算好
dQ_i = Σ_j dS_ij K_j / √d
dK_j = Σ_i dS_ij^T Q_i / √d
```

**FA-2 backward 結構**：
- 先預計算 `D_i = rowsum(dO_i ⊙ O_i)`，每行一個 scalar。
- 外迴圈 j（K, V block），內迴圈 i（Q block），這樣 `dK_j, dV_j` 在外迴圈累加，避免 atomic。
- `dQ_i` 需要 atomic add 或第二次 pass。

## 5. v1 vs v2 vs v3（要記得的差異）

| 版本 | 年份 | 關鍵改變 | 加速來源 |
|------|------|----------|----------|
| v1   | 2022 | 提出 tiling + online softmax + recompute | 減少 HBM IO |
| v2   | 2023 | 外迴圈改為 Q（不是 K），減少非 matmul FLOPs；warp 分工改良 | 提升 occupancy 與 GEMM 比例 |
| v3   | 2024 | Hopper 專屬：WGMMA、TMA、producer-consumer warp、FP8 | 利用 H100 async tensor core |

**v1 vs v2 的核心差異**：
- v1 外迴圈 K → 每個 block 要 atomic 寫回 O。
- v2 外迴圈 Q → 一個 Q block 完整算完才寫一次 O，且最後才 normalize（scale by 1/ℓ）。

## 6. 複雜度速查

| 指標     | Standard | FlashAttention |
|----------|----------|----------------|
| FLOPs    | O(N² d)  | O(N² d)（一樣） |
| HBM IO   | O(N² + Nd) | O(N² d² / M)，M=SRAM size |
| Memory   | O(N²)    | O(N) |
| 數學等價 | -        | ✅ 精確 |

## 7. 常見變體 / 延伸

- **Causal mask**：只算下三角，省一半 FLOPs；FA kernel 內建支援。
- **ALiBi / Rotary**：bias 直接加在 S_ij，不影響架構。
- **PagedAttention** (vLLM)：把 KV cache 拆成 page 管理，FA kernel 變體。
- **Ring Attention**：跨 GPU 切 sequence，online softmax 在 ring 中傳遞 `(m, ℓ, O_partial)`。
- **FlashDecoding**：decode 時只有一個 query token，把 KV 切塊跨 SM 並行 reduction。
