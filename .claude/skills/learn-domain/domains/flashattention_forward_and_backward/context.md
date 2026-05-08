# FlashAttention Forward and Backward — 學習脈絡

## Big Picture

FlashAttention 是一種 **IO-aware** 的精確 attention 演算法，由 Tri Dao 等人於 2022 提出（FlashAttention v1），後續演進為 v2 (2023)、v3 (2024)。

它解決的是 Transformer 中 attention 層的兩個瓶頸：
- **記憶體**：標準 attention 必須在 HBM 中物化 N×N 的 attention matrix `S = QK^T` 與 `P = softmax(S)`，記憶體與運算複雜度都是 O(N²)。
- **頻寬**：在 GPU 上，attention 不是 compute-bound 而是 **memory-bound**——大部分時間在 HBM ↔ SRAM 之間搬資料。

FlashAttention 用三個關鍵技巧達成「**精確結果 + 線性記憶體 + 顯著加速**」：
1. **Tiling（分塊）**：把 Q、K、V 切成可放進 SRAM 的小 block，逐 block 計算。
2. **Online softmax（流式 softmax）**：在不看到完整 row 的情況下，用 running max 與 running sum 增量更新 softmax，數學上等價。
3. **Recomputation（重算）**：backward 時不存 attention matrix，僅存 softmax 統計量 `(m, ℓ)` 或 logsumexp `L`，反向時用 Q、K、V 重算 P。

結果：
- Memory：O(N²) → O(N)
- HBM access：O(N²·d) → O(N²·d²/M)（M 為 SRAM 大小，等於減少數倍 wall-clock）
- 數學上 **精確**，不是近似

## Why It Matters

1. **長序列訓練可行化**：GPT-3 等模型上下文 2k → 32k → 128k 都仰賴此類 kernel。
2. **是 LLM inference 的事實標準**：vLLM、SGLang、TensorRT-LLM 內部都用 FlashAttention 系列 kernel 或其變體（PagedAttention、FlashDecoding）。
3. **CUDA / kernel 設計範例**：完整體現了 GPU memory hierarchy 思考、tiling、warp-level primitive、async copy（cp.async）的重要性。
4. **面試高頻題**：系統工程師（NVIDIA、OpenAI、Anthropic、Meta）面試常問 online softmax 推導、recomputation trade-off、forward/backward 差異。

## 與本路線的銜接

| 你正在學的領域 | FlashAttention 的角色 |
|----------------|----------------------|
| CUDA           | 第一個值得徹底拆解的 kernel |
| LLM inference  | KV cache + prefill / decode 的核心 |
| Multi-GPU 訓練 | 與 sequence parallelism、ring attention 結合 |
| RLHF / PPO     | rollout / reference model forward 的瓶頸 |

## 學習目標（讀完此 pack 後你應該能）

- 用紙筆推導 online softmax 為什麼數學等價於標準 softmax。
- 寫出 FlashAttention forward 的 Python / NumPy 模擬版本（不需 CUDA）。
- 解釋 backward 為什麼只需要存 logsumexp `L` 一個 scalar per row。
- 區分 FlashAttention v1 / v2 / v3 的關鍵差異（外迴圈順序、warp 分工、Hopper 特化）。
- 在白板上畫出 SRAM / HBM 的資料流並回答 IO 複雜度。
