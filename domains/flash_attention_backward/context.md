# FlashAttention Backward — 學習脈絡包

## Big Picture（大局觀）

FlashAttention 的核心賣點是：在**不把 N×N 的注意力矩陣寫回 HBM（顯存）**的前提下，算出 attention。前向（forward）大家都比較熟，難的是**反向（backward）**——因為反向需要用到前向的中間矩陣 `S = QKᵀ` 和 `P = softmax(S)`，但我們在前向**根本沒有把它們存下來**。

> 反向傳播的中心矛盾：我需要 P，但我沒存 P。
> FlashAttention 的答案：**重算（recompute）**，而不是**重存（re-materialize from HBM）**。

整個 backward 就圍繞一件事：用前向存下的少量統計量（`O` 與 logsumexp `L`），在每個 tile 內**即時重算** `S`、`P`，然後 fused 地累加出 `dQ`、`dK`、`dV`，全程不讓 N×N 矩陣落地到 HBM。

```
Forward 存下:  O (N×d),  L = logsumexp per row (N,)
Backward 輸入: dO (N×d)
Backward 輸出: dQ, dK, dV (各 N×d)
關鍵手法:      tile 內重算 P，online 累加，避免 O(N²) memory
```

## Why It Matters（為什麼重要）

1. **記憶體是真瓶頸**：標準 attention 反向要存 `P`（N×N），長序列下 N=8192 時就是 64M 個元素 ×2 bytes ×多頭，直接爆 HBM。FlashAttention backward 把記憶體從 O(N²) 降到 O(N)。
2. **面試高頻**：能講清楚「為什麼 backward 需要 recompute、recompute 了什麼、為什麼 logsumexp 就夠」是 LLM 系統工程師的分水嶺問題。
3. **是 kernel 工程的範本**：tiling、online softmax、recomputation、register/SRAM 管理，這些技巧在所有 fused kernel（如 fused MLP、Mamba）都會重複出現。
4. **理解 LLM 訓練成本**：訓練比推理更吃 backward，搞懂這裡才知道長 context 訓練的成本結構。

## 這個 pack 涵蓋什麼

- `prerequisites.md`：你需要先會的（softmax 微分、矩陣求導、online softmax、forward 概念）
- `core_concepts.md`：backward 的數學推導 + 演算法逐步拆解（重點檔）
- `roadmap.md`：分階段學習路線
- `labs.md`：從 PyTorch 純數學版 → tile 版 → 對拍的 minimal 實作
- `interview_questions.md`：面試題與標準答法
- `confusion_log.md`：困惑記錄（邊學邊填）

## 學習主線（建議順序）

1. 先確定你能**手推 softmax 的 Jacobian**（這是一切的根）。
2. 推導 attention 的 `dQ, dK, dV` 閉式解（先不管 tiling）。
3. 理解前向只存 `O, L` 為何足以重算 `P`。
4. 理解 `D = rowsum(dO ⊙ O)` 這個關鍵簡化量怎麼來的。
5. 把上面整合成 tiled backward 演算法。
6. 寫 PyTorch reference → 與 `torch.autograd` 對拍 → 再看 tile 版。
