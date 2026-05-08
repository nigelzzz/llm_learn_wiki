# Learning Roadmap — FlashAttention

## 階段 0：暖身（0.5 天）

- [ ] 複習標準 attention，能在白板上寫出 forward + backward 公式。
- [ ] 複習 numerically stable softmax（減 max 的證明）。
- [ ] 用 PyTorch 寫一個 naive attention，並打印中間 `S, P` 的 shape 與記憶體用量。

## 階段 1：理解 Online Softmax（1 天）

- [ ] 紙筆推導 online softmax 遞推式（看 core_concepts §3）。
- [ ] 做 Lab 1：純 NumPy，實作 online softmax，與 `scipy.special.softmax` 比對誤差 < 1e-6。
- [ ] 寫 200 字解釋：「為什麼 `e^(m_old - m_new)` 能修正舊的部分和？」

**關卡**：能在 5 分鐘內推完遞推式，並指出每一項的物理意義。

## 階段 2：Forward 演算法（1.5 天）

- [ ] 讀 FA-1 paper §3.1 + Algorithm 1（先讀 v1 比較好懂）。
- [ ] 做 Lab 2：用 PyTorch 實作 tiled forward（CPU 也行），驗證輸出與 naive attention 數值一致。
- [ ] 做 Lab 3：把 Lab 2 改成 `B_r, B_c` 可調，畫出 N=2048 時不同 block size 的 max memory 曲線。
- [ ] 讀 FA-2 paper §3.1，畫出 v1 vs v2 外迴圈順序差異的示意圖。

**關卡**：在白板畫出 SRAM 中同時駐留的 tensor 與大小。

## 階段 3：Backward 演算法（1.5 天）

- [ ] 從 attention 標準 backward 推一次（手推 dQ, dK, dV）。
- [ ] 確認 `L = m + log ℓ` 為什麼足以重建 `P`。
- [ ] 做 Lab 4：在 Lab 2 上加 backward，使用 `torch.autograd.gradcheck` 比對。
- [ ] 推導 `D_i = rowsum(dO ⊙ O)` 為什麼省了一次 N×N matmul。

**關卡**：能解釋為什麼 backward 用「外迴圈 K」更自然，以及 `dQ` 的 atomic 問題。

## 階段 4：CUDA / Triton 實作（2-3 天，視底子）

- [ ] 讀 Triton 官方 FlashAttention 教學：<https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html>
- [ ] 做 Lab 5：跑通 Triton 版本，benchmark vs `torch.nn.functional.scaled_dot_product_attention`。
- [ ] 看 `flash-attn` repo 的 CUDA 版本，找出 SRAM 配置（共享記憶體）程式碼。
- [ ] 做 Lab 6（選修）：把 Triton 版本擴充到 causal mask。

**關卡**：在自己 GPU 上看到 FA 比 naive 快至少 2x，且輸出一致。

## 階段 5：進階 / v3 / 變體（1-2 天）

- [ ] 讀 FA-3 paper，理解 producer-consumer warp 分工。
- [ ] 看 Ring Attention 與 FA 的整合方式。
- [ ] 看 PagedAttention（vLLM）如何修改 FA kernel 處理 page table。
- [ ] 做 Lab 7（選修）：把 forward 改成支援 `block_table` 的 paged 版本。

## 階段 6：面試與表達（0.5 天）

- [ ] 練習 5 分鐘版本：「請解釋 FlashAttention」。
- [ ] 練習 1 分鐘版本：「FA 為何快？」
- [ ] 對著鏡子或錄音回答 interview_questions.md 中的 5 題。

---

## 推薦資源

- FA-1 Paper：Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)
- FA-2 Paper：Tri Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (2023)
- FA-3 Paper：Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision" (2024)
- Repo：<https://github.com/Dao-AILab/flash-attention>
- Triton 教學：<https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html>
- 部落格：Horace He "Making Deep Learning Go Brrrr From First Principles"

---

## Next 7-Day Plan

針對你目前的程度（懂 C++/Python/系統、CUDA 與 LLM inference 在學）量身設計，每天 ~2 小時。

| Day | 主題 | 產出 |
|-----|------|------|
| Day 1 (週三 5/7) | 階段 0 暖身 + 階段 1 上半 | 紙筆推完 stable softmax；寫完 naive attention 並量 memory |
| Day 2 (週四 5/8) | 階段 1 下半：Lab 1 | online softmax NumPy 版過測試；200 字解釋 |
| Day 3 (週五 5/9) | 階段 2：Lab 2 | tiled forward 與 SDPA 數值一致 (atol 1e-5) |
| Day 4 (週六 5/10) | 階段 2 收尾：Lab 3 + 讀 FA-2 §3.1 | memory 曲線圖；v1 vs v2 示意圖 |
| Day 5 (週日 5/11) | 階段 3：手推 backward + Lab 4 | gradcheck 通過 |
| Day 6 (週一 5/12) | 階段 4 上半：Triton 教學 + Lab 5 | benchmark 表格 |
| Day 7 (週二 5/13) | 階段 6：面試題演練 | 對著鏡子回答 interview_questions A1, A3, B1, B2, C1 各一次 |

**Buffer**：第 8 天起可選做 causal mask (Lab 6) 或讀 FA-3。

**每日 ritual**：
- 開始前看 `confusion_log.md`（5 分鐘）。
- 結束前 append 今天的新 confusion（即使是「沒卡關」也記下哪題最有感）。
- 週日做 weekly review，回到 `notes/weekly_review.md`。
