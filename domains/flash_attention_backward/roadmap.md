# Learning Roadmap — FlashAttention Backward

分階段，每階段都有「完成判準」。不要跳階段。

## Stage 0：暖身（0.5 天）
- 重看 forward：S, P, O 三步、online softmax、存 L 的理由。
- **判準**：能默寫前向，並說明「為什麼前向要存 logsumexp 而不是 sum」。

## Stage 1：手推閉式梯度（1 天）⭐ 最關鍵
- 推 softmax Jacobian → `dS = P⊙(dP − D)`。
- 推 `D = rowsum(dO⊙O)` 的化簡。
- 推 `dQ, dK, dV` 三式。
- **判準**：白紙上、不看筆記，從 `O=PV` 推到 `dQ,dK,dV` 全程無斷點。

## Stage 2：PyTorch reference 實作（0.5 天）
- 用純張量運算實作 §1 閉式解（不 tiling）。
- 與 `torch.autograd.grad` / `gradcheck` 對拍。
- **判準**：dQ,dK,dV 與 autograd 在 fp32 下 `max abs err < 1e-4`。

## Stage 3：重算機制（0.5 天）
- 不存 P，只存 `O, L`；backward 時 `P = exp(S − L)` 重算。
- 驗證重算的 P 與直接 softmax 的 P 數值一致。
- **判準**：能解釋並驗證「存 L 即可精確還原 P」。

## Stage 4：Tiled 版（手寫迴圈，仍在 PyTorch）（1–2 天）
- 實作 §3 的 block 迴圈：外層 K/V block、內層 Q block。
- dQ 用 accumulate；先用簡單 Python 累加，理解 across-j 求和。
- **判準**：tiled 結果 == reference 結果；能講出 dQ 為何要跨 block 累加。

## Stage 5（選讀）：Triton/CUDA kernel（3–5 天）
- 把 Stage 4 搬到 Triton：`tl.load/store`、SRAM、program_id over blocks。
- 對照 FlashAttention-2 的迴圈順序改良與 dQ atomics 處理。
- **判準**：Triton backward 與 PyTorch reference 對拍通過。

## Stage 6：面試表達（0.5 天）
- 練 `interview_questions.md`，每題口頭講 ≤ 2 分鐘。
- **判準**：能在白板上 5 分鐘內講完「backward 為何需要 recompute、recompute 什麼、記憶體怎麼省」。

---

### 里程碑檢查
- [ ] Stage 1 手推通過（最重要，過不了別往下）
- [ ] Stage 2 reference 對拍通過
- [ ] Stage 4 tiled 對拍通過
- [ ] Stage 6 能流暢口述
