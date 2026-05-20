# Confusion Log — FlashAttention Backward

邊學邊填。格式：日期 / 困惑點 / 當下理解 / 解決狀態。

---

## 預先放幾個「常見卡點」當提示（你可以逐一驗證後標✅）

### C1. 「不存 P 怎麼可能算 backward？」
- 卡點：直覺以為反傳一定要原始 P。
- 解：P 可由 `exp(S − L)` 重算，S 由 q,k 重算，L 前向存好。重算 ≠ 重存。
- 狀態：⬜ 待自己驗證（見 labs Lab 2）

### C2. 「D = rowsum(dO⊙O) 是從哪冒出來的？」
- 卡點：看到公式直接給 D，不知由來。
- 解：來自 softmax Jacobian 的 `Pᵢ·dPᵢ` 標量，代入 dP=dO Vᵀ 後化簡成 dOᵢ·Oᵢ。**關鍵動作：`dOᵢ` 與 j 無關，提出 Σⱼ 外，括號內塌縮回前向 Oᵢ。**
- 狀態：✅ 2026-05-20 Stage 1 手推通過

### C3. 「dQ 為什麼要累加，dV 不用？」
- 卡點：三個梯度看起來對稱，為何待遇不同。
- 解：取決於迴圈方向。外層 j 時 dK/dV 對固定 j 累加完一次寫回；dQ 對所有 j 求和故需跨 block 累加（atomic 或第二 kernel）。是排程問題不是數學問題。
- 狀態：⬜ 待 Lab 3 把 `+=` 改 `=` 驗證

### C4. 「logsumexp vs sum 差在哪？」
- 卡點：為何不存普通 sum。
- 解：L 含 max，重算 P 時 exp 不 overflow；且 L 足以重建 P。
- 狀態：⬜ 待 Lab 2 大數值測試

### C5. 「scaling τ=1/√d 在 backward 出現幾次？」
- 卡點：容易漏乘。
- 解：dQ、dK 各乘一次 τ（因為 S=QKᵀτ）；dV 不含 τ。
- 狀態：⬜ 待確認

### C6. 「為什麼說 attention 是 memory-bound，重算才划算？」
- 卡點：不理解 compute vs memory 的取捨。
- 解：GPU 算力遠大於 HBM 頻寬，省搬運比省計算重要。
- 狀態：⬜ 待自己用 roofline 直覺說一遍

### C7. 「softmax 反傳寫 J dp 還是 Jᵀ dp？」
- 卡點：以為「反向 = J dp」是通則。
- 解：通則是 `ds = Jᵀ dp`；softmax 的 `J = diag(p) − ppᵀ` 剛好**對稱**才退化成 `J dp`。別把對稱當通則，一律先記 Jᵀ。
- 狀態：✅ 2026-05-20 確認

### C8. 「dS 公式裡還有 dP，不就還是要 N×N？」
- 卡點：把「公式出現 dP」誤當成「執行時要存整個 dP」。
- 解：tiled kernel 裡 `dPᵢⱼ = dOᵢ Vⱼᵀ` 是**逐 tile 當場算小 block**，算完 dSᵢⱼ、累加進 dK/dQ 後即丟。SRAM 同時只住一個 tile，從不物化 N×N。
- 狀態：✅ 2026-05-20 確認

---

## 我的困惑（自行新增）

## 2026-05-20 — Stage 1 手推通關
- 完成：白紙從 `O=PV` 推到 dQ,dK,dV；C2/C7/C8 三點打通。
- 已對拍：closed / 重算P / tiled 三版與 torch.autograd fp64 誤差 ~1e-16（labs Lab 1–3）。
- 下一步：Stage 2 PyTorch reference + gradcheck。
- 仍待驗證：C1, C3, C4, C5, C6（多數靠 Lab 2/3 與 roofline 直覺）。
