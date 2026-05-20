# Interview Questions — FlashAttention Backward

每題附「面試官想聽什麼」+「精簡標準答法」。練習時請口頭講，控制在 2 分鐘內。

---

### Q1. FlashAttention backward 比 forward 難在哪？
**想聽**：抓到核心矛盾。
**答**：backward 需要 attention 機率矩陣 `P` 來反傳，但 forward 為了省記憶體**沒有存 P**（N×N 太大）。所以 backward 必須**重算 P**，而非從 HBM 讀回。難點在於如何在 tile 內即時重算 P 並 fuse 地累加 dQ/dK/dV，全程不落地 N×N。

---

### Q2. 前向只存 `O` 和 `L`（logsumexp），backward 怎麼還原 P？
**答**：`Pᵢⱼ = exp(Sᵢⱼ − Lᵢ)`，其中 `Lᵢ = logsumexp_j Sᵢⱼ`。backward 用該 block 的 q,k 重算 `Sᵢⱼ = qᵢ·kⱼ/√d`，減去存好的 `Lᵢ` 再取 exp，就**精確**還原 P。因為只存了 (N,) 的 L 和本來就要輸出的 O，額外記憶體是 O(N) 而非 O(N²)。

---

### Q3. 推導 `dS`。為什麼會出現 `D = rowsum(dO⊙O)`？
**答**：softmax 的 row Jacobian 給出 `dSᵢ = Pᵢ⊙(dPᵢ − Pᵢ·dPᵢ)`。其中標量 `Pᵢ·dPᵢ`，代入 `dP = dO Vᵀ` 化簡為 `dOᵢ·(Σⱼ Pᵢⱼ Vⱼ) = dOᵢ·Oᵢ`，即 `Dᵢ = rowsum(dO⊙O)ᵢ`。意義：本來要整列 P 才能算的量，變成只靠 dO 和 O 的逐 row 內積，**不需要 N×N**，這是 backward 能省記憶體的關鍵化簡。

---

### Q4. 寫出 dQ, dK, dV 的最終式子。
**答**（τ=1/√d）：
```
P  = exp(S − L)
dV = Pᵀ dO
dP = dO Vᵀ
D  = rowsum(dO ⊙ O)
dS = P ⊙ (dP − D)
dQ = τ · dS K
dK = τ · dSᵀ Q
```

---

### Q5. 為什麼重算 P 反而更快？不是多做了 FLOPs 嗎？
**答**：attention 是 **memory-bound** 不是 compute-bound。重算 P 多花的是算力（QKᵀ 再做一次，常數倍 FLOPs），但省下的是把 N×N 矩陣寫回又讀回 HBM 的**頻寬**。在現代 GPU 上頻寬才是瓶頸，所以「多算少搬」是淨贏。

---

### Q6. 為什麼 forward 要存 logsumexp 而不是單純的 sum？
**答**：兩個理由。(1) 數值穩定：`L ≥ rowmax(S)`，重算時 `exp(S−L)` 指數恆 ≤0 不會 overflow。(2) 一個 (N,) 的 L 就同時編碼了 max 與 sum 資訊，足以精確重建 P，省記憶體。

---

### Q7. dQ 的累加為什麼比 dK/dV 麻煩？
**答**：在「外層迴圈跑 K/V block j、內層跑 Q block i」的結構下，`dKⱼ, dVⱼ` 對固定 j 累加所有 i，可常駐 SRAM 後一次寫回。但 `dQᵢ = τΣⱼ dSᵢⱼ Kⱼ` 要對所有 j 求和，會被不同 j 反覆更新，需要 **atomic add 到 HBM** 或**另設一個外層 i 的 kernel** 專算 dQ（FlashAttention-2 的做法），以提升平行度、避免原子衝突。

---

### Q8. FlashAttention-2 相對 v1 在 backward 改了什麼？
**答**：主要是**工作分配與迴圈順序**的優化——調整 dQ/dK/dV 的計算順序與 thread block 分工，減少非 matmul 運算與 atomic、提高 GPU occupancy 與平行度，讓 backward 更接近硬體峰值。數學公式不變，變的是排程。

---

### Q9. 記憶體複雜度從多少降到多少？
**答**：標準 attention backward 需存 O(N²) 的中間矩陣（P 或 S）；FlashAttention backward 只需 O(N)（L 和 D，O 本來就要存）。HBM 流量也從 O(N²) 降到約 O(N²d/M)（M 為 SRAM tile 容量），這才是加速來源。

---

### Q10. 數值上有哪些坑？
**答**：(1) `D = rowsum(dO⊙O)` 在低精度要用 fp32 累加。(2) 重算 P 一定要減 L 防 overflow。(3) 對拍要用 fp32/fp64，否則 fp16 誤差會讓你誤判公式錯。(4) 別忘了 scaling τ=1/√d 在 dQ/dK 都要帶。

---

### 白板總結（30 秒電梯版）
> forward 不存 N×N 的 P，只存 O 和 logsumexp L。backward 在每個 tile 用 `P=exp(S−L)` 重算 P，靠 `D=rowsum(dO⊙O)` 把 softmax 反傳簡化掉 N×N 依賴，然後 fuse 地累加 dQ/dK/dV。用常數倍 FLOPs 換掉 O(N²) 的 HBM 流量，因為 attention 是 memory-bound。
