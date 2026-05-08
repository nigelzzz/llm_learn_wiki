# Prerequisites — 學 FlashAttention 之前你需要會什麼

## 必要（缺了就無法理解推導）

### 1. Scaled Dot-Product Attention 標準形式
```
S = Q K^T / sqrt(d)        # (N, N)
P = softmax(S, dim=-1)     # row-wise softmax
O = P V                    # (N, d)
```
- 知道 `Q, K, V ∈ R^{N×d}`，N = sequence length，d = head dim。
- 知道 multi-head：每個 head 獨立做一次 attention。

### 2. Softmax 的數值穩定形式
```
softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
```
**關鍵**：必須先減 max 再 exp，否則 fp16/bf16 會溢位。FlashAttention 的 online softmax 直接建立在這個技巧上。

### 3. 微積分 / 反向傳播
- 熟悉 chain rule。
- 能推導 softmax 的 Jacobian：
  `dL/dS_ij = P_ij (dL/dP_ij - Σ_k P_ik · dL/dP_ik)`
- 能寫出 attention backward 的標準公式（dQ, dK, dV）。

### 4. GPU Memory Hierarchy（CUDA）
| 層級   | 大小（H100 為例） | 頻寬       | 特性 |
|--------|-------------------|------------|------|
| HBM    | ~80 GB            | ~3 TB/s    | 大、慢 |
| L2     | 50 MB             | ~7 TB/s    | 中 |
| SRAM (shared mem / register) | ~228 KB / SM | >19 TB/s | 小、快 |

**核心觀念**：GPU 是 memory-bound 機器；attention 的瓶頸是搬 N×N 矩陣，不是 FLOPs。

### 5. Tiling / Block Matrix Multiplication
- 會推導 `C = AB` 拆成 tile 後每個 block 的計算量。
- 知道 GEMM 在 SRAM 的 K 維度 reduction。

## 加分（讓你看 v2/v3 paper 不會迷路）

- **Warp / warp-group / thread block** 階層
- **cp.async** 與 `__pipeline` 雙緩衝
- **Tensor Core** 的 mma 指令（wmma / mma.sync）
- **CUTLASS** / **CuTe** 的 layout 概念
- **Triton** 語言（FlashAttention 有官方 Triton 教學版本）
- **bf16 / fp16 / fp8 numeric**：accumulation 必須在 fp32

## 自我檢測題（答得出再開始）

1. 為什麼 `softmax(x)` 要減 max？減 max 後結果為何不變？
2. Standard attention 在 N=8192, d=128 時，attention matrix 佔多少 GB（fp16）？
3. 在 A100 上，HBM 頻寬約 2 TB/s，做完一次完整 N=4096 的 attention 至少要讀寫多少 byte？大約幾 ms？
4. 推導 `O = P V` 對 V 的梯度 `dV`，假設 `dO` 已知。
5. 一個 SM 的 shared memory 約 100 KB，`Q_block` 與 `K_block` 各 64×128 fp16，能塞下幾組？

> 答對 4/5 以上才開始讀 v1 paper；否則先補 softmax 推導與 CUDA memory model。
