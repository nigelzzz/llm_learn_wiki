# Confusion Log

> 每次 interview-drill 或學習 session 結束後新增條目。**先寫卡點 → 寫錯誤直覺 → 寫修正後直覺 → 給最小例子 → 列下次要複習什麼**。

---

## 2026-05-07

### Topic
鏈表 §1.1 遍歷 — 迴圈條件 `while(cur)` vs `while(cur->next)` 的判斷。

### What confused me
看到鏈表迴圈，下意識說成「因為要讀 `cur->next`，所以 cur 不能 null」。在 1290、2181 兩題各被糾正一次仍重犯。

### My wrong intuition
「鏈表的核心是 next 指標，所以迴圈條件一定是在判 next」 — 把「鏈表的本質」與「這個迴圈體實際讀取什麼」混為一談。

### Correct intuition
迴圈條件由**迴圈體內實際讀取的最深成員**決定：
- 迴圈體讀 `cur->val` → `while (cur != null)`
- 迴圈體要看 `cur->next->val` 或要修改 `cur->next` → `while (cur->next != null)`
- 快慢指針要 `fast->next->next` → `while (fast != null && fast->next != null)`

判斷流程：先寫迴圈體，找出最深的 `cur->X`，這個 X 必須存在 → 反推 cur 或 cur->next 哪個必須非 null。

### Minimal example
```cpp
// 1290 / 2181 都用這種：迴圈體讀 cur->val
while (cur) {
    sum += cur->val;       // 讀 cur->val
    cur = cur->next;       // 推進（cur->next 即使是 null 也合法，下輪 while 自然結束）
}

// 203 刪除節點：迴圈體要改 prev->next
while (prev->next) {       // 必須保證 prev->next 存在才能比較與重接
    if (prev->next->val == val) prev->next = prev->next->next;
    else prev = prev->next;
}
```

### Review later
- §1.2 刪除節點所有題目 — 強迫自己每題寫迴圈條件前先說「迴圈體讀什麼」。
- §1.6 找中點 / 判環 — 三條件 `fast && fast->next` 的推導。

---

## 2026-05-07

### Topic
鏈表 §1.8 合併鏈表 / 構造新鏈 — `tail` 游標推進。

### What confused me
寫 2181 的 pseudo-code 時，每次結算都寫了 `tail->next = new_node`，但**忘記寫 `tail = tail->next`**。同樣的混淆出現在 trace 裡：cur=2 時就把 tail 寫成 11。

### My wrong intuition
「sum 累積到 11」與「tail 變成 11」是同一件事 — 把「資料的計算」與「指標的推進」混為一談。覺得只要 `tail->next = x`，tail 自己就會「跟過去」。

### Correct intuition
**指標不會自己移動**。`tail->next = X` 只是接尾，tail 變數本身仍指向舊節點。要把游標推進到新尾，必須**手動寫一行 `tail = tail->next`**。

事件分離：
- **資料事件**：sum += cur->val（每次 else 分支）
- **構造事件**：tail->next = node(sum); **tail = tail->next**; sum = 0（每次 if 分支）

兩件事發生在不同時刻，trace 表必須分開列。

### Minimal example
```cpp
// 錯：tail 從來沒推進，新節點互相覆蓋
tail->next = new ListNode(4);   // dummy.next = 4
tail->next = new ListNode(11);  // dummy.next = 11，4 被覆蓋掉！
// 輸出只有 11

// 對：每次接尾後立刻推進
tail->next = new ListNode(4); tail = tail->next;
tail->next = new ListNode(11); tail = tail->next;
// 輸出 4 → 11
```

口訣：「**接尾必推進**」— 寫 `tail->next =` 之後第一反應就是補一行 `tail = tail->next`。

### Review later
- 21 合併兩個有序鏈表（連寫 3 遍，不看答案）
- 2 兩數相加（同一個 dummy + tail 模板）
- 23 合併 K 個升序鏈表（同模板放大）
- 寫完任何「構造新鏈」的程式碼，**回頭數一下 `tail->next =` 與 `tail = tail->next` 是不是配對出現**。

---

## 2026-05-07

### Topic
回答結構 — 面試表達的「結構化」訓練。

### What confused me
被問到「不變式 / 邊界 / 複雜度」時，習慣用 1-2 個英文片語回答，例如「dummy: avoid null」、「O(N)」。被面試官打回多次後才意識到：簡短回答 ≠ 高效回答。

### My wrong intuition
「我心裡知道答案就好了，越短越專業」。

### Correct intuition
面試官**看不到**心裡的答案，只能透過你說出口的句子評估。對 5 個子問題各給 1 個片語 → 等於 5 個都沒答。

正確的表達應該照「白板六步法」：
1. 複述題意 → 2. 小例子 → 3. 演算法+不變式 → 4. 邊界決策 → 5. Pseudo-code → 6. Trace+複雜度

每一步都要寫**完整句子**，不要用片語。

### Minimal example
- ❌ 「dummy: avoid null」
- ✅ 「用 dummy node 是為了讓新鏈表的構造能用統一的 `tail->next = new_node` 動作，避免為「第一個輸出節點」寫特例分支。」

### Review later
- §1.1 剩下的 817、2058 — 強迫自己用六步法答，沒有模板提示。
- 做完每題後自問：「我講的內容，能不能被沒看過題目的人 60 秒內聽懂？」

---

## 2026-05-20

### Topic
FlashAttention backward Stage 1 — 手推閉式梯度 `dQ, dK, dV`（softmax 反傳 + `D` 化簡）。

### What confused me
三個卡點：(1) softmax 反傳到底寫 `J dp` 還是 `Jᵀ dp`；(2) `D = rowsum(dO⊙O)` 從哪冒出來、為何不需要 `dP`；(3) 既然 `dS` 公式裡還有 `dP`，為何說 backward 不物化 `N×N`。

### My wrong intuition
- 以為「反向傳播 = `J dp`」是通則。
- 看到 `D = Σⱼ Pᵢⱼ dPᵢⱼ` 直覺它一定要整個 `N×N` 的 `dP` 才能算。
- 把「公式裡出現 `dP`」誤當成「執行時必須存下整個 `dP`」。

### Correct intuition
- 反向傳播通則是 **`ds = Jᵀ dp`**；softmax 的 `J = diag(p) − ppᵀ` 剛好**對稱**，才退化成 `J dp`。別把對稱當通則，一律先記 `Jᵀ`。
- `Dᵢ = Σⱼ Pᵢⱼ dPᵢⱼ`，代入 `dPᵢⱼ = dOᵢ·Vⱼ` 後，**關鍵動作是「`dOᵢ` 與 `j` 無關，提出 `Σⱼ` 外」**，括號內 `Σⱼ Pᵢⱼ Vⱼ` 塌縮回前向的 `Oᵢ`，得 `Dᵢ = dOᵢ·Oᵢ`。於是只需 `dO`(輸入)和 `O`(前向已存)，零 `N×N`。
- `dP` 是**逐 tile 當場算 `dPᵢⱼ = dOᵢ Vⱼᵀ`**（小 block），算完 `dSᵢⱼ`、累加進 `dK/dQ` 後即丟。SRAM 同時只住一個 tile，從不物化整個 `N×N`。

### Minimal example
```python
# D 化簡的核心：dOᵢ 與 j 無關 → 提出求和
# Dᵢ = Σⱼ Pᵢⱼ (dOᵢ·Vⱼ) = dOᵢ·(Σⱼ Pᵢⱼ Vⱼ) = dOᵢ·Oᵢ
D = (dO * O).sum(-1)          # rowsum(dO⊙O)，(N,)，預先一次算好
# 完整鏈：dV=PᵀdO; dP=dO Vᵀ; dS=P⊙(dP−D); dQ=τ dS K; dK=τ dSᵀQ
```
（已用 fp64 對拍 torch.autograd，誤差 ~1e-16，見 domains/flash_attention_backward/labs.md Lab 1–3）

### Review later
- Stage 2：把閉式解寫成 PyTorch reference 並 gradcheck（labs Lab 1）。
- Stage 4：tiled 版，故意把 `dQ[i] +=` 改 `=` 體會「dQ 跨 key block 累加」。
- 口頭測驗：2 分鐘內從 `O=PV` 默推到 `dQ,dK,dV`，並講出 `D` 由來一句話。
