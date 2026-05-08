# 核心概念（Core Concepts）

按題單的 §1.1 ~ §1.11 順序拆解。每一節包含：**第一原理 → 模板 → 該節題目的歸類**。

---

## §1.1 遍歷鏈表

### 第一原理
鏈表沒有 index，唯一的存取方式是「從某個起點沿 `next` 走」。所有看起來複雜的題目，本質都是「邊走邊維護某個狀態」。

### 模板
```python
cur = head
while cur:
    # 用 cur.val 做事
    cur = cur.next
```

如果需要「相鄰兩節點」資訊：
```python
while cur and cur.next:
    # 用 cur 與 cur.next 比較或計算
    cur = cur.next
```

### 題目映射
| 題號 | 維護什麼狀態 |
|---|---|
| 1290 二進制鏈表轉整數 | `result = result * 2 + cur.val` |
| 2058 臨界點距離 | 上一個臨界點位置 + 第一個臨界點位置 |
| 2181 合併零之間 | 累加和，遇到 0 結算一段 |
| 725 分隔鏈表 | 先算長度 → 算每段 size → 截斷 |
| 817 鏈表組件 | 把 nums 放 set，邊走邊判斷「是否進入新組件」 |
| 3263/3294/3062/3063 | 雙鏈表轉陣列、找最大值、計頻率 → 都是純遍歷 |

**關鍵體會**：這節幾乎都用 `while cur != null`，因為要存取 `cur.val`。

---

## §1.2 刪除節點

### 第一原理
單向鏈表「刪除節點 X」= 讓 X 的前驅節點 `prev.next = X.next`。
所以**永遠需要 prev**。head 沒有 prev → 用 dummy 製造 prev。

### 模板（Dummy + Prev）
```python
dummy = ListNode(0, head)
prev = dummy
while prev.next:
    if should_delete(prev.next):
        prev.next = prev.next.next   # 刪
    else:
        prev = prev.next             # 走
return dummy.next
```

注意這裡迴圈條件是 `while prev.next`，因為要存取 `prev.next.val` 來判斷。**刪了之後不要動 prev**（因為新的 `prev.next` 還沒判斷過）。

### 題目映射
| 題號 | 條件 |
|---|---|
| 203 移除指定值 | `prev.next.val == val` |
| 3217 刪除 nums 中存在的節點 | nums 進 HashSet |
| 83 排序鏈表去重 | `cur.val == cur.next.val`（這題不用 dummy，head 不會被刪） |
| 82 排序鏈表去重 II（重複的全刪） | 需要 dummy。判斷 `prev.next.val == prev.next.next.val`，相等就把整段同值節點都剝掉 |
| 237 刪除指定節點（無 head） | 經典「val 覆蓋」技巧：`node.val = node.next.val; node.next = node.next.next` |
| 1669 區間替換 | 把 listA 的 [a, b] 區間整段替換成 listB |
| 2487 刪除右側更大的 | 反轉 → 單調棧 → 再反轉 |
| 1836 未排序去重 | HashMap 計頻 → dummy 過濾 |

**Q1 回答（部分）**：刪除類題目，當 head 可能被刪 → 一定要 dummy（203、82、3217）。head 不可能被刪（83，因 head 是去重後第一個出現）→ 可以不用。

---

## §1.3 插入節點

### 第一原理
插入 = 「找到要插入位置的前驅 prev」+ 「新節點 new」+ 「new.next = prev.next; prev.next = new」。
**順序很重要**：先接尾，再改 prev。順序反了會斷鏈。

### 模板
```python
new_node = ListNode(val)
new_node.next = prev.next
prev.next = new_node
```

### 題目映射
| 題號 | 在哪裡插？ |
|---|---|
| 2807 插入 GCD | 相鄰兩節點之間，計算 gcd 後插入 |
| 147 插入排序 | 從 dummy 開始找第一個比 cur 大的位置插入 |
| LCR 029 / 708 循環有序鏈表插入 | 找「val 介於 cur 與 cur.next」之間的位置；要處理「跨越循環邊界」與「全部相等」兩個 corner |
| 2046 絕對值排序 → 普通有序 | `|a| <= |b|` 排好的鏈表，把負數都拿到前面 |

**Q1 回答（部分）**：插入排序與「在 head 前可能插入」→ 用 dummy；循環鏈表插入時 head 沒有特殊性 → 不用 dummy。

---

## §1.4 反轉鏈表

### 第一原理
反轉 = 把每個節點的 `next` 指向「它原本的前驅」。需要三個指標：

```
prev    cur    next
 ↓       ↓      ↓
None    head    ...
```

每一輪：暫存 `next = cur.next` → 反指 `cur.next = prev` → 推進 `prev = cur, cur = next`。
迴圈結束後 `prev` 是新 head。

### 模板（迭代）
```python
def reverse(head):
    prev, cur = None, head
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return prev
```

### 模板（區間反轉，反轉 head 開始的 k 個節點，回傳新 head 與「原 head（現尾巴）」）
```python
def reverse_k(head, k):
    prev, cur = None, head
    while k > 0 and cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
        k -= 1
    return prev, cur   # 新頭, 反轉後段的下一個節點
```

### 題目映射
| 題號 | 變化 |
|---|---|
| 206 反轉鏈表 | 模板原版 |
| 92 反轉 [left, right] | 用 dummy 找到 `p = 第 left-1 個`，從 p.next 反轉 right-left+1 個，再把段首段尾接回去 |
| 24 兩兩交換 | 等價於 25 題 k=2，或用三指針一次處理兩個 |
| 25 K 個一組翻轉 | 先檢查剩餘是否夠 K 個（不夠就停），否則反轉這 K 個並接回去 |
| 2074 反偶數長度組 | 邊算組長邊反轉，奇數組跳過 |

**為何 92、25 需要 dummy？** head 可能被包含在反轉區間 → 反轉後 head 變了 → 要 dummy 統一接回。

---

## §1.5 前後指針（錯位距離 K）

### 第一原理
單向鏈表只能往前走。要找「倒數第 K 個」→ **讓兩個指針之間相差 K 步**，等快指針到尾，慢指針剛好在倒數第 K 個。

### 模板
```python
dummy = ListNode(0, head)
fast = slow = dummy
for _ in range(k):
    fast = fast.next
while fast.next:           # 注意是 fast.next，停在尾節點
    fast = fast.next
    slow = slow.next
# 此時 slow 是倒數第 (k+1) 個節點，slow.next 是倒數第 k 個
```

### 題目映射
| 題號 | 怎麼用錯位 |
|---|---|
| 19 刪倒數第 N 個 | 錯位 N，slow 停在倒數第 N+1 個的前驅 |
| 61 旋轉鏈表 k 次 | 錯位 `k % len`，斷成兩段重接（也可用「先連成環，再走 len-k 步斷開」） |
| 1721 交換正第 k 與倒數第 k | 錯位 + 兩個位置交換 val |
| 1474 刪 M 留 N 後再刪 M | 雙計數迴圈 |

**為何幾乎都需要 dummy？** 倒數第 K 個可能就是 head 本身（k = 鏈長），刪除/交換它需要 prev。

---

## §1.6 快慢指針（Floyd）

### 第一原理（找中點）
`fast` 每次 2 步，`slow` 每次 1 步。當 `fast` 到尾，`slow` 剛好在中點。
- 偶數長度時，slow 停在「中間偏右」(`while fast and fast.next`) 還是「中間偏左」(`while fast.next and fast.next.next`) 取決於迴圈條件，**面試請當場確認題意**。

### 第一原理（判環 + 找入口）
若有環，fast 必定追上 slow（在環內，fast 每輪縮短 1 步距離）。
**找入口**的數學：相遇時，head 和相遇點同步走，每次 1 步，再次相遇處就是入口。
- 證明：設 head 到入口距離 a，入口到相遇點距離 b，環長 L。
- slow 走了 `a + b`，fast 走了 `a + b + nL`，且 fast = 2 * slow → `a + b = nL` → `a = nL - b = (n-1)L + (L - b)`。
- 即「從 head 走 a 步」= 「從相遇點再走 (L-b) 步繞 (n-1) 圈到入口」。

### 模板
```python
# 找中點（偏右）
slow = fast = head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
# slow = 中點

# 判環
slow = fast = head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
    if slow is fast:
        return True
return False

# 找環入口
# ... 沿用上面，找到相遇點後：
ptr = head
while ptr is not slow:
    ptr = ptr.next
    slow = slow.next
return ptr
```

### 題目映射
| 題號 | 應用 |
|---|---|
| 876 中間結點 | 模板 |
| 2095 刪中間 | 中點 + 前驅 → 用 dummy |
| 234 回文鏈表 | 找中點 → 反轉後半 → 雙指針比對 |
| 2130 最大孿生和 | 找中點 → 反轉後半 → 雙端對應加 |
| 143 重排 L0→Ln→L1→Ln-1... | 中點 + 反轉後半 + 交錯合併 |
| 141 環形 | 模板 |
| 142 環入口 | 模板 |
| 457 環形陣列 | 把陣列當鏈表，`next(i) = (i + nums[i]) % n`，用 Floyd |
| 2674 拆分循環鏈表 | 找中點斷開 |
| 287 找重複數 | 把 `nums[i]` 看成 next 指標，鴿巢原理保證有環，找環入口 |
| 1015 / 3790 可被 K 整除最小全 1 | 把「餘數」看成節點，下一個餘數 = `(r * 10 + 1) % k`，找環判無解 |

---

## §1.7 雙指針（同向但邏輯不同）

### 第一原理
不是快慢，而是「兩條鏈表 / 兩條子鏈」同步走 — 用來歸類、分流、求交點。

### 題目映射
| 題號 | 思路 |
|---|---|
| 328 奇偶鏈表 | 兩條子鏈 `odd / even` 並行，最後 `odd.next = evenHead` |
| 86 分隔鏈表 | 兩條子鏈 `less / ge`，最後接起來。注意 `ge` 的尾巴要設 `null` |
| 160 相交鏈表 | A 走完接 B，B 走完接 A，第二輪會在交點相遇（或都到 null） |

160 的證明：A 走 `lenA + lenB`，B 走 `lenB + lenA`，路徑長度相同。若有交點 → 同步到交點；若無 → 同時到 null。

---

## §1.8 合併鏈表

### 第一原理
合併 = 用 dummy 構造新鏈，用 `tail` 不斷接尾。

### 模板
```python
dummy = ListNode()
tail = dummy
while a and b:
    if a.val <= b.val:
        tail.next = a; a = a.next
    else:
        tail.next = b; b = b.next
    tail = tail.next
tail.next = a or b  # 接上剩下的
return dummy.next
```

### 題目映射
| 題號 | 變化 |
|---|---|
| 21 合併兩有序 | 模板 |
| 2 兩數相加（個位在 head） | 同步走，維護 carry |
| 445 兩數相加（高位在 head） | 反轉 → 用 2 的方法 → 反轉回去；或用 stack |
| 2816 翻倍 | 等同於「鏈表 × 2」，反轉後逐位乘 2 |
| 369 給單鏈表加一 | 反轉 → +1 → 反轉；或遞迴帶 carry |
| 1634 多項式相加 | 同步走，按指數合併 |

---

## §1.9 分治

### 第一原理
鏈表分治 ≈ 把陣列 mergesort 套到鏈表上。**找中點靠快慢指針 → 切兩半 → 遞迴排 → 合併**（§1.8 模板）。

### 題目映射
| 題號 | 思路 |
|---|---|
| 23 合併 K 個升序 | 兩種解：① 兩兩合併（分治樹，O(N log K)）② 最小堆，每次取 K 個頭部最小 |
| 148 排序鏈表 | mergesort：找中點 → 斷開 → 各自排 → merge。要求 O(1) 空間就改成自底向上的迭代版 |

---

## §1.10 綜合應用（系統設計類）

| 題號 | 重點 |
|---|---|
| 1019 鏈表中下一個更大 | 把 val 抽成陣列 → 單調棧 |
| 1171 刪總和為零的連續區段 | 前綴和 → HashMap：相同前綴和之間的節點都刪掉 |
| 707 設計鏈表 | 練習：dummy + size 欄位 |
| 146 LRU | **HashMap + 雙向鏈表 + dummy head/tail**。`get` 與 `put` 都要 O(1)，所以節點要能在 O(1) 內 detach + insert |
| 460 LFU | LRU 升級版：`HashMap<key, node>` + `HashMap<freq, list>` + `min_freq` |
| 432 全 O(1) 結構 | 雙向鏈表存「相同 count 的桶」，每個桶內再用 HashSet |
| 1206 跳表 | 多層鏈表，每層一半的機率向上索引；查詢 O(log n) |

**LRU 是面試最高頻題，務必能徒手寫出。**

---

## §1.11 其他

| 題號 | 重點 |
|---|---|
| 138 帶 random 指標的複製 | 兩種解：① HashMap：`old → new` 兩遍掃描 ② 原地交織：`A → A' → B → B' → ...`，再拆開 |
| 382 隨機節點 | 蓄水池抽樣（Reservoir sampling）：第 i 個節點以 1/i 機率被選 |
| 430 多級雙向鏈表扁平化 | DFS / 遞迴：遇到 child 就把 child 子鏈插在 next 前 |

---

## 心法總結（Cheat Sheet）

| 場景 | 用什麼 | 迴圈條件 |
|---|---|---|
| 純遍歷 | cur | `while cur` |
| 看相鄰兩節點 | cur, cur.next | `while cur and cur.next` |
| 刪除 | dummy + prev | `while prev.next` |
| 插入 | dummy + prev | `while prev.next`（找位置時） |
| 反轉 | prev, cur, next | `while cur` |
| 倒數第 K | dummy + fast/slow 錯位 K | `while fast.next`（停尾） |
| 找中點 | fast/slow | `while fast and fast.next` |
| 判環 | fast/slow | `while fast and fast.next` |
| 合併 | dummy + tail | `while a and b` |
