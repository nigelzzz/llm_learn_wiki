# 先備知識（Prerequisites）

## 必備（沒有這些寫不下去）

### 1. 指標 / 引用語意
- 你已具備（C++ 背景）。但 Python / Java 寫鏈表時要記得：
  - Python：`a = b` 是綁定到同一物件，`a.next = b` 改的是物件成員。
  - Java：所有非原始型別都是 reference，`ListNode a = b` 後 `a` 與 `b` 指向同一節點。
- 關鍵心法：**改「變數」不會影響鏈表，改「節點的 next」才會影響鏈表**。
  - `cur = cur.next` → 只是把區域變數指到下一個節點，**鏈表結構未變**。
  - `cur.next = X` → 真正修改鏈表結構。

### 2. ListNode 的標準定義

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

```cpp
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x = 0, ListNode* nxt = nullptr) : val(x), next(nxt) {}
};
```

雙向：再加一個 `prev`。循環：尾節點的 `next` 指回 head。

### 3. 不變式（Invariant）思維
寫鏈表程式時，**永遠在腦中維持一張圖**：
- 哪些節點目前還串在鏈上？
- 哪些指標暫時懸空？
- 在離開這段程式前，所有節點是否回到合法的鏈表結構？

訓練方法：寫每一行 `x.next = y` 之前，先在草稿紙畫一次箭頭。

## 強烈建議（不會也能寫，但會讓你少踩坑）

### 4. 雙指針（Two Pointers）的兩種思路
- **同向快慢**：`slow` 走一步、`fast` 走兩步 → 找中點、判環。
- **同向錯位**：`fast` 先走 K 步，`slow` 再開始 → 找倒數第 K 個。
- **反向夾擊**：在陣列上常見，鏈表上少用（因為單向鏈表無法反向走）。

### 5. 數學工具
- 快慢指針判環的數學推導（Floyd 演算法）：見 `core_concepts.md` §1.6。
- 鴿巢原理（287. 寻找重复数 用得到）。

### 6. 基本資料結構
- HashSet / HashMap：在 §1.2、§1.7、§1.10、§1.11 經常用作輔助。
- 堆（priority queue）：23 題（合併 K 個升序鏈表）的另一種解法。
- 雙向鏈表 + HashMap：LRU / LFU 的標配。

## 你已具備（不用補）

- C++ 指標、Python class、Java reference。
- Big-O 複雜度分析。
- 遞迴思維（會用在 25、24、206 的遞迴版）。

## 一個 5 分鐘暖身

如果好久沒寫鏈表，先做這個熱身練習，能寫出來再開始正式題單：

```python
# 熱身 1：印出鏈表
def print_list(head):
    cur = head
    while cur:
        print(cur.val, end=' -> ')
        cur = cur.next
    print('null')

# 熱身 2：把陣列轉成鏈表
def build(arr):
    dummy = ListNode()
    cur = dummy
    for x in arr:
        cur.next = ListNode(x)
        cur = cur.next
    return dummy.next

# 熱身 3：在不用 dummy 的情況下反轉鏈表
# （想清楚為什麼這題不需要 dummy）
def reverse(head):
    prev, cur = None, head
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return prev
```

寫完問自己：
- `build` 為何要 dummy？（→ 回答 Q1）
- `reverse` 為何不用 dummy？（→ head 沒有特殊性，反轉後新 head 就是 prev，dummy 也省不下任何 if）
- `reverse` 的迴圈條件為何是 `while cur`？（→ 要存取 `cur.next` 與 `cur.val`，cur 自己必須非 null）
