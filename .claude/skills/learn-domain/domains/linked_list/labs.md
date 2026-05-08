# 最小可運行 Labs

每個 Lab 都是「20-40 行 Python，不依賴外部 lib，含內建測試」。建議放在 `labs/linked_list/lab_XX.py` 自己刻一遍。

---

## Lab 0：通用工具與測試骨架

```python
# labs/linked_list/_common.py
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def build(arr):
    """陣列 → 鏈表"""
    dummy = ListNode()
    cur = dummy
    for x in arr:
        cur.next = ListNode(x)
        cur = cur.next
    return dummy.next

def to_list(head):
    """鏈表 → 陣列（測試比對用）"""
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out

def build_with_cycle(arr, pos):
    """造一個尾部接到 index=pos 的環，pos=-1 表示無環。"""
    head = build(arr)
    if pos == -1 or not head:
        return head
    tail = head
    target = None
    idx = 0
    while tail.next:
        if idx == pos:
            target = tail
        tail = tail.next; idx += 1
    if idx == pos:
        target = tail
    tail.next = target
    return head
```

---

## Lab 1：反轉鏈表（§1.4 模板）

```python
# labs/linked_list/lab_reverse.py
from _common import ListNode, build, to_list

def reverse(head):
    prev, cur = None, head
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return prev

def reverse_recursive(head):
    if not head or not head.next:
        return head
    new_head = reverse_recursive(head.next)
    head.next.next = head   # 把下一個的 next 反指回自己
    head.next = None
    return new_head

if __name__ == "__main__":
    cases = [[], [1], [1,2,3,4,5]]
    for arr in cases:
        assert to_list(reverse(build(arr))) == arr[::-1]
        assert to_list(reverse_recursive(build(arr))) == arr[::-1]
    print("OK Lab 1")
```

**要回答**：遞迴版的 `head.next.next = head` 為什麼能反指？（因為遞迴後 `head.next` 已經是新鏈表的尾節點。）

---

## Lab 2：找中點與判環（§1.6 模板）

```python
# labs/linked_list/lab_fast_slow.py
from _common import ListNode, build, build_with_cycle, to_list

def middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow

def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False

def cycle_entry(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            ptr = head
            while ptr is not slow:
                ptr = ptr.next; slow = slow.next
            return ptr
    return None

if __name__ == "__main__":
    assert middle(build([1,2,3,4,5])).val == 3
    assert middle(build([1,2,3,4])).val == 3   # 偶數時偏右
    assert has_cycle(build_with_cycle([1,2,3,4], 1)) is True
    assert has_cycle(build([1,2,3])) is False
    assert cycle_entry(build_with_cycle([1,2,3,4,5], 2)).val == 3
    print("OK Lab 2")
```

---

## Lab 3：刪除節點（§1.2 dummy + prev 模板）

```python
# labs/linked_list/lab_delete.py
from _common import build, to_list

def remove_val(head, val):
    dummy = type(head)(0, head) if head else None
    if dummy is None: return None
    prev = dummy
    while prev.next:
        if prev.next.val == val:
            prev.next = prev.next.next
        else:
            prev = prev.next
    return dummy.next

# 82：重複的全部刪掉（only keep distinct）
def delete_duplicates_ii(head):
    from _common import ListNode
    dummy = ListNode(0, head)
    prev = dummy
    while prev.next:
        if prev.next.next and prev.next.val == prev.next.next.val:
            v = prev.next.val
            while prev.next and prev.next.val == v:
                prev.next = prev.next.next
        else:
            prev = prev.next
    return dummy.next

if __name__ == "__main__":
    assert to_list(remove_val(build([1,2,6,3,4,5,6]), 6)) == [1,2,3,4,5]
    assert to_list(delete_duplicates_ii(build([1,2,3,3,4,4,5]))) == [1,2,5]
    assert to_list(delete_duplicates_ii(build([1,1,1,2,3]))) == [2,3]
    print("OK Lab 3")
```

**要回答**：在 82 題裡，為什麼 `prev = prev.next` 與「跳過一整段」是兩個分支？（因為跳過一整段後新的 `prev.next` 尚未驗證是否還與下一個相等，需再次進入 while 重判。）

---

## Lab 4：合併兩個有序鏈表（§1.8 模板）

```python
# labs/linked_list/lab_merge_two.py
from _common import ListNode, build, to_list

def merge(a, b):
    dummy = ListNode()
    tail = dummy
    while a and b:
        if a.val <= b.val:
            tail.next, a = a, a.next
        else:
            tail.next, b = b, b.next
        tail = tail.next
    tail.next = a or b
    return dummy.next

if __name__ == "__main__":
    assert to_list(merge(build([1,3,5]), build([2,4,6]))) == [1,2,3,4,5,6]
    assert to_list(merge(build([]), build([1]))) == [1]
    print("OK Lab 4")
```

---

## Lab 5：K 個一組翻轉（§1.4 進階）

```python
# labs/linked_list/lab_reverse_k.py
from _common import ListNode, build, to_list

def reverse_k_group(head, k):
    dummy = ListNode(0, head)
    group_prev = dummy
    while True:
        # 檢查剩餘是否夠 k 個
        kth = group_prev
        for _ in range(k):
            kth = kth.next
            if not kth: return dummy.next
        group_next = kth.next

        # 反轉 [group_prev.next, kth]
        prev, cur = group_next, group_prev.next
        while cur is not group_next:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
        # 接回去
        tmp = group_prev.next
        group_prev.next = kth
        group_prev = tmp

if __name__ == "__main__":
    assert to_list(reverse_k_group(build([1,2,3,4,5]), 2)) == [2,1,4,3,5]
    assert to_list(reverse_k_group(build([1,2,3,4,5]), 3)) == [3,2,1,4,5]
    assert to_list(reverse_k_group(build([1,2,3,4,5]), 1)) == [1,2,3,4,5]
    print("OK Lab 5")
```

---

## Lab 6：LRU Cache（§1.10 系統設計）

```python
# labs/linked_list/lab_lru.py

class Node:
    __slots__ = ('key', 'val', 'prev', 'next')
    def __init__(self, key=0, val=0):
        self.key, self.val = key, val
        self.prev = self.next = None

class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.map = {}
        # 雙 dummy（head/tail），免邊界
        self.head, self.tail = Node(), Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_front(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key):
        if key not in self.map: return -1
        node = self.map[key]
        self._remove(node); self._add_front(node)
        return node.val

    def put(self, key, value):
        if key in self.map:
            node = self.map[key]
            node.val = value
            self._remove(node); self._add_front(node)
            return
        if len(self.map) == self.cap:
            lru = self.tail.prev
            self._remove(lru)
            del self.map[lru.key]
        node = Node(key, value)
        self.map[key] = node
        self._add_front(node)

if __name__ == "__main__":
    c = LRUCache(2)
    c.put(1,1); c.put(2,2)
    assert c.get(1) == 1
    c.put(3,3)             # 淘汰 key=2
    assert c.get(2) == -1
    c.put(4,4)             # 淘汰 key=1
    assert c.get(1) == -1
    assert c.get(3) == 3
    assert c.get(4) == 4
    print("OK Lab 6 LRU")
```

**要回答**：
1. 為何要雙 dummy（head + tail）？→ 讓 `_remove` 不需判斷 `node.prev/next` 是否為 null。
2. 為何 HashMap 存 `key → node` 而不是 `key → val`？→ 拿到 node 才能 O(1) 從鏈表 detach。

---

## Lab 7：複製帶 random 指標的鏈表（§1.11）

```python
# labs/linked_list/lab_copy_random.py

class RNode:
    def __init__(self, val=0, nxt=None, rnd=None):
        self.val = val; self.next = nxt; self.random = rnd

def copy_random_list_hashmap(head):
    if not head: return None
    mp = {}
    cur = head
    while cur:
        mp[cur] = RNode(cur.val)
        cur = cur.next
    cur = head
    while cur:
        mp[cur].next = mp.get(cur.next)
        mp[cur].random = mp.get(cur.random)
        cur = cur.next
    return mp[head]

def copy_random_list_interleave(head):
    if not head: return None
    # 1) A → A' → B → B' → ...
    cur = head
    while cur:
        clone = RNode(cur.val, cur.next)
        cur.next = clone
        cur = clone.next
    # 2) 設 random
    cur = head
    while cur:
        if cur.random:
            cur.next.random = cur.random.next
        cur = cur.next.next
    # 3) 拆開
    new_head = head.next
    cur = head
    while cur:
        clone = cur.next
        cur.next = clone.next
        clone.next = clone.next.next if clone.next else None
        cur = cur.next
    return new_head
```

**要回答**：交織法的優點？→ O(1) 額外空間（不用 HashMap）。

---

## 練習方式

1. 先**不看 lab 答案**，看著上方 §x 模板自己寫一遍。
2. 寫不出來再對照。
3. 對照後**用自己的話寫一段註解**：「這題 dummy 解決了什麼？迴圈條件的選擇是？」
4. 一週後重做，對比第一次寫的版本。
