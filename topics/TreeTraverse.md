# Summerize useful and representative tree traverse methods and trick

## 1. Binary Tree Zigzag Level Order Traversal
https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/description/

This is a BFS problem and can be achieved by using a queue, the trick part is we need to store the result level by level. To achieve this, the queue should store both the node and the level. Then we can conveniently store the result level by level.

```python
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        # bfs with stored level
        q = deque([(root, 0)])
        levels = defaultdict(list)
        while q:
            node, l = q.popleft()
            levels[l].append(node.val)
            if node.left:
                q.append((node.left, l+1))
            if node.right:
                q.append((node.right, l+1))

        res = []
        for i in range(len(levels)):
            if i % 2 == 0:
                res.append(levels[i])
            else:
                res.append(levels[i][::-1])
        return res
```
