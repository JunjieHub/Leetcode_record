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

## Inorder Successor in BST
https://leetcode.com/problems/inorder-successor-in-bst/description/

Inorder travese can be used to get the sorted list of a tree. Keep in mind this propoerty, it allows us to find a specific node in O(logn) time.

```python
class Solution:
    def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
        successor = None
        while root:
            if p.val <= root.val:
                successor = root
                root = root.left
            else:
                root = root.right
        return successor
```
note: in this solution, be careful with the condition `p.val <= root.val`, it is not `p.val < root.val` because we need to find the smallest node that is larger than p.

## Inorder Successor in BST II
https://leetcode.com/problems/inorder-successor-in-bst-ii/description/
In this case, the node is defined to have access to its parent, which give us some convenience.
1) if node.right exist: we go to the right and then go to the leftmost node
2) if node.right does not exist: we go to the parent until the parent is larger than the node, which will be its successor.

```python
class Solution:
    def inorderSuccessor(self, node: 'Node') -> 'Node':
        if node.right:
            node = node.right
            while node.left:
                node = node.left
            return node
        else:
            while node.parent and node.parent.val < node.val:
                node = node.parent
            return node.parent
``` 



