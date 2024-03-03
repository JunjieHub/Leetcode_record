## Unionfind
Here is a class for unionfind that I found can be repretedly used

```python
class UnionFind:
    def __init__(self, n):
        self.root = list(range(n)) # each node is initilized with itself as its root
        self.rank = [1] * n

    def find(self, x):
        cur = self.root[x]
        while cur != x:
            x = cur
            cur = self.root[x]
        return cur
        # if self.root[x] != x:
        #     self.root[x] = self.find(self.root[x])
        # return self.root[x]

    def union(self,x,y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x != root_y:
            # tend to add x to y, if rank(x) > rank(y), we swap
            if self.rank[x] > self.rank[y]:
                root_x, root_y = root_y, root_x
            self.rank[root_y] += self.rank[root_x]
            self.root[root_x] = root_y
```

# Problems:
## Find if path exists in a graph
https://leetcode.com/problems/find-if-path-exists-in-graph/

## Most Stones Removed with Same Row or Column
https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/

## Accounts Merge
https://leetcode.com/problems/accounts-merge/

