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

## Possible Bipartition
https://leetcode.com/problems/possible-bipartition/

We want to split a group of n people (labeled from 1 to n) into two groups of any size. Each person may dislike some other people, and they should not go into the same group.

Given the integer n and the array dislikes where dislikes[i] = [ai, bi] indicates that the person labeled ai does not like the person labeled bi, return true if it is possible to split everyone into two groups in this way.

 

Example 1:

Input: n = 4, dislikes = [[1,2],[1,3],[2,4]]
Output: true
Explanation: The first group has [1,4], and the second group has [2,3].
Example 2:

Input: n = 3, dislikes = [[1,2],[1,3],[2,3]]
Output: false
Explanation: We need at least 3 groups to divide them. We cannot put them in two groups.
'''
class unionfind:
    def __init__(self, size):
        self.root = list(range(size))
        self.rank = [1] * size
    
    def find(self, x):
        cur = x
        while self.root[cur] != cur:
            cur = self.root[cur]
        return cur
    
    def union(self, x, y):
        xroot = self.find(x)
        yroot = self.find(y)
        if xroot != yroot:
            if self.rank[xroot] < self.rank[yroot]:
                xroot, yroot = yroot, xroot
            self.root[yroot] = xroot
            self.rank[xroot] += self.rank[yroot]

    
class Solution:
    def possibleBipartition(self, n: int, dislikes: List[List[int]]) -> bool:
        uf = unionfind(n)
        dislike_graph = collections.defaultdict(list)
        for x1,x2 in dislikes:
            dislike_graph[x1-1].append(x2-1)
            dislike_graph[x2-1].append(x1-1)

        n_group = n
        for i in range(n):
            for dis_nei in dislike_graph[i]:
                if uf.find(i) == uf.find(dis_nei):
                    return False
                uf.union(dislike_graph[i][0], dis_nei)
        return True
'''

tip: for this problem, when we do union, considering we need to put all persons that person x dislike or be disliked in to another group, that's why we union as uf.union(dislike_graph[i][0], dis_nei).