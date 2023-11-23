## topological sorting

Key points:
this is a general appraoch to solve DAG problems, we can use both BFS and DFS to solve it

### BFS (Kahn's algorithm)
1. initialize a list to represents the indegree of each node
2. preserve a queue to store the nodes with 0 indegree, in python we can simply use a list to simulate a queue
3. pop the node from the queue, and add it to the result list, then decrease the indegree of its neighbors by 1
4. while queue is not empty, we repeat 3 until the queue is empty
5. if the result list is not equal to the number of nodes (or not all value in indegree list is empty) then there is a cycle in the graph, else we can simply return the result list

## Union Find
Union Find is a data structure that can be used to solve the connectivity problem, it can be used to determine whether two nodes are connected in a graph, and it can also be used to find the number of connected components in a graph



## prim's algorithm for MINIMUM SPANNING TREE
the prim algorithm is a greedy algorithm to find the minimum spanning tree in a graph, we traverse the graph from a random node, and find the minimum edge to connect to the current tree, and repeat this process until all nodes are connected

implementation trick:
1. preserve a set to store the nodes that are already in the tree so we can determine the termination condition
2. preserve a minheap to store the edges, the element of the heap is [cost, node], python determine the order of the element by the first element, so we can use the cost to determine the order
3. we can expand the current tree by pop the minimum edge from the heap, we can also update the heap by adding the newly added node's neighbors to the heap


## Djikstra's algorithm for SHORTEST PATH
example problem: 
https://leetcode.com/problems/network-delay-time/
https://leetcode.com/problems/cheapest-flights-within-k-stops/


## Bellman Ford's algorithm for SHORTEST PATH
https://leetcode.com/problems/cheapest-flights-within-k-stops/




