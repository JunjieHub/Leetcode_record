## topological sorting

Key points:
this is a general appraoch to solve DAG problems, we can use both BFS and DFS to solve it

### BFS (Kahn's algorithm)
1. initialize a list to represents the indegree of each node
2. preserve a queue to store the nodes with 0 indegree, in python we can simply use a list to simulate a queue
3. pop the node from the queue, and add it to the result list, then decrease the indegree of its neighbors by 1
4. while queue is not empty, we repeat 3 until the queue is empty
5. if the result list is not equal to the number of nodes (or not all value in indegree list is empty) then there is a cycle in the graph, else we can simply return the result list