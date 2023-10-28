There are several patterns of dynamic programming problems:
1. 0/1 Knapsack
2. Fibonacci Sequence
1) climbing stairs  or Min Cost Climbing Stairs is very clear a Fibonacci Sequence
2) House Robber is similar as a Fibonacci Sequence, since the value of each house is non-negative, at each house we only need to make a decision whether to rob or not to rob, and we can summarize dp[i] = max(dp[i-1], dp[i-2] + nums[i])
3) Delete and Earn can be confused initially, but once we perm counter and sort,it will boil down to robber house problem, we only need to take care that the adjacency is based on value not index


