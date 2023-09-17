## Longest Substring Without Repeating Characters
[! Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
key point: how to update the left part of the sliding window? 
A: remove the character of current window's left from the 'seen' set until we find the duplicate one that is same as current right character. 
```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if len(s) == 0:
            return 0
        if len(s) == 1:
            return 1
        
        l = 0
        r = 0
        seen = set()
        res = 0
        while r < len(s):
            if s[r] not in seen:
                res = max(res, r-l+1)
            else:
                while s[r] in seen:
                    seen.remove(s[l])
                    l += 1
            seen.add(s[r])
            r += 1
        return res
```

## Longest Repeating Character Replacement
[! Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/)
key point: preserving a hashmap to store the count of each character in the current window, which help decide when to update the left part of the sliding window. 
```python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:       
        l = 0
        r = 0
        MaxLen = 0
        count = {}
        maxf = 0
        while r < len(s):
            count[s[r]] = count.get(s[r], 0) + 1
            maxf = max(maxf, count[s[r]])
            # while the sliding window is invalid, increase left index until it is valid
            while (r-l+1)-maxf > k:
                count[s[l]] -= 1
                l += 1
            MaxLen = max(MaxLen, r-l+1)
            r += 1
        return MaxLen
```

## Permutation in String
[! Permutation in String](https://leetcode.com/problems/permutation-in-string/)
key point: use a hashmap to store all lowerletter's count and initialize with 0, after sliding the window once, we update the hashmap instead of re-counting the whole string. 
```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        count_s1 = {letter: 0 for letter in string.ascii_lowercase}
        sub_count = {letter: 0 for letter in string.ascii_lowercase}

        for c in s1:
            count_s1[c] += 1
        
        l = 0
        r = len(s1)-1

        for c in s2[l:r+1]:
            sub_count[c] += 1
        
        while r < len(s2):
            if sub_count == count_s1:
                return True
            else:
                if r+1 < len(s2):
                    sub_count[s2[l]] -= 1
                    sub_count[s2[r+1]] += 1
                l += 1
                r += 1

        return False
```

## Minimum Window Substring
[! Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)
key point:
1. Use two pointers: start and end to represent a window.
2. Move end to find a valid window.
3. When a valid window is found, move start to find a smaller window.

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        countT = Counter(t)
        
        # initialize
        countS = {}
        l, r = 0, 0
        minl = float('infinity')
        res = ""
        have, need = 0, len(countT.keys())
        while r < len(s):
            # # update S table, we don't need to store other character
            if s[r] in countT:
                countS[s[r]] = countS.get(s[r], 0) + 1
                # update have only when the count match exactly, which means:
                # after updateing we just complete a task that is not finished before
                if countS[s[r]] == countT[s[r]]:
                    have +=1
                # after updating the table check whether ths substring is valide
                while have == need:
                    # check and update minl and res if necessary
                    if r-l+1 < minl:
                        minl = min(minl, r-l+1)
                        res = s[l:r+1]
                    # update left side of the window
                    if s[l] in countT:
                        countS[s[l]] -= 1
                        # update have
                        if countS[s[l]] < countT[s[l]]:
                            have -= 1
                    
                    l += 1
            r += 1
        return res
```

## Sliding Window Maximum
[! Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/)
key point: use a deque to store the index of the maximum value in the current window, and the index of the maximum value is always at the left of the deque. 
```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        # important: deq store the index instead of value because access value from
        # index is constant time and we need index to decide whether the left side of
        # the window has removed our  previous largest element
        deq = collections.deque()
        res = []
        l, r = 0, 0
        while r < len(nums):
            # pop the queue until decreasing property is maintained
            while deq and nums[deq[-1]]< nums[r]:
                deq.pop()
            # append the element
            deq.append(r)

            # decide whether we need to remove the left side of deq if it
            # is no longer within the sliding window

            if deq[0] < l:
                deq.popleft()

            # get res after reaching to the window size
            if r-l+1 == k:
                res.append(nums[deq[0]])
                l += 1
            r += 1
        return res
```

## Min stack
[! Min stack](https://leetcode.com/problems/min-stack/)
key point: A trick, preserving stack it self and also a min stack so we can access min value with constant complexity
```python
class MinStack:

    def __init__(self):
        self.stack = []
        self.stack_min = []
        

    def push(self, val: int) -> None:
        self.stack.append(val)
        if not self.stack_min:
            self.stack_min.append(val)
        else:
            if val < self.stack_min[-1]:
                self.stack_min.append(val)
            else:
                self.stack_min.append(self.stack_min[-1])

    def pop(self) -> None:
        self.stack.pop()
        self.stack_min.pop()


    def top(self) -> int:
        return self.stack[-1]
        

    def getMin(self) -> int:
        return self.stack_min[-1] if self.stack else None


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
```
## Evaluate Reverse Polish Notation
[! Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/)
key point: use a stack to store the number, and pop two numbers when we encounter an operator
```python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        # def character to operator
        def Operate(ope_char, opr1, opr2):
            if ope_char == '+':
                return opr1+opr2
            if ope_char == '-':
                return opr1-opr2
            if ope_char == '/':
                return opr1/opr2
            if ope_char == '*':
                return opr1*opr2
        
        operators = ['+', '-', '/', '*']
        work_stack = []
        for c in tokens:
            if c in operators:
                opr2 = int(work_stack.pop())
                opr1 = int(work_stack.pop())
                work_stack.append(Operate(c, opr1, opr2))
            else:
                work_stack.append(c)
        
        return int(work_stack[0])
```

## Generate Parentheses
[! Generate Parentheses](https://leetcode.com/problems/generate-parentheses/)
key point: backtracking and use stack to store the workspace
```python
class Solution(object):
    def generateParenthesis(self, n):

        res = []
        stack = []
        def backtrack(openN, closedN):
            # terminate condition
            if openN == closedN == n:
                res.append(''.join(stack))
                return
            
            if openN < n:
                stack.append('(')
                backtrack(openN+1, closedN)
                stack.pop()

            if closedN < openN:
                stack.append(')')
                backtrack(openN, closedN+1)
                stack.pop()

        backtrack(0,0)
        return res
```
Notice: it is good to systematically understand how backtracking works before I try to practice more problems.

## Daily Temperatures
[! Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)
key point: use a stack to store the temperature in monotonically decreasing order, and pop the stack when we encounter a higher temperature
```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        stack = []
        res = [0]*len(temperatures)

        # store both index and temp as a pair
        for i in range(len(temperatures)):
            while stack and temperatures[i]>stack[-1][0]:
                temp, tempInd = stack.pop()
                # update results since an increasement in temperature has been found
                res[tempInd] = i - tempInd
            stack.append([temperatures[i], i])
        return res
```

## The following are examples for backtracking using template
## Subsets
[! Subsets](https://leetcode.com/problems/subsets/)
key point: backtracking
```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        solutions = []
        state = []
        self.search(state, solutions, nums)
        return solutions
    
    def is_valid_state(slef, state, nums):
        if not state:
            return True
        else:
            return state[-1]<len(nums)

    def get_candidates(self, state, solutions, nums):
        if not state:
            # all elements can be candidate
            candidates = range(len(nums))
        elif state[-1] == len(nums)-1:
            candidates = []
        else:
            candidates = [i for i in range(state[-1]+1, len(nums))]
        return candidates

    def search(self, state, solutions, nums):
        if self.is_valid_state(state, nums):
            
            solutions.append([nums[i] for i in state])

        for candidate in self.get_candidates(state, solutions, nums):
            state.append(candidate)
            self.search(state, solutions, nums)
            state.pop()
```

## Combination Sum
[! Combination Sum](https://leetcode.com/problems/combination-sum/)

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        solutions = []
        state = [] # store index
        self.search(state, solutions, candidates, target)
        return solutions
     
    def is_valid_state(self, state, target, candidates):
        # print('state', state)
        if not state:
            return False
        else:
            return sum([candidates[i] for i in state]) == target

    def get_candidates(self, state, candidates, target):
        if not state:
            remain = target
            cand = range(len(candidates))
        else:
            remain = target - sum([candidates[i] for i in state])
            cand = [i for i in range(state[-1], len(candidates)) if candidates[i] <= remain]
        # print('cand', cand)
        return cand

    def search(self, state, solutions, candidates, target):
        if self.is_valid_state(state, target, candidates):
            # print('valid')
            temp = [candidates[i] for i in state]
            solutions.append(temp)
        
        for candidate in self.get_candidates(state, candidates, target):
            # print(state)
            state.append(candidate)
            self.search(state, solutions, candidates, target)
            state.pop()
```

## Permutations
[! Permutations](https://leetcode.com/problems/permutations/)
```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        solutions = []
        state = []
        remain = set(nums)
        self.search(state, solutions, remain, nums)
        return solutions


    def is_valid_state(self, state, nums):
        return len(state) == len(nums)

    def get_candidates(self, remain):
        candidates = list(remain)
        return candidates

    def search(self, state, solutions, remain, nums):
        if self.is_valid_state(state, nums):
            solutions.append(list(state.copy()))

        for c in self.get_candidates(remain):
            state.append(c)
            remain.discard(c)
            self.search(state,solutions, remain, nums)
            state.pop()
            remain.add(c)
```
## Subsets II
[Subsets II](https://leetcode.com/problems/subsets-ii/)
```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        sorted_nums = sorted(nums)
        solutions = []
        state = []
        self.search(state, solutions, sorted_nums)
        return solutions



    def is_valid_state(self, state, nums):
        # print(state)
        # print(nums)
        return len(state)<=len(nums)

    def get_candidates(self, state, solutions, nums):
        if not state:
            candidates = range(len(nums))
        elif state[-1] == len(nums)-1:
            candidates = []
        else:
            # print(state)
            candidates = [i for i in range(state[-1]+1, len(nums))]
            # print('candidates',candidates)
        print('candidates', candidates)
        return candidates

    def search(self, state, solutions, nums):
        # print(state)
        if self.is_valid_state(state, nums):
            temp = [nums[i] for i in state]
            if temp not in solutions: ## this part is very ineffcient but I don't know how to improve it
                solutions.append(temp)
        
        for c in self.get_candidates(state, solutions, nums):
                state.append(c)
                self.search(state, solutions, nums)
                state.pop()
```
A more concise solution:

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        def backtrack(state, solutions, cur_ind):

            for i in range(cur_ind, len(nums)):
                if i > cur_ind and nums[i] == nums[i-1]:
                    continue

                state.append(nums[i])
                backtrack(state, solutions, i+1)
                state.pop()
            solutions.append(state.copy())

        nums.sort()
        state, solutions = [], []
        backtrack(state, solutions, 0)
        return solutions
```



Note:
The template is clean but my be overcomplicated, especillay a lot self and arguments need to be written for the helper function. Now I understand the general architecture of backtracking, I can write it in a more concise way.
for backtrack, we
1) check whether current state is valid, if true, add state to solutions, the state must be a deep copy of the current state, otherwise, it will be changed when the state is changed.
2) get candidates for next state, usually using for loop and if condition for filtering
3) for each candidate, add it to state, and call backtrack function recursively and then pop it out of the state.

## Combination Sum II
[Combination Sum II](https://leetcode.com/problems/combination-sum-ii/)
```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        
        def backtrack(state, remain, cur_ind, solutions):

            if remain == 0:
                solutions.append(list(state))
                return

            for i in range(cur_ind, len(candidates)):

                if i > cur_ind and candidates[i] == candidates[i-1]:
                    continue

                if remain - candidates[i] < 0:
                    break

                state.append(candidates[i])
                backtrack(state, remain-candidates[i], i+1, solutions)
                state.pop()

        candidates.sort()
        state, solutions = [],[]
        backtrack(state, target, 0, solutions)
        return solutions
```
key points: use 'continue' and 'break' to prune the search tree for consice code.

## Word Search
[Word Search](https://leetcode.com/problems/word-search/)
```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        
        m, n = len(board), len(board[0])

        def dfs(state, visited, i, j):
            # print('state', state)
            if ''.join(state) == word:
            if (min(i, j) < 0
                or i >= m
                or j >= n
                or word[len(state)] != board[i][j]
                or (i,j) in visited):
                return False
            

            state.append(board[i][j])
            visited.add((i,j))
            found = False
            for move in [(-1,0), (1,0), (0,-1), (0,1)]:
                visited.add((i,j)+move)
                found = found or dfs(state, visited, i+move[0],j+move[1])
            state.pop()
            visited.remove((i,j))
            return found

        count = defaultdict(int, sum(map(Counter, board), Counter()))
        if count[word[0]] > count[word[-1]]:
            word = word[::-1]
            
        for i in range(m):
            for j in range(n):
                state = []
                visited = set()
                if dfs(state, visited, i, j):
                    return True
        return False
```
key point: understand difference between dfs and backtracking later

## Letter Combinations of a Phone Number
[Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)
```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:

        DSmap = {
            '2': ['a', 'b','c'],
            '3': ['d','e','f'],
            '4': ['g','h','i'],
            '5': ['j','k','l'],
            '6':['m','n','o'],
            '7': ['p','q','r','s'],
            '8': ['t','u','v'],
            '9': ['w','x','y','z']
        }
        
    
        def is_valid_state(state):
            if len(state) == len(digits):
                return True
                res.append(state.copy())
        
        def get_candidates(res, state):
            if len(state) == len(digits):
                return []
            else:
                d = digits[len(state)]
                return DSmap[d]

        def search(res, state):
            if is_valid_state(state):
                temp = state.copy()
                res.append(''.join(temp))

            for c in get_candidates(res, state):
                state.append(c)
                search(res, state)
                state.pop()

        if digits == "":
            return []
        else:
            res = []
            state = []
            search(res,state)
            return res
```
key point: this is an example show the backtracking template can really be helpful to finish the code quickly without thinking too much about the details, even though the solution may not be very concise.


## The following are examples of binary search

## Find Minimum in Rotated Sorted Array
[Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)
```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        l,r = 0, len(nums)-1
        while l<r:
            mid = (l+r)//2
            if nums[mid]>nums[r]:
                # left part is sorted
                l  = mid + 1
            else:
                # right part is sorted
                r = mid
        return nums[r]
```
Key points:
the appraoch is easy to understand, however, edge case can be very annoying
see https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/solutions/158940/beat-100-very-simple-python-very-detailed-explanation/ for more details

## Search in Rotated Sorted Array
[Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)
```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums)-1
        while l<=r:
            mid = (l+r)//2

            if nums[mid] == target:
                return mid

            if nums[l] <= nums[mid]:
                # left part is sorted
                if target>nums[mid] or target<nums[l]:
                    l = mid+1
                else:
                    r = mid - 1
            else:
                if target>nums[r] or target<nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1

        return -1
```
keypoint:
same as above, edge case is annoying








        


        






        



