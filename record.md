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
## Car Fleet
[! Car Fleet](https://leetcode.com/problems/car-fleet/)
key point: this problem is complex but if we transform the problem as a time of arrival problem, it is much clear.
Hard to solve it if I have never seen this problem before.



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

## Linked List examples
 
## reverse linked list
[reverse linked list](https://leetcode.com/problems/reverse-linked-list/)

## merge two sorted lists
[merge two sorted lists](https://leetcode.com/problems/merge-two-sorted-lists/)

The above two easy problems are good examples to understand the structure.

## reorder list
[reorder list](https://leetcode.com/problems/reorder-list/)

appraoch 1: store the node in an array, more straightforward but requir O(n) space
```python
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        # store all node in am array
        node_arr = deque()
        dummy = head
        while head:
            node_arr.append(head)
            head = head.next
        head = dummy

        l, r = 0, len(node_arr)-1

        prev = head
        while l<r:
            node_arr[l].next = node_arr[r]
            l += 1

            if l == r:
                prev = node_arr[r]
                break

            node_arr[r].next = node_arr[l]
            r -= 1

            prev = node_arr[l]

        prev.next = None
```

approach 2: find the middle node, reverse the second half, merge the two lists.

Tip: using slow and fast pointer to find the middle node is a useful trick
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next


class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        # find the node that split the linked list half-half
        slow, fast = head, head.next

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        mid = slow
        # for odd, mid is the center, for even number, mid is that last node of the first half

        # reverse the second half
        # for consistency, we include the mid as the first half for odd number

        # break the part
        right = mid.next
        mid.next = None
        prev = None

        while right:
            tmp = right.next
            right.next = prev
            prev = right
            right = tmp

        # merge the two linked list
        left, right = head, prev
        while right:
            tmp1, tmp2 = left.next, right.next
            left.next = right
            right.next = tmp1
            left, right = tmp1, tmp2
```
## remove Nth node from end of list
[remove Nth node from end of list](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)

key point: 
1) using sliding window to find the node to be removed
2) using dummy node and return dummy.next to avoid edge case

```python
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:

        # get the length of linked list
        dummy = ListNode(0, next = head)
        
        l = dummy
        r = head
        # update r until offset n is meet
        while n>0:
            r = r.next
            n -= 1

        # slinding window
        while r:
            l = l.next
            r = r.next

        # skip
        l.next = l.next.next

        return dummy.next
```
## copy list with random pointer
[copy list with random pointer](https://leetcode.com/problems/copy-list-with-random-pointer/)
```python
class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        
        # initialize all the node with a hashmap to store the link relationship

        oldTocopy = {None: None}

        cur = head
        while cur:
            # create a new node
            temp_node = Node(cur.val)          
            oldTocopy[cur] = temp_node
            cur = cur.next

        cur = head
        while cur:
            copy = oldTocopy[cur]
            copy.next = oldTocopy[cur.next]
            copy.random = oldTocopy[cur.random]
            cur = cur.next
        
        return oldTocopy[head]
```
key point: using hashmap to link old node and new node, complete the task with two pass, the first pass is to create all the new node, the second pass is to assign relationship between the new node.

## add two numbers
[add two numbers](https://leetcode.com/problems/add-two-numbers/)
```python
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        
        dummy = ListNode()
        cur = dummy
        carry = 0
        while l1 or l2:
            v1 = l1.val if l1 else 0
            v2 = l2.val if l2 else 0
            if carry != 0:
                cur_val = v1 + v2 + carry
                # reset carry
                carry = 0
            else:
                cur_val = v1 + v2

            tmp_node = ListNode()
            if cur_val<10:
                tmp_node.val = cur_val
            else:
                tmp_node.val = cur_val % 10
                carry = 1
            cur.next = tmp_node
            cur = cur.next

            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None

        # edge case
        if carry == 1:
            cur.next = ListNode(carry)

        return dummy.next
```
key point: use a carry to store the value if the sum is greater than 10
trick:
```python
v1 = l1.val if l1 else 0
v2 = l2.val if l2 else 0
```
assign 0 to node that is None, nice to address edge case

similarily
```python
1 = l1.next if l1 else None
l2 = l2.next if l2 else None
```
move to None instead of next of next doesn't exist

## find duplicate number
[find duplicate number](https://leetcode.com/problems/find-the-duplicate-number/)
```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        # Floyd algorithm for loop detection
        slow, fast = 0, 0
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]] # move one step forward
            if slow == fast:
                # intersection found
                break

        slow2 = 0
        while True:
            slow = nums[slow]
            slow2 = nums[slow2]
            if slow == slow2:
                return slow
```
key point: Floyd algorithm for loop detection, just need to memorize the algorithm, no way to derive it during interview

## LRU Cache
[LRU Cache](https://leetcode.com/problems/lru-cache/)
```python
class LRUCache:

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.least_recent_key = None


    def get(self, key: int) -> int:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        else:
            return -1

        
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # update the value and move the key to last
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            self.cache[key] = value
        
        if len(self.cache) > self.capacity:
            self.cache.popitem(last = False)
        


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```
key point: this problem relay on doubled linked list, we need to understand remove and insert operation in double linked list.
Also, python collection orderedDict use linked list, it has function move_to_end and popitem, which is very useful for this problem.

##  Merge k Sorted Lists
[ Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        k = len(lists)
        if k == 0:
            return None
        if k == 1:
            return lists[0]

        while len(lists) > 1:
            mergedLists = []
            k = len(lists)
            for i in range(0,k,2):
                l1 = lists[i]
                l2 = lists[i+1] if (i+1)<k else None
                mergedLists.append(self.mergeTwoLists(l1,l2))
                # print(prev)
            lists = mergedLists
        return lists[0]





    def mergeTwoLists(self, list1: ListNode, list2: ListNode) -> ListNode:
        dummy = node = ListNode()

        while list1 and list2:
            if list1.val < list2.val:
                node.next = list1
                list1 = list1.next
            else:
                node.next = list2
                list2 = list2.next
            node = node.next

        node.next = list1 or list2

        return dummy.next
```
key point: once we can write merge two sorted linked list as a helper function, this problem becomes easy, since we can repeatedly merge two linked list

# examples of tree

## Invert Binary Tree
[Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/)
```python
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return

        temp = root.left
        root.left = root.right
        root.right = temp
        self.invertTree(root.left)
        self.invertTree(root.right)

        return root
```
key point: this is a recursive problem, we can use dfs to solve it, the key point is to swap left and right child of each node

## Maximum Depth of Binary Tree
[Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)

### dfs solution
```python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        return 1 + max(self.maxDepth(root.right), self.maxDepth(root.left))
```
### bfs solution
```python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        q = deque([root])
        res = 0
        while q:
            res += 1
            for _ in range(len(q)):
                node = q.popleft()
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
        return res
```

## Validate Binary Search Tree
[Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)
```python
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        # dfs
        def dfs(r, min_val, max_val):
            if not r:
                return True
            if r.val <= min_val or r.val >= max_val:
                return False
            else:
                return dfs(r.left, min_val, r.val) and dfs(r.right, r.val, max_val)
        
        min_val = float('-infinity')
        max_val = float('infinity')

        return dfs(root, min_val, max_val)
```
key point: this is a dfs problem, we need to pass min and max value to each node, and check if the node value is in the range of min and max value (i do it all by myself :))

## Kth Smallest Element in a BST
[Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)
```python
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        # in order dfs
        
        stack = [] # store the unprocessed node, use LIFO style, then we can achieve inorder dfs search, without recursive
        cur = root
        count = 0
        while cur or stack:
            # add the most left node in stack
            while cur:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            count += 1
            if count == k:
                return cur.val
            cur = cur.right
```
key point: this is an important problem, it basically show how we can transform a binary tree to a sorted array. the standard appraoch is to use inorder dfs traversal, which is a dfs approach. We utilize a stack to implement LIFO, and we use a while loop to traverse the tree. The key point is to understand how to use stack to implement dfs traversal.




















        


        






        




