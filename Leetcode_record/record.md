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
