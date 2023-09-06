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
