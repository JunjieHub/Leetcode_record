This file collects probloems that involve tricks of logic taht may be confusing and easy to make mistakes.

## contain with most water
[11. Container With Most Water](https://leetcode.com/problems/container-with-most-water/)

You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return the maximum amount of water a container can store.

Notice that you may not slant the container.

Trick:
when we update the two pointer, what we actually need to compare is the height of two current pointer, not the height of the two pointer that we just moved. We always preserve the larger height side because if we move to the smaller height side, the valume of the container will always decrease, which means we will never find a larger container.

One thing i tend to make mistake is to compare the volume after move left and move right pointer, that is wrong, we want to make decision that can possiblly lead to larger volume of cintained water

## Minimum Number of Coins to be Added
You are given a 0-indexed integer array coins, representing the values of the coins available, and an integer target.

An integer x is obtainable if there exists a subsequence of coins that sums to x.

Return the minimum number of coins of any value that need to be added to the array so that every integer in the range [1, target] is obtainable.

A subsequence of an array is a new non-empty array that is formed from the original array by deleting some (possibly none) of the elements without disturbing the relative positions of the remaining elements.

 

Example 1:

Input: coins = [1,4,10], target = 19
Output: 2
Explanation: We need to add coins 2 and 8. The resulting array will be [1,2,4,8,10].
It can be shown that all integers from 1 to 19 are obtainable from the resulting array, and that 2 is the minimum number of coins that need to be added to the array. 

This is actually a hard problem and require greedy algorithm to solve. Suppose our current range is [1,cur] once we add a new coin with value c, our range becomes [1, cur+c].

we want to continuously increase the reachable value until it cover the target
1) if the current coin <= reachable+1, adding the coin help increasing the reachable
2) if not, we must add the reachable+1 as a new coin, we need to update the reachabel accordingly


## Count Complete Substrings
You are given a string word and an integer k.

A substring s of word is complete if:

Each character in s occurs exactly k times.
The difference between two adjacent characters is at most 2. That is, for any two adjacent characters c1 and c2 in s, the absolute difference in their positions in the alphabet is at most 2.
Return the number of complete substrings of word.

A substring is a non-empty contiguous sequence of characters in a string.

 

Example 1:

Input: word = "igigee", k = 2
Output: 3
Explanation: The complete substrings where each character appears exactly twice and the difference between adjacent characters is at most 2 are: igigee, igigee, igigee.
Example 2:

Input: word = "aaabbbccc", k = 3
Output: 6
Explanation: The complete substrings where each character appears exactly three times and the difference between adjacent characters is at most 2 are: aaabbbccc, aaabbbccc, aaabbbccc, aaabbbccc, aaabbbccc, aaabbbccc.

trick: when deal with problem that has a bottom limit, e.g. lower case letter only have 26, so we can deal with (1,26) unique number of letter which will simplify the problem a lot, especially when design the updating rule of slinding window.


