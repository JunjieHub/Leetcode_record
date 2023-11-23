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
