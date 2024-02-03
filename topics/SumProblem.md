# Multiple variitions of Sum Problem

## 1. Two Sum
Simply use hashtable

## Two sum with sorted array
Use two pointers, one from left, one from right

## 3 Sum
General just use 2Sum with sorted array under another for loop.
The tricky part is how to avoid duplicate triplets.

to avoid duplicate triplets, 
1) for the outer loop i, we need to skip the same value as i-1, 
2) for the two pointer, we need to skip the same value as left+1 or right-1

## 3 Sum Closest
Same as 3Sum, just need to keep track of the closest sum

## 3 sum with multiplity
Use 3 sum, after we get the unique triplets, we use basic math to calculate the multiplicity and count the total number 
```
total_count = 0
        count_arr = Counter(arr)
        for trip in res:
            count_trip = Counter(trip)
            temp = 1
            for key,val in count_trip.items():
                temp *= math.comb(count_arr[key], val)
            total_count += temp

```

## 4 Sum
Same as 3Sum, just need to add another for loop. Similar to 3Sum, we need to avoid duplicate quadruplets.

## 4 Sum II
Use hashtable to store the sum of A and B, then use hashtable to store the sum of C and D, then check if the sum of A and B is in the hashtable of C and D.

