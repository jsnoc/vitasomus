import os
import math


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class LinkList:
    def __init__(self):
        self.head=None

    def init_linklist(self, data):
        if len(data) == 0:
            return None
        self.head = ListNode(data[0])
        point = self.head
        for i in data[1:]:
            node = ListNode(i)
            point.next = node
            point = point.next
        return self.head

    def print_linklist(self, head):
        if head is None:
            return None
        point = head
        while point is not None:
            print(point.val, end='->')
            point = point.next
        print('None')


def test_link():
    l1 = [2, 4, 3]
    l2 = [5, 6, 4]
    link_list = LinkList()
    head1 = link_list.init_linklist(l1)
    head2 = link_list.init_linklist(l2)
    link_list.print_linklist(head1)
    link_list.print_linklist(head2)

#两数之和
def twoSum(nums=[2,7,11,15], target=9):
    '''
    :param:  [3,2,4], [3,3]
    :param: 6,6
    :return: [1,2], [0,1]
    '''
    check_table = {}
    for i, num in enumerate(nums):
        if target - num in check_table.keys():
            return [i, check_table[target - num]]
        check_table[num] = i

# 两数相加
def addTwoNumbers(l1_list=[], l2_list=[]):
    l1_list = [9,9,9]
    l2_list = [9,9,9,9,9]
    link_list = LinkList()
    l1 = link_list.init_linklist(l1_list)
    l2 = link_list.init_linklist(l2_list)
    newPoint = ListNode(l1.val + l2.val)
    rt, tp = newPoint, newPoint
    while (l1 and (l1.next != None)) or (l2 and (l2.next != None)) or (tp.val > 9):
        l1 = l1.next
        l2 = l2.next
        # l1, l2 = l1.next if l1 else l1, l2.next if l2 else l2
        tmpsum = (l1.val if l1 else 0) + (l2.val if l2 else 0)
        tp.next = ListNode(tp.val // 10 + tmpsum)
        tp.val %= 10
        tp = tp.next
    return rt

# 无重复最长子串
def lengthOfLongestSubstring(s='abcabcbd'):
    # Step 1: 定义需要维护的变量, 本题求最大长度，所以需要定义max_len, 该题又涉及去重，因此还需要一个哈希表
    max_len, hashmap = 0, {}

    # Step 2: 定义窗口的首尾端 (start, end)， 然后滑动窗口
    start = 0
    for end in range(len(s)):
        # Step 3
        # 更新需要维护的变量 (max_len, hashmap)
        # i.e. 把窗口末端元素加入哈希表，使其频率加1，并且更新最大长度
        hashmap[s[end]] = hashmap.get(s[end], 0) + 1
        if len(hashmap) == end - start + 1:
            max_len = max(max_len, end - start + 1)

        # Step 4:
        # 根据题意,  题目的窗口长度可变: 这个时候一般涉及到窗口是否合法的问题
        # 这时要用一个while去不断移动窗口左指针, 从而剔除非法元素直到窗口再次合法
        # 当窗口长度大于哈希表长度时候 (说明存在重复元素)，窗口不合法
        # 所以需要不断移动窗口左指针直到窗口再次合法, 同时提前更新需要维护的变量 (hashmap)
        while end - start + 1 > len(hashmap):
            head = s[start]
            hashmap[head] -= 1
            if hashmap[head] == 0:
                del hashmap[head]
            start += 1
    # Step 5: 返回答案 (最大长度)
    return max_len


#寻找两个正序数组的中位数
def findMedianSortedArrays(nums1=[1,2], nums2=[3,4]) -> float:
    # 2.50000
    def getKthElement(k):
        """
        - 主要思路：要找到第 k (k>1) 小的元素，那么就取 pivot1 = nums1[k/2-1] 和 pivot2 = nums2[k/2-1] 进行比较
        - 这里的 "/" 表示整除
        - nums1 中小于等于 pivot1 的元素有 nums1[0 .. k/2-2] 共计 k/2-1 个
        - nums2 中小于等于 pivot2 的元素有 nums2[0 .. k/2-2] 共计 k/2-1 个
        - 取 pivot = min(pivot1, pivot2)，两个数组中小于等于 pivot 的元素共计不会超过 (k/2-1) + (k/2-1) <= k-2 个
        - 这样 pivot 本身最大也只能是第 k-1 小的元素
        - 如果 pivot = pivot1，那么 nums1[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums1 数组
        - 如果 pivot = pivot2，那么 nums2[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums2 数组
        - 由于我们 "删除" 了一些元素（这些元素都比第 k 小的元素要小），因此需要修改 k 的值，减去删除的数的个数
        """

        index1, index2 = 0, 0
        while True:
            # 特殊情况
            if index1 == m:
                return nums2[index2 + k - 1]
            if index2 == n:
                return nums1[index1 + k - 1]
            if k == 1:
                return min(nums1[index1], nums2[index2])

            # 正常情况
            newIndex1 = min(index1 + k // 2 - 1, m - 1)
            newIndex2 = min(index2 + k // 2 - 1, n - 1)
            pivot1, pivot2 = nums1[newIndex1], nums2[newIndex2]
            if pivot1 <= pivot2:
                k -= newIndex1 - index1 + 1
                index1 = newIndex1 + 1
            else:
                k -= newIndex2 - index2 + 1
                index2 = newIndex2 + 1

    m, n = len(nums1), len(nums2)
    totalLength = m + n
    if totalLength % 2 == 1:
        return getKthElement((totalLength + 1) // 2)
    else:
        return (getKthElement(totalLength // 2) + getKthElement(totalLength // 2 + 1)) / 2

#最长回文子串
def longestPalindrome(s='babad'):
    # 示例
    # 1：
    #
    # 输入：s = "babad"
    # 输出："bab"
    # 解释："aba"
    # 同样是符合题意的答案。
    # 示例
    # 2：
    #
    # 输入：s = "cbbd"
    # 输出："bb"
    n = len(s)
    if n < 2:
        return s

    max_len = 1
    begin = 0
    # dp[i][j] 表示 s[i..j] 是否是回文串
    dp = [[False] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = True

    # 递推开始
    # 先枚举子串长度
    for L in range(2, n + 1):
        # 枚举左边界，左边界的上限设置可以宽松一些
        for i in range(n):
            # 由 L 和 i 可以确定右边界，即 j - i + 1 = L 得
            j = L + i - 1
            # 如果右边界越界，就可以退出当前循环
            if j >= n:
                break

            if s[i] != s[j]:
                dp[i][j] = False
            else:
                if j - i < 3:
                    dp[i][j] = True
                else:
                    dp[i][j] = dp[i + 1][j - 1]

            # 只要 dp[i][L] == true 成立，就表示子串 s[i..L] 是回文，此时记录回文长度和起始位置
            if dp[i][j] and j - i + 1 > max_len:
                max_len = j - i + 1
                begin = i
    return s[begin:begin + max_len]


#正则匹配
def isMatch(s="aa", p='a') -> bool:
    # 示例
    # 1：
    #
    # 输入：s = "aa", p = "a"
    # 输出：false
    # 解释："a"
    # 无法匹配
    # "aa"
    # 整个字符串。
    # 示例
    # 2:
    #
    # 输入：s = "aa", p = "a*"
    # 输出：true
    # 解释：因为
    # '*'
    # 代表可以匹配零个或多个前面的那一个元素, 在这里前面的元素就是
    # 'a'。因此，字符串
    # "aa"
    # 可被视为
    # 'a'
    # 重复了一次。
    # 示例 3：
    #
    # 输入：s = "ab", p = ".*"
    # 输出：true
    # 解释：".*"
    # 表示可匹配零个或多个（'*'）任意字符（'.'）。
    m, n = len(s), len(p)

    def matches(i: int, j: int) -> bool:
        if i == 0:
            return False
        if p[j - 1] == '.':
            return True
        return s[i - 1] == p[j - 1]

    f = [[False] * (n + 1) for _ in range(m + 1)]
    f[0][0] = True
    for i in range(m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                f[i][j] |= f[i][j - 2]
                if matches(i, j - 1):
                    f[i][j] |= f[i - 1][j]
            else:
                if matches(i, j):
                    f[i][j] |= f[i - 1][j - 1]
    return f[m][n]


#容器接水
def maxArea(height=[1,8,6,2,5,4,8,3,7]) -> int:
    l, r = 0, len(height) - 1
    ans = 0
    while l < r:
        area = min(height[l], height[r]) * (r - l)
        ans = max(ans, area)
        if height[l] <= height[r]:
            l += 1
        else:
            r -= 1
    return ans


#三数之和
def threeSum(nums=[-1,0,1,2,-1,-4]):
    # 示例
    # 1：
    #
    # 输入：nums = [-1, 0, 1, 2, -1, -4]
    # 输出：[[-1, -1, 2], [-1, 0, 1]]
    # 示例
    # 2：
    #
    # 输入：nums = []
    # 输出：[]
    # 示例
    # 3：
    #
    # 输入：nums = [0]
    # 输出：[]
    n = len(nums)
    nums.sort()
    ans = list()

    # 枚举 a
    for first in range(n):
        # 需要和上一次枚举的数不相同
        if first > 0 and nums[first] == nums[first - 1]:
            continue
        # c 对应的指针初始指向数组的最右端
        third = n - 1
        target = -nums[first]
        # 枚举 b
        for second in range(first + 1, n):
            # 需要和上一次枚举的数不相同
            if second > first + 1 and nums[second] == nums[second - 1]:
                continue
            # 需要保证 b 的指针在 c 的指针的左侧
            while second < third and nums[second] + nums[third] > target:
                third -= 1
            # 如果指针重合，随着 b 后续的增加
            # 就不会有满足 a+b+c=0 并且 b<c 的 c 了，可以退出循环
            if second == third:
                break
            if nums[second] + nums[third] == target:
                ans.append([nums[first], nums[second], nums[third]])

    return ans

#电话号码的字母组合
def letterCombinations(digits="23"):
    # 示例
    # 1：
    #
    # 输入：digits = "23"
    # 输出：["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"]
    # 示例
    # 2：
    #
    # 输入：digits = ""
    # 输出：[]
    # 示例
    # 3：
    #
    # 输入：digits = "2"
    # 输出：["a", "b", "c"]
    if not digits:
        return list()

    phoneMap = {
        "2": "abc",
        "3": "def",
        "4": "ghi",
        "5": "jkl",
        "6": "mno",
        "7": "pqrs",
        "8": "tuv",
        "9": "wxyz",
    }

    def backtrack(index: int):
        if index == len(digits):
            combinations.append("".join(combination))
        else:
            digit = digits[index]
            for letter in phoneMap[digit]:
                combination.append(letter)
                backtrack(index + 1)
                combination.pop()

    combination = list()
    combinations = list()
    backtrack(0)
    return combinations


#删除链表的倒数第 N 个结点
def removeNthFromEnd(head_list=[1, 2, 3, 4, 5], n=2) -> ListNode:
    # 输入：head = [1, 2, 3, 4, 5], n = 2
    # 输出：[1, 2, 3, 5]
    # 示例
    # 2：
    #
    # 输入：head = [1], n = 1
    # 输出：[]
    # 示例
    # 3：
    #
    # 输入：head = [1, 2], n = 1
    # 输出：[1]
    def getLength(head) -> int:
        length = 0
        while head:
            length += 1
            head = head.next
        return length
    lst = LinkList()
    head = lst.init_linklist(head_list)
    ori_list = []
    while head:
        ori_list.append(head.val)
        head = head.next
    ori_list.pop(-n)
    if len(ori_list) == 0:
        return None
    res = ListNode(ori_list[0])
    h = res
    for i in ori_list[1:]:
        h.next = ListNode(i)
        h = h.next
    return res

if __name__ == '__main__':
    print(removeNthFromEnd())
    pass