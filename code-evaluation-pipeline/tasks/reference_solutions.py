# path: code-evaluation-pipeline/tasks/reference_solutions.py
"""
Reference solutions for coding tasks in the evaluation pipeline.

This file contains canonical implementations of common coding problems that serve as:
1. Ground truth solutions for evaluating code submissions
2. Baselines for performance and correctness comparison
3. Reference implementations for test case validation
4. Gold standards for edit-distance based similarity scoring

Each function represents a complete, correct solution to its respective problem
and follows Python best practices for style and efficiency. These solutions are
used by the evaluation pipeline to assess the quality of generated or submitted code.
"""


def fizzbuzz(n: int):
    for i in range(1, n + 1):
        if i % 15 == 0:
            print("FizzBuzz")
        elif i % 3 == 0:
            print("Fizz")
        elif i % 5 == 0:
            print("Buzz")
        else:
            print(i)


def is_palindrome(s: str) -> bool:
    s_clean = "".join(ch.lower() for ch in s if not ch.isspace())
    return s_clean == s_clean[::-1]


def binary_search(arr: list, target: int) -> int:
    """Binary search in sorted array."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


class ListNode:
    """Simple linked list node for testing."""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_linked_list(head):
    """Reverse a singly linked list."""
    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev


def validate_email(email: str) -> bool:
    """Basic email validation."""
    if not email or '@' not in email:
        return False
    parts = email.split('@')
    if len(parts) != 2:
        return False
    local, domain = parts
    if not local or not domain:
        return False
    if '.' not in domain:
        return False
    if domain.startswith('.') or domain.endswith('.'):
        return False
    return True