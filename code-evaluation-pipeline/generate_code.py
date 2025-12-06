# path: code-evaluation-pipeline/generate_code.py
"""
Generate code solutions for tasks using the OpenAI API (or a mock fallback).

This script:
- Reads tasks from tasks/tasks.json
- For each task, requests a solution from a model (if OPENAI_API_KEY is set)
- Saves generated code into code_solutions/{task_id}.py

NOTE:
- For safety and portability, this file uses a simple placeholder if
  OPENAI_API_KEY is not set. You can plug in the official OpenAI client
  and model of your choice.
"""

import json
from pathlib import Path
import os
from textwrap import dedent

# If you want to use the real OpenAI API, uncomment and configure:
# from openai import OpenAI
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_tasks(path: str = "tasks/tasks.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["tasks"]


def call_model(prompt: str) -> str:
    """
    Placeholder model caller.

    If OPENAI_API_KEY is set, you can replace this with a real OpenAI call.
    Otherwise, we return a naive but runnable solution for demo purposes.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Simple built-in fallback for demo.
        if "fizzbuzz" in prompt.lower():
            return dedent(
                """
                def fizzbuzz(n):
                    for i in range(1, n+1):
                        if i % 15 == 0:
                            print("FizzBuzz")
                        elif i % 3 == 0:
                            print("Fizz")
                        elif i % 5 == 0:
                            print("Buzz")
                        else:
                            print(i)
                """
            )
        elif "is_palindrome" in prompt.lower():
            return dedent(
                """
                def is_palindrome(s):
                    s_clean = ''.join(ch.lower() for ch in s if not ch.isspace())
                    return s_clean == s_clean[::-1]
                """
            )
        elif "binary_search" in prompt.lower():
            return dedent(
                """
                def binary_search(arr, target):
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
                """
            )
        elif "reverse_linked_list" in prompt.lower():
            return dedent(
                """
                def reverse_linked_list(head):
                    prev = None
                    current = head
                    while current:
                        next_node = current.next
                        current.next = prev
                        prev = current
                        current = next_node
                    return prev
                """
            )
        elif "validate_email" in prompt.lower():
            return dedent(
                """
                def validate_email(email):
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
                """
            )
        else:
            return "# TODO: model not configured, placeholder only\n"

    # Example call for real usage (you need to adapt to latest OpenAI SDK):
    # completion = client.chat.completions.create(
    #     model="gpt-4.1-mini",
    #     messages=[
    #         {"role": "system", "content": "You are an expert Python coding assistant."},
    #         {"role": "user", "content": prompt},
    #     ],
    #     temperature=0.2,
    # )
    # return completion.choices[0].message.content

    raise NotImplementedError("Replace call_model implementation with real OpenAI API call.")


def save_code(task_id: str, code: str, out_dir: str = "code_solutions"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / f"{task_id}.py"
    file_path.write_text(code, encoding="utf-8")
    print(f"✅ Wrote code for task {task_id} to {file_path}")


def main():
    tasks = load_tasks()
    for task in tasks:
        prompt = task["prompt"]
        print(f"Generating code for task: {task['id']}")
        code = call_model(prompt)
        save_code(task["id"], code)


if __name__ == "__main__":
    main()
