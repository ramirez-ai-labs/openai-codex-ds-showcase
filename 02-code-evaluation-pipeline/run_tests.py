# path: code-evaluation-pipeline/run_tests.py
"""
Run simple correctness checks for generated code.

For each task:
- Import the generated function from code_solutions/{task_id}.py
- Run basic test cases
- Report:
    - passed/failed
    - error type if failed
    - edit distance to reference solution (optional)
"""

import importlib.util
from pathlib import Path
import json
import traceback

from tasks import reference_solutions
from compute_edit_distance import edit_distance


def load_tasks(path: str = "tasks/tasks.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["tasks"]


def load_generated_function(task_id: str, entry_point: str):
    """
    Dynamically import a function from code_solutions/{task_id}.py.
    """
    module_path = Path("code_solutions") / f"{task_id}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"No generated code for task {task_id} at {module_path}")

    spec = importlib.util.spec_from_file_location(task_id, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    func = getattr(module, entry_point, None)
    if func is None:
        raise AttributeError(f"{entry_point} not found in {module_path}")
    return func, module_path.read_text(encoding="utf-8")


def run_fizzbuzz_tests(func):
    # Basic test: ensure it prints expected lines for n=15
    import io
    import sys

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        func(15)
    finally:
        sys.stdout = old_stdout
    output = buf.getvalue().strip().splitlines()
    expected = [
        "1",
        "2",
        "Fizz",
        "4",
        "Buzz",
        "Fizz",
        "7",
        "8",
        "Fizz",
        "Buzz",
        "11",
        "Fizz",
        "13",
        "14",
        "FizzBuzz",
    ]
    return output == expected


def run_is_palindrome_tests(func):
    tests = {
        "racecar": True,
        "RaceCar": True,
        "A man a plan a canal Panama": True,
        "hello": False,
        "": True,
        "abcba": True,
    }
    for s, expected in tests.items():
        if func(s) != expected:
            return False
    return True


def run_binary_search_tests(func):
    tests = [
        ([1, 2, 3, 4, 5], 3, 2),
        ([1, 2, 3, 4, 5], 1, 0),
        ([1, 2, 3, 4, 5], 5, 4),
        ([1, 2, 3, 4, 5], 6, -1),
        ([], 1, -1),
        ([1], 1, 0),
        ([1], 2, -1),
    ]
    for arr, target, expected in tests:
        result = func(arr, target)
        if result != expected:
            return False
    return True


def run_reverse_linked_list_tests(func):
    from tasks.reference_solutions import ListNode
    
    # Test 1: [1, 2, 3] -> [3, 2, 1]
    node1 = ListNode(1)
    node2 = ListNode(2)
    node3 = ListNode(3)
    node1.next = node2
    node2.next = node3
    
    result = func(node1)
    if result.val != 3 or result.next.val != 2 or result.next.next.val != 1 or result.next.next.next is not None:
        return False
    
    # Test 2: Single node
    single = ListNode(42)
    result = func(single)
    if result.val != 42 or result.next is not None:
        return False
    
    # Test 3: Empty (None)
    result = func(None)
    if result is not None:
        return False
    
    return True


def run_validate_email_tests(func):
    tests = {
        "user@example.com": True,
        "test.email@domain.co.uk": True,
        "invalid": False,
        "@domain.com": False,
        "user@": False,
        "user@domain": False,
        "user@.com": False,
        "": False,
    }
    for email, expected in tests.items():
        if func(email) != expected:
            return False
    return True


def main():
    tasks = load_tasks()
    results = []

    for task in tasks:
        task_id = task["id"]
        entry_point = task["entry_point"]
        print(f"🔍 Evaluating task: {task_id}")

        try:
            func, gen_code = load_generated_function(task_id, entry_point)
            # Get reference code (for edit distance)
            ref_code = getattr(reference_solutions, entry_point).__code__
            ref_source = reference_solutions.__dict__[entry_point].__code__
        except Exception as e:
            print(f"❌ Failed to import function for {task_id}: {e}")
            results.append(
                {
                    "task_id": task_id,
                    "status": "import_error",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "edit_distance": None,
                }
            )
            continue

        try:
            if task_id == "fizzbuzz":
                passed = run_fizzbuzz_tests(func)
            elif task_id == "is_palindrome":
                passed = run_is_palindrome_tests(func)
            elif task_id == "binary_search":
                passed = run_binary_search_tests(func)
            elif task_id == "reverse_linked_list":
                passed = run_reverse_linked_list_tests(func)
            elif task_id == "validate_email":
                passed = run_validate_email_tests(func)
            else:
                passed = False  # unknown task
        except Exception as e:
            print(f"❌ Runtime error for {task_id}: {e}")
            results.append(
                {
                    "task_id": task_id,
                    "status": "runtime_error",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "edit_distance": None,
                }
            )
            continue

        # For demo purposes, we compare generated code to reference code as strings.
        # In a real system you might normalize/format first.
        from inspect import getsource
        ref_source_str = getsource(getattr(reference_solutions, entry_point))
        edit = edit_distance(gen_code, ref_source_str)

        status = "passed" if passed else "failed_tests"
        print(f"  Result: {status}, edit_distance={edit}")
        results.append(
            {
                "task_id": task_id,
                "status": status,
                "error": None,
                "traceback": None,
                "edit_distance": edit,
            }
        )

    # Write summary report
    out_path = Path("code_eval_results.json")
    import json

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n📄 Wrote evaluation results to {out_path.resolve()}")


if __name__ == "__main__":
    main()
