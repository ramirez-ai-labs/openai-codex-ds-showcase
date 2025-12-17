# Code Evaluation Report

This document summarizes the code-generation evaluation pipeline in the
`code-evaluation-pipeline/` directory.  
It describes **what is measured**, **why these metrics matter**, and **how to reproduce results**.

This type of analysis mirrors real workflows for **Data Scientists working on AI coding assistants** such as OpenAI Codex, GitHub Copilot, or Cursor.

---

## 🎯 Purpose of This Evaluation

The evaluation pipeline is designed to measure how well an AI code model performs on small programming tasks.

We assess:

- **Correctness** — Does the code run and pass tests?
- **Developer Effort** — How much editing would a developer need to fix it?
- **Failure Modes** — Why does the model fail?
- **Task-Level Comparison** — Which tasks are harder/easier for the model?

These metrics are widely used in LLM code-generation research and internal Codex workflows.

---

## 🧪 Tasks Evaluated

| Task ID         | Description                                  | Entry Point        |
|-----------------|----------------------------------------------|--------------------|
| `fizzbuzz`      | Print numbers 1..n with Fizz/Buzz rules      | `fizzbuzz()`       |
| `is_palindrome` | Check if a string is a palindrome            | `is_palindrome()`  |

Each task includes:

- A natural-language prompt  
- A reference implementation  
- A correctness test suite  

This allows **automated** evaluation without human intervention.

---

## 📏 Evaluation Metrics

### ✔ **Correctness Metrics**

| Metric             | Meaning |
|--------------------|---------|
| **Import success** | Can Python import the generated code without errors? |
| **Runtime success** | Does the function run without exceptions? |
| **Test pass rate** | Does the output match expected behavior? |

Correctness metrics reveal whether the model understands syntax, semantics, and control flow.

---

### ✔ **Quality Metrics**

#### **Edit Distance (Levenshtein)**  
Measures how different the generated code is from the reference solution.

- Lower distance → fewer edits needed  
- Higher distance → more developer intervention  

This is a proxy for **developer productivity** and is widely used in model evaluation systems.

---

### ✔ **Failure Mode Classification**

| Failure Mode     | Description |
|------------------|-------------|
| Import error     | Syntax error or missing definitions block import |
| Syntax error     | Bad Python code |
| Runtime error    | Code runs but crashes |
| Logical error    | Tests fail even though code executes |
| Hallucination    | Nonsensical or irrelevant output |

These patterns help diagnose **model weaknesses** and guide future improvements.

---

## 📊 Summary Template (Replace With Your Results)

| Task           | Status         | Edit Distance | Notes                     |
|----------------|----------------|---------------|---------------------------|
| fizzbuzz       | passed         | 5             | Output matches expected   |
| is_palindrome  | failed_tests   | 12            | Logical bug in edge cases |

**Overall Metrics:**

- **Pass rate:** 50%  
- **Average edit distance:** 8.5  
- **Most common failure:** Logical errors  

---

## 🔍 Error Analysis (Example)

Common issues observed:

- Missing edge-case handling  
- Off-by-one errors  
- Incorrect string cleaning  
- Hard-coded values  
- Partial or incomplete logic  

### Why this matters
Failure-mode analysis helps teams refine:

- decoding strategies  
- prompting techniques  
- training datasets  
- UI-level fallback behavior  

This mirrors real Codex DS responsibilities.

---

## 🚀 Future Extensions

Recommended improvements for version 2:

- Add more tasks (sorting, BFS/DFS, regex extraction)
- Add static analysis tools (`flake8`, `pylint`)
- Add cyclomatic complexity scoring
- Measure execution time + cost per passing task
- Compare multiple model versions (v1 vs v2)
- Add a leaderboard visualization inside the dashboard
- Record latency + tokens to approximate developer workflow friction

---

## 🛠 How to Reproduce the Evaluation

### Run the full pipeline
```bash
python app.py all
```

## Run steps individually

Generate code
```bash
python app.py generate
```

Evaluate
```bash
python app.py evaluate
```

---

### 📁 Output Files

Results are stored in:
```bash
code-evaluation-pipeline/code_solutions/
code-evaluation-pipeline/code_eval_results.json
dashboard/
```

These files contain:

- raw model outputs
- correctness labels
- metrics (edit distance, failure mode)
- data for dashboard visualization

---

### ✅ Final Notes

This evaluation report demonstrates:

- How you measure LLM code quality
- How you interpret results
- How you communicate findings
- How you think like a Codex data scientist
