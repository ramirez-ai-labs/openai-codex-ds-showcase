"""
Codex DS Showcase: NLP Analysis for Code Generation
====================================================

This module teaches **beginner-friendly NLP techniques** that a Data Scientist on
the OpenAI Codex team would use to evaluate and understand model-generated code.

Why NLP (Natural Language Processing) matters for code models:
--------------------------------------------------------------
AI coding models don’t just produce code—they process text prompts, generate
natural-language reasoning, and sometimes fail in ways that can only be
understood by analyzing text.

This module demonstrates how to analyze:

1. **Prompt Complexity**
   - How hard was the prompt?
   - Did it contain multiple requirements?
   - Longer / more complex prompts often lead to more mistakes.

2. **Semantic Similarity**
   - How close is the generated code to a reference solution?
   - Uses TF-IDF and cosine similarity (beginner-friendly techniques).

3. **Code-as-Text Features**
   - Loops, conditionals, nesting, function definitions.
   - Helps understand complexity of generated code.

4. **Failure Mode Classification Through NLP**
   - Read error messages and categorize failures (syntax, type, logical, etc.).

5. **Prompt–Code Alignment**
   - Does the code actually address the user's instructions?
   - Measures requirement fulfillment.

Together, these allow Codex Data Scientists to:
- Diagnose weak spots in the model
- Understand why some prompts fail
- Improve dataset design and model behavior
- Conduct deeper evaluation beyond test pass/fail
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------
# Optional TF-IDF Tools for Semantic Similarity
# These are simple NLP methods useful for beginners.
# --------------------------------------------------------------
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ==============================================================
# DATA LOADING HELPERS
# ==============================================================

def load_evaluation_results(results_path: str = None) -> List[Dict]:
    """
    Load JSON results from the code evaluation pipeline.

    Beginners:
    ----------
    These results include model outputs, errors, status (passed/failed),
    and metadata that we analyze with NLP.
    """
    if results_path is None:
        root = Path(__file__).parent.parent
        results_path = root / "code-evaluation-pipeline" / "code_eval_results.json"
    
    if not Path(results_path).exists():
        raise FileNotFoundError(f"Evaluation results not found at {results_path}")
    
    with open(results_path, "r") as f:
        return json.load(f)


def load_tasks(tasks_path: str = None) -> List[Dict]:
    """
    Load the definitions for each coding task, including:
    - The prompt text (NLP input)
    - The task ID
    - The reference solution
    """
    if tasks_path is None:
        root = Path(__file__).parent.parent
        tasks_path = root / "code-evaluation-pipeline" / "tasks" / "tasks.json"
    
    with open(tasks_path, "r") as f:
        data = json.load(f)
        return data["tasks"]


# ==============================================================
# PROMPT ANALYSIS — How difficult is the prompt?
# ==============================================================

def analyze_prompt_complexity(prompt: str) -> Dict:
    """
    Analyze basic NLP features of a prompt.

    Beginners:
    ----------
    The goal is to measure how *complex* the user instructions are.
    More complex prompts → harder for the model → more failures.

    We measure:
    - Word count
    - Number of sentences
    - Keywords that imply requirements (return, print, sort, search, etc.)
    - Whether the prompt includes conditions or loops
    - Whether multiple tasks are mixed in the same prompt
    """
    words = prompt.split()
    sentences = re.split(r'[.!?]+', prompt)
    sentences = [s.strip() for s in sentences if s.strip()]

    requirement_keywords = [
        'function', 'return', 'print', 'check', 'validate',
        'find', 'search', 'sort', 'reverse', 'calculate'
    ]
    found_keywords = [kw for kw in requirement_keywords if kw in prompt.lower()]

    has_conditions = any(word in prompt.lower() for word in ['if', 'when', 'unless', 'except'])
    has_iteration = any(word in prompt.lower() for word in ['loop', 'for', 'while', 'each', 'all'])
    has_multiple_requirements = prompt.count(',') > 2 or len(sentences) > 2

    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_words_per_sentence": len(words) / max(len(sentences), 1),
        "requirement_keywords": len(found_keywords),
        "has_conditions": has_conditions,
        "has_iteration": has_iteration,
        "has_multiple_requirements": has_multiple_requirements,

        # Simple beginner-friendly complexity metric
        "complexity_score": (
            len(words) * 0.1 +
            len(sentences) * 2 +
            len(found_keywords) * 3 +
            (1 if has_conditions else 0) * 5 +
            (1 if has_iteration else 0) * 5 +
            (1 if has_multiple_requirements else 0) * 3
        )
    }


# ==============================================================
# CODE FEATURE ANALYSIS — Treat code as text
# ==============================================================

def extract_code_features(code: str) -> Dict:
    """
    Extract NLP-like features from code.

    Beginners:
    ----------
    Even though this is *code*, we analyze it like text to understand structure:

    - Does it use loops?
    - Does it have conditionals?
    - How long is the code?
    - Does it define a function?
    - How nested are blocks? (indentation → complexity)
    """
    if not code or code.strip().startswith("#"):
        return {"is_empty": True}

    lines = [l.strip() for l in code.split('\n')
             if l.strip() and not l.strip().startswith('#')]

    # Basic code structure detection (text-based, beginner friendly)
    has_loops = bool(re.search(r'\b(for|while)\s+', code))
    has_conditionals = bool(re.search(r'\bif\s+', code))
    has_functions = bool(re.search(r'\bdef\s+\w+', code))
    has_returns = bool(re.search(r'\breturn\s+', code))
    has_imports = bool(re.search(r'\bimport\s+', code))

    # Indentation depth approximates complexity
    nesting_level = max([len(re.findall(r'^\s+', line)) for line in lines] + [0])
    avg_line_length = np.mean([len(line) for line in lines]) if lines else 0

    return {
        "line_count": len(lines),
        "has_loops": has_loops,
        "has_conditionals": has_conditionals,
        "has_functions": has_functions,
        "has_returns": has_returns,
        "has_imports": has_imports,
        "nesting_level": nesting_level,
        "avg_line_length": avg_line_length,
        "code_length": len(code),
        "token_estimate": len(code.split())  # Rough, beginner-friendly token estimate
    }


# ==============================================================
# SEMANTIC SIMILARITY — How close is generated code to reference?
# ==============================================================

def compute_semantic_similarity(code1: str, code2: str) -> float:
    """
    Compare two code snippets using TF-IDF + cosine similarity.

    Beginners:
    ----------
    TF-IDF = Term Frequency–Inverse Document Frequency  
    → Turns text into numerical vectors

    Cosine similarity = measures how similar two vectors are  
    → 1.0 = identical  
    → 0.0 = unrelated

    This is a simple and beginner-friendly way to measure similarity.
    """
    if not HAS_SKLEARN:
        return 0.0

    # Clean code for better comparison
    def normalize_code(code):
        code = re.sub(r'#.*', '', code)  # remove comments
        code = ' '.join(code.split())    # collapse whitespace
        return code.lower()

    norm1 = normalize_code(code1)
    norm2 = normalize_code(code2)

    if not norm1 or not norm2:
        return 0.0

    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=100)

    try:
        vectors = vectorizer.fit_transform([norm1, norm2])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return float(similarity)
    except:
        return 0.0


# ==============================================================
# FAILURE MODE CLASSIFICATION — Reading error messages with NLP
# ==============================================================

def classify_failure_mode_nlp(error_msg: str, code: str, test_result: str) -> str:
    """
    Classify the type of failure using simple NLP.

    Beginners:
    ----------
    This function reads the error message and predicts:
        - syntax_error
        - name_error
        - type_error
        - index_error
        - attribute_error
        - logical_error (tests fail)
        - incomplete (TODO, pass, not implemented)
        - unknown_error
    """
    if not error_msg or error_msg == "None":
        error_msg = ""
    
    error_lower = error_msg.lower()
    code_lower = code.lower() if code else ""

    if any(k in error_lower for k in ['syntax', 'invalid syntax', 'unexpected']):
        return "syntax_error"
    elif any(k in error_lower for k in ['nameerror', 'not defined']):
        return "name_error"
    elif any(k in error_lower for k in ['typeerror', 'type error']):
        return "type_error"
    elif any(k in error_lower for k in ['indexerror', 'out of range']):
        return "index_error"
    elif any(k in error_lower for k in ['attributeerror', 'has no attribute']):
        return "attribute_error"
    elif test_result == "failed_tests" and not error_msg:
        return "logical_error"
    elif any(k in code_lower for k in ['todo', 'not implemented', 'pass']):
        return "incomplete"
    else:
        return "unknown_error"


# ==============================================================
# PROMPT–CODE ALIGNMENT — Does the model follow instructions?
# ==============================================================

def analyze_prompt_code_alignment(prompt: str, code: str) -> Dict:
    """
    Compare what the prompt asks for vs. what the code actually does.

    Beginners:
    ----------
    Example:
      Prompt: "Return True if string is a palindrome"
      Code:   prints result but does NOT return → misaligned

    We extract requirements from the prompt and check whether:
    - return behavior exists
    - print output exists
    - validation logic exists
    - palindrome logic exists
    - fizzbuzz logic exists
    """
    prompt_lower = prompt.lower()
    code_lower = code.lower() if code else ""

    requirements = []

    # Simple requirement extraction using keyword detection
    if 'return' in prompt_lower:
        requirements.append('return_value')
    if 'print' in prompt_lower:
        requirements.append('print_output')
    if 'check' in prompt_lower or 'validate' in prompt_lower:
        requirements.append('validation')
    if 'palindrome' in prompt_lower:
        requirements.append('palindrome_check')
    if 'fizz' in prompt_lower or 'buzz' in prompt_lower:
        requirements.append('fizzbuzz_logic')
    if 'reverse' in prompt_lower:
        requirements.append('reverse_operation')

    # Does the generated code satisfy the requirements?
    requirement_coverage = {}
    for req in requirements:
        if req == 'return_value':
            requirement_coverage[req] = 'return' in code_lower
        elif req == 'print_output':
            requirement_coverage[req] = 'print' in code_lower
        elif req == 'validation':
            requirement_coverage[req] = 'if' in code_lower or 'assert' in code_lower
        elif req == 'palindrome_check':
            requirement_coverage[req] = (
                'palindrome' in code_lower or
                'reverse' in code_lower or 
                '==' in code
            )
        elif req == 'fizzbuzz_logic':
            requirement_coverage[req] = (
                'fizz' in code_lower or
                'buzz' in code_lower or
                '%' in code or
                'mod' in code_lower
            )
        elif req == 'reverse_operation':
            requirement_coverage[req] = (
                'reverse' in code_lower or
                '[::-1]' in code or
                'reversed' in code_lower
            )

    coverage_rate = sum(requirement_coverage.values()) / max(len(requirements), 1)

    return {
        "requirements_found": len(requirements),
        "requirements_covered": sum(requirement_coverage.values()),
        "coverage_rate": coverage_rate,
        "requirement_details": requirement_coverage
    }


# ==============================================================
# MAIN NLP ANALYSIS PIPELINE
# ==============================================================

def run_nlp_analysis():
    """
    Run NLP analysis on all generated code solutions.

    Beginners:
    ----------
    This function stitches together all the analyses:
      ✓ prompt complexity
      ✓ code feature extraction
      ✓ failure mode classification
      ✓ prompt–code alignment
      ✓ prints insights and summary statistics
    """
    print("\n" + "="*70)
    print("NLP ANALYSIS: Code Generation Evaluation")
    print("="*70)

    results = load_evaluation_results()
    tasks = load_tasks()

    # Create lookup for prompts by task ID
    task_lookup = {task["id"]: task for task in tasks}

    # Load generated code files
    root = Path(__file__).parent.parent
    code_solutions_dir = root / "code-evaluation-pipeline" / "code_solutions"

    nlp_results = []

    for result in results:
        task_id = result["task_id"]
        task = task_lookup.get(task_id, {})
        prompt = task.get("prompt", "")

        code_file = code_solutions_dir / f"{task_id}.py"
        generated_code = code_file.read_text() if code_file.exists() else ""

        # ---------------------------------------------------------
        # Apply all NLP analyses
        # ---------------------------------------------------------
        prompt_analysis = analyze_prompt_complexity(prompt)
        code_features = extract_code_features(generated_code)
        failure_mode = classify_failure_mode_nlp(
            result.get("error", ""),
            generated_code,
            result.get("status", "")
        )
        alignment = analyze_prompt_code_alignment(prompt, generated_code)

        nlp_results.append({
            "task_id": task_id,
            "status": result.get("status", ""),
            "prompt_complexity": prompt_analysis["complexity_score"],
            "prompt_word_count": prompt_analysis["word_count"],
            "code_line_count": code_features.get("line_count", 0),
            "code_has_loops": code_features.get("has_loops", False),
            "code_has_conditionals": code_features.get("has_conditionals", False),
            "failure_mode_nlp": failure_mode,
            "requirement_coverage": alignment["coverage_rate"],
            "requirements_found": alignment["requirements_found"],
        })

    df = pd.DataFrame(nlp_results)

    # ---------------------------------------------------------
    # PRINT SUMMARY STATISTICS
    # ---------------------------------------------------------

    print("\n📊 PROMPT COMPLEXITY")
    print("-"*70)
    print(f"Average prompt complexity: {df['prompt_complexity'].mean():.1f}")
    print(f"Average word count: {df['prompt_word_count'].mean():.1f}")

    print("\n📝 CODE FEATURES")
    print("-"*70)
    print(f"Average code length: {df['code_line_count'].mean():.1f} lines")
    print(f"Tasks using loops: {df['code_has_loops'].sum()}")
    print(f"Tasks using conditionals: {df['code_has_conditionals'].sum()}")

    print("\n🎯 REQUIREMENT ALIGNMENT")
    print("-"*70)
    print(f"Average coverage rate: {df['requirement_coverage'].mean():.1%}")

    print("\n🔍 FAILURE MODES (NLP-based)")
    print("-"*70)
    for mode, count in df['failure_mode_nlp'].value_counts().items():
        print(f"{mode}: {count}")

    print("\n" + "="*70)
    print("KEY INSIGHTS FOR BEGINNERS")
    print("="*70)
    print("1. NLP reveals prompt difficulty before the model even runs.")
    print("2. Requirement coverage is a semantic metric — not based on tests.")
    print("3. Failure mode classification helps debug model behavior faster.")
    print("4. Code-as-text features highlight patterns in generated code.")
    print("="*70 + "\n")

    return df


# ==============================================================
# ENTRY POINT
# ==============================================================

def main():
    """Run NLP analysis and save results."""
    try:
        df = run_nlp_analysis()

        output_path = Path(__file__).parent / "nlp_analysis_results.csv"
        df.to_csv(output_path, index=False)
        print(f"✅ Saved NLP analysis results to {output_path}")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nRun the code evaluation pipeline first:")
        print("   python app.py generate")
        print("   python app.py evaluate")


if __name__ == "__main__":
    main()
