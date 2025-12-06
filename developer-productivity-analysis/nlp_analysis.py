"""
Codex DS Showcase: NLP Analysis for Code Generation
====================================================

This module demonstrates NLP skills relevant to evaluating AI code generation:

1. **Prompt Analysis**: Understanding how prompt characteristics affect code quality
2. **Semantic Similarity**: Comparing generated code to references using embeddings
3. **Code-to-Text Understanding**: Analyzing how well code matches natural language intent
4. **Token Analysis**: Understanding model behavior through token-level metrics
5. **Failure Mode NLP**: Using text analysis to classify and understand failures

This shows how a Codex Data Scientist would use NLP techniques to:
- Understand model behavior beyond just correctness
- Analyze prompt effectiveness
- Measure semantic quality (not just syntactic)
- Diagnose failure modes through text analysis
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

# For semantic similarity (if available)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def load_evaluation_results(results_path: str = None) -> List[Dict]:
    """Load code evaluation results."""
    if results_path is None:
        root = Path(__file__).parent.parent
        results_path = root / "code-evaluation-pipeline" / "code_eval_results.json"
    
    if not Path(results_path).exists():
        raise FileNotFoundError(f"Evaluation results not found at {results_path}")
    
    with open(results_path, "r") as f:
        return json.load(f)


def load_tasks(tasks_path: str = None) -> List[Dict]:
    """Load task definitions with prompts."""
    if tasks_path is None:
        root = Path(__file__).parent.parent
        tasks_path = root / "code-evaluation-pipeline" / "tasks" / "tasks.json"
    
    with open(tasks_path, "r") as f:
        data = json.load(f)
        return data["tasks"]


def analyze_prompt_complexity(prompt: str) -> Dict:
    """
    Analyze prompt characteristics that might affect code generation quality.
    
    NLP techniques:
    - Token counting
    - Readability metrics
    - Keyword extraction
    - Requirement complexity
    """
    # Basic text statistics
    words = prompt.split()
    sentences = re.split(r'[.!?]+', prompt)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Count requirements/keywords
    requirement_keywords = ['function', 'return', 'print', 'check', 'validate', 
                            'find', 'search', 'sort', 'reverse', 'calculate']
    found_keywords = [kw for kw in requirement_keywords if kw in prompt.lower()]
    
    # Estimate complexity
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
        "complexity_score": (
            len(words) * 0.1 +
            len(sentences) * 2 +
            len(found_keywords) * 3 +
            (1 if has_conditions else 0) * 5 +
            (1 if has_iteration else 0) * 5 +
            (1 if has_multiple_requirements else 0) * 3
        )
    }


def extract_code_features(code: str) -> Dict:
    """
    Extract NLP-relevant features from code.
    
    Analyzes code as text to understand:
    - Code structure through keywords
    - Complexity indicators
    - Style patterns
    """
    if not code or code.strip().startswith("#"):
        return {"is_empty": True}
    
    # Count code elements
    lines = [l.strip() for l in code.split('\n') if l.strip() and not l.strip().startswith('#')]
    
    # Pattern matching
    has_loops = bool(re.search(r'\b(for|while)\s+', code))
    has_conditionals = bool(re.search(r'\bif\s+', code))
    has_functions = bool(re.search(r'\bdef\s+\w+', code))
    has_returns = bool(re.search(r'\breturn\s+', code))
    has_imports = bool(re.search(r'\bimport\s+', code))
    
    # Complexity indicators
    nesting_level = max([len(re.findall(r'^\s+', line)) for line in lines] + [0])
    avg_line_length = np.mean([len(line) for line in lines]) if lines else 0
    
    # Code style patterns
    uses_list_comp = bool(re.search(r'\[.*for.*in.*\]', code))
    uses_lambda = bool(re.search(r'\blambda\s+', code))
    
    return {
        "line_count": len(lines),
        "has_loops": has_loops,
        "has_conditionals": has_conditionals,
        "has_functions": has_functions,
        "has_returns": has_returns,
        "has_imports": has_imports,
        "nesting_level": nesting_level,
        "avg_line_length": avg_line_length,
        "uses_list_comp": uses_list_comp,
        "uses_lambda": uses_lambda,
        "code_length": len(code),
        "token_estimate": len(code.split())  # Rough token estimate
    }


def compute_semantic_similarity(code1: str, code2: str) -> float:
    """
    Compute semantic similarity between two code snippets using TF-IDF.
    
    This is a simple approach - in production, you might use:
    - Code embeddings (CodeBERT, GraphCodeBERT)
    - AST-based similarity
    - Execution trace similarity
    """
    if not HAS_SKLEARN:
        return 0.0
    
    # Normalize code (remove extra whitespace, normalize identifiers)
    def normalize_code(code):
        # Remove comments
        code = re.sub(r'#.*', '', code)
        # Normalize whitespace
        code = ' '.join(code.split())
        return code.lower()
    
    norm1 = normalize_code(code1)
    norm2 = normalize_code(code2)
    
    if not norm1 or not norm2:
        return 0.0
    
    # Use character n-grams for code similarity
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=100)
    try:
        vectors = vectorizer.fit_transform([norm1, norm2])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return float(similarity)
    except:
        return 0.0


def classify_failure_mode_nlp(error_msg: str, code: str, test_result: str) -> str:
    """
    Use NLP to classify failure modes from error messages and code.
    
    This demonstrates text classification skills for understanding model failures.
    """
    if not error_msg or error_msg == "None":
        error_msg = ""
    
    error_lower = error_msg.lower()
    code_lower = code.lower() if code else ""
    
    # Pattern matching for error types
    if any(keyword in error_lower for keyword in ['syntax', 'invalid syntax', 'unexpected']):
        return "syntax_error"
    elif any(keyword in error_lower for keyword in ['nameerror', 'not defined', 'undefined']):
        return "name_error"
    elif any(keyword in error_lower for keyword in ['typeerror', 'type error']):
        return "type_error"
    elif any(keyword in error_lower for keyword in ['indexerror', 'index out', 'out of range']):
        return "index_error"
    elif any(keyword in error_lower for keyword in ['attributeerror', 'has no attribute']):
        return "attribute_error"
    elif test_result == "failed_tests" and not error_msg:
        return "logical_error"
    elif any(keyword in code_lower for keyword in ['todo', 'not implemented', 'pass']):
        return "incomplete"
    else:
        return "unknown_error"


def analyze_prompt_code_alignment(prompt: str, code: str) -> Dict:
    """
    Analyze how well the generated code aligns with the prompt requirements.
    
    Uses NLP to extract requirements from prompt and check if code addresses them.
    """
    prompt_lower = prompt.lower()
    code_lower = code.lower() if code else ""
    
    # Extract requirements from prompt
    requirements = []
    if 'return' in prompt_lower or 'returns' in prompt_lower:
        requirements.append('return_value')
    if 'print' in prompt_lower:
        requirements.append('print_output')
    if 'check' in prompt_lower or 'validate' in prompt_lower:
        requirements.append('validation')
    if 'palindrome' in prompt_lower:
        requirements.append('palindrome_check')
    if 'fizz' in prompt_lower or 'buzz' in prompt_lower:
        requirements.append('fizzbuzz_logic')
    if 'search' in prompt_lower:
        requirements.append('search_algorithm')
    if 'reverse' in prompt_lower:
        requirements.append('reverse_operation')
    
    # Check if code addresses requirements
    requirement_coverage = {}
    for req in requirements:
        if req == 'return_value':
            requirement_coverage[req] = 'return' in code_lower
        elif req == 'print_output':
            requirement_coverage[req] = 'print' in code_lower
        elif req == 'validation':
            requirement_coverage[req] = 'if' in code_lower or 'assert' in code_lower
        elif req == 'palindrome_check':
            requirement_coverage[req] = 'palindrome' in code_lower or 'reverse' in code_lower or '==' in code
        elif req == 'fizzbuzz_logic':
            requirement_coverage[req] = ('fizz' in code_lower or 'buzz' in code_lower or 
                                         '%' in code or 'mod' in code_lower)
        elif req == 'search_algorithm':
            requirement_coverage[req] = 'search' in code_lower or 'find' in code_lower or 'index' in code_lower
        elif req == 'reverse_operation':
            requirement_coverage[req] = 'reverse' in code_lower or '[::-1]' in code or 'reversed' in code_lower
    
    coverage_rate = sum(requirement_coverage.values()) / max(len(requirements), 1)
    
    return {
        "requirements_found": len(requirements),
        "requirements_covered": sum(requirement_coverage.values()),
        "coverage_rate": coverage_rate,
        "requirement_details": requirement_coverage
    }


def run_nlp_analysis():
    """Run comprehensive NLP analysis on code evaluation results."""
    print("\n" + "="*70)
    print("NLP ANALYSIS: Code Generation Evaluation")
    print("="*70)
    
    # Load data
    results = load_evaluation_results()
    tasks = load_tasks()
    
    # Create task lookup
    task_lookup = {task["id"]: task for task in tasks}
    
    # Load generated code
    root = Path(__file__).parent.parent
    code_solutions_dir = root / "code-evaluation-pipeline" / "code_solutions"
    
    nlp_results = []
    
    for result in results:
        task_id = result["task_id"]
        task = task_lookup.get(task_id, {})
        prompt = task.get("prompt", "")
        
        # Load generated code
        code_file = code_solutions_dir / f"{task_id}.py"
        generated_code = ""
        if code_file.exists():
            generated_code = code_file.read_text(encoding="utf-8")
        
        # NLP analyses
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
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(nlp_results)
    
    # Print results
    print("\n📊 PROMPT COMPLEXITY ANALYSIS")
    print("-" * 70)
    print(f"Average prompt complexity: {df['prompt_complexity'].mean():.1f}")
    print(f"Average word count: {df['prompt_word_count'].mean():.1f}")
    print("\nComplexity by task:")
    for task_id in df['task_id'].unique():
        task_df = df[df['task_id'] == task_id]
        print(f"  {task_id}: {task_df['prompt_complexity'].iloc[0]:.1f} complexity, "
              f"{task_df['prompt_word_count'].iloc[0]} words")
    
    print("\n📝 CODE FEATURE ANALYSIS")
    print("-" * 70)
    print(f"Average code length: {df['code_line_count'].mean():.1f} lines")
    print(f"Tasks with loops: {df['code_has_loops'].sum()}/{len(df)}")
    print(f"Tasks with conditionals: {df['code_has_conditionals'].sum()}/{len(df)}")
    
    print("\n🎯 PROMPT-CODE ALIGNMENT")
    print("-" * 70)
    print(f"Average requirement coverage: {df['requirement_coverage'].mean():.1%}")
    print("\nCoverage by task:")
    for task_id in df['task_id'].unique():
        task_df = df[df['task_id'] == task_id]
        print(f"  {task_id}: {task_df['requirement_coverage'].iloc[0]:.1%} coverage "
              f"({task_df['requirements_found'].iloc[0]} requirements)")
    
    print("\n🔍 FAILURE MODE CLASSIFICATION (NLP-based)")
    print("-" * 70)
    failure_counts = df['failure_mode_nlp'].value_counts()
    for mode, count in failure_counts.items():
        print(f"  {mode}: {count}")
    
    print("\n📈 CORRELATION ANALYSIS")
    print("-" * 70)
    if len(df) > 1:
        # Correlation between prompt complexity and success
        passed = df['status'] == 'passed'
        if passed.sum() > 0:
            complexity_passed = df[passed]['prompt_complexity'].mean()
            complexity_failed = df[~passed]['prompt_complexity'].mean()
            print(f"Average complexity (passed): {complexity_passed:.1f}")
            print(f"Average complexity (failed): {complexity_failed:.1f}")
            
            coverage_passed = df[passed]['requirement_coverage'].mean()
            coverage_failed = df[~passed]['requirement_coverage'].mean()
            print(f"\nRequirement coverage (passed): {coverage_passed:.1%}")
            print(f"Requirement coverage (failed): {coverage_failed:.1%}")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    print("1. NLP analysis reveals prompt characteristics that affect code quality")
    print("2. Requirement coverage is a semantic metric (beyond syntax)")
    print("3. Text classification helps understand failure modes")
    print("4. Code features extracted through NLP reveal generation patterns")
    print("="*70 + "\n")
    
    return df


def main():
    """Main entry point."""
    try:
        df = run_nlp_analysis()
        
        # Save results
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

