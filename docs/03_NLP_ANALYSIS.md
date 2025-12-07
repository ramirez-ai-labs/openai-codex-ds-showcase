# NLP Analysis for Code Generation

This document explains how **Natural Language Processing (NLP)** skills are demonstrated in this repository, relevant to the Codex Data Scientist role.

## Why NLP Matters for Code Generation

For a Data Scientist working on AI code generation tools, NLP skills are essential because:

1. **Code is Text**: Generated code needs to be analyzed as text
2. **Prompts are Natural Language**: Understanding prompt characteristics affects code quality
3. **Semantic Understanding**: Beyond syntax, we need to measure semantic correctness
4. **Failure Analysis**: Error messages and code need text classification
5. **Code-to-Text Alignment**: Measuring if code matches natural language intent

---

## NLP Techniques Demonstrated

### 1. **Prompt Analysis** (`analyze_prompt_complexity`)

**What it does:**
- Extracts text features from prompts (word count, sentence count)
- Identifies requirement keywords
- Estimates complexity scores
- Detects structural patterns (conditions, iterations)

**NLP Skills Shown:**
- Text preprocessing and tokenization
- Keyword extraction
- Readability metrics
- Pattern matching with regex

**Example:**
```python
prompt = "Write a function that checks if a string is a palindrome"
# Analyzes: word_count=10, complexity_score=15.2, has_conditions=True
```

### 2. **Code Feature Extraction** (`extract_code_features`)

**What it does:**
- Analyzes code as text to extract structural features
- Identifies programming patterns (loops, conditionals, functions)
- Measures code complexity through text analysis
- Estimates token counts

**NLP Skills Shown:**
- Pattern matching in structured text
- Feature extraction from code-as-text
- Text-based complexity metrics

### 3. **Semantic Similarity** (`compute_semantic_similarity`)

**What it does:**
- Compares generated code to reference solutions using TF-IDF
- Uses cosine similarity on character n-grams
- Measures semantic (not just syntactic) similarity

**NLP Skills Shown:**
- TF-IDF vectorization
- Cosine similarity
- Character n-gram analysis
- Text normalization

### 4. **Failure Mode Classification** (`classify_failure_mode_nlp`)

**What it does:**
- Uses NLP to classify error types from error messages
- Pattern matching on error text
- Text classification for failure modes

**NLP Skills Shown:**
- Text classification
- Pattern matching
- Error message parsing
- Multi-class classification

**Example:**
```python
error = "NameError: name 'x' is not defined"
# Classifies as: "name_error"
```

### 5. **Prompt-Code Alignment** (`analyze_prompt_code_alignment`)

**What it does:**
- Extracts requirements from natural language prompts
- Checks if generated code addresses those requirements
- Measures semantic alignment (not just syntax)

**NLP Skills Shown:**
- Requirement extraction from text
- Semantic matching
- Text-to-code alignment
- Coverage analysis

---

## Detailed Technique Explanations

### TF-IDF Vectorization

**TF-IDF** = **Term Frequency-Inverse Document Frequency**

Converts text into numbers (vectors) that represent how important words are.

**How It Works:**
1. **Term Frequency (TF)**: How often a word appears in a document
2. **Inverse Document Frequency (IDF)**: How rare/common a word is across all documents
3. **TF-IDF Score**: TF × IDF
   - Common words (like "def", "return") get lower scores
   - Rare, meaningful words get higher scores

**Example:**
```python
Code 1: "def fizzbuzz(n): return 'Fizz'"
Code 2: "def is_palindrome(s): return s == s[::-1]"

# TF-IDF converts these to vectors:
Code 1: [0.5, 0.8, 0.3, 0.0, 0.0, ...]  # "fizzbuzz" has high score
Code 2: [0.5, 0.0, 0.0, 0.9, 0.7, ...]  # "palindrome" has high score
```

**Why It Matters for Code:**
- Captures which functions/concepts are unique to each code snippet
- Ignores common Python keywords (def, return, if) that appear everywhere
- Focuses on the meaningful, task-specific parts

### Cosine Similarity

Measures how similar two vectors are, regardless of their length.

**How It Works:**
1. Convert code to vectors (using TF-IDF)
2. Calculate the angle between the vectors
3. Cosine of that angle = similarity score (0 to 1)
   - **1.0** = Identical
   - **0.8** = Very similar
   - **0.5** = Somewhat similar
   - **0.0** = Completely different

**Example:**
```python
Reference: "def is_palindrome(s): return s == s[::-1]"
Generated: "def is_palindrome(s): s_clean = s.lower(); return s_clean == s_clean[::-1]"

# Even though lengths differ, they have similar patterns:
# - Both have "is_palindrome"
# - Both use "==" and "[::-1]"
# - Both return boolean
# Cosine similarity = 0.85 (high similarity!)
```

**Why It Matters for Code:**
- Measures semantic similarity, not just exact match
- Finds code that does the same thing, even if written differently
- Better than edit distance for understanding "does this code solve the same problem?"

### Character N-gram Analysis

**N-gram** = A sequence of N consecutive characters

**Character n-gram** = Breaking text into overlapping chunks of N characters

**Example:**
```python
Code: "def palindrome"
# 3-grams: ["def", "ef ", "f p", " pa", "pal", "ali", ...]
# Captures patterns like "def ", "palindrome", etc.
```

**Why Character N-grams for Code?**
1. **Language-agnostic**: Works for any programming language
2. **Captures patterns**: Finds common code patterns (like "def ", "return", "==")
3. **Robust to formatting**: "def f(x):" and "def f(x):" have similar n-grams
4. **Finds similarities**: Even if variable names differ, structure is captured

### Text Normalization

Converting text to a standard form for comparison.

**Common Steps:**
1. **Lowercasing**: "DEF" → "def"
2. **Whitespace normalization**: "def  f(x):" → "def f(x):"
3. **Removing comments**: "# This is a comment" → removed
4. **Removing extra spaces**: "def  f(x) :" → "def f(x):"

**Example:**
```python
Before: "def IsPalindrome(s):\n    # comment\n    return s == s[::-1]"
After:  "def ispalindrome(s): return s == s[::-1]"
```

**Why It Matters:**
- Ensures "def F(x):" and "def f(x):" are treated similarly
- Comments don't affect similarity
- Whitespace differences are ignored

---

## How They Work Together

Complete pipeline:

```
1. Original Code:
   "def is_palindrome(s):\n    return s == s[::-1]"

2. Text Normalization:
   "def is_palindrome(s): return s == s[::-1]"

3. Character N-gram Extraction:
   ["def", "ef ", "f i", " is", "isp", "spa", "pal", ...]

4. TF-IDF Vectorization:
   [0.3, 0.5, 0.2, 0.8, 0.1, ...]  (vector of numbers)

5. Cosine Similarity:
   Compare vector A to vector B → 0.85 (85% similar)
```

---

## How to Run NLP Analysis

```bash
# After running code generation and evaluation
python app.py generate
python app.py evaluate

# Run NLP analysis
python app.py nlp
```

**Output includes:**
- Prompt complexity analysis
- Code feature extraction
- Requirement coverage metrics
- Failure mode classification
- Correlation analysis

---

## NLP Skills Relevant to Codex DS Role

### ✅ **Text Analysis**
- Analyzing prompts to understand complexity
- Extracting features from code as text
- Understanding error messages

### ✅ **Semantic Understanding**
- Measuring code-to-prompt alignment
- Understanding requirements from natural language
- Semantic similarity beyond syntax

### ✅ **Classification**
- Classifying failure modes from text
- Categorizing code patterns
- Text-based quality assessment

### ✅ **Feature Extraction**
- Extracting meaningful features from text
- Token analysis
- Pattern recognition

### ✅ **Code-to-Text Understanding**
- Understanding how code matches natural language intent
- Requirement extraction and validation
- Semantic correctness measurement

---

## Production Enhancements

In a real Codex DS role, you'd enhance this with:

1. **Code Embeddings**: Use CodeBERT, GraphCodeBERT, or similar
2. **AST Analysis**: Parse code into Abstract Syntax Trees
3. **Execution Traces**: Compare execution behavior, not just text
4. **Prompt Engineering Analysis**: A/B test different prompt formulations
5. **Token-Level Analysis**: Analyze model behavior at token level
6. **Multi-Language Support**: Extend NLP analysis to multiple languages
7. **Code Quality Metrics**: Style, complexity, maintainability from text

---

## Connection to Other Analyses

The NLP analysis complements:

- **Code Evaluation**: Adds semantic understanding beyond correctness
- **A/B Testing**: Can test prompt variations
- **Failure Diagnostics**: Text-based failure classification
- **Productivity Metrics**: Understanding prompt effectiveness

---

## Summary

This repository demonstrates NLP skills through:

1. ✅ **Text Analysis**: Prompt and code feature extraction
2. ✅ **Semantic Similarity**: TF-IDF-based code comparison
3. ✅ **Text Classification**: Failure mode classification
4. ✅ **Requirement Extraction**: Natural language to code alignment
5. ✅ **Pattern Matching**: Code structure analysis through text

These skills are essential for a Data Scientist working on AI code generation, where understanding the relationship between natural language (prompts) and code (outputs) is central to the role.

