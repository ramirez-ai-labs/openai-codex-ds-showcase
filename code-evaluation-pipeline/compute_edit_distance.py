# path: code-evaluation-pipeline/compute_edit_distance.py
"""
Levenshtein Edit Distance Calculator

This module calculates the "edit distance" between two strings of code.
Edit distance = minimum number of single-character edits (insertions, deletions, 
or substitutions) needed to transform one string into another.

Used to measure how far generated code is from a reference solution.
This is a proxy for "amount of refactoring a human might need to do."

Example:
    Reference: "def fizzbuzz(n): return 'Fizz'"
    Generated: "def fizzbuzz(n): return 'Buzz'"
    Edit distance: 4 (need to change 'Fizz' to 'Buzz' = 4 character changes)

Lower distance = code is closer to reference = less editing needed = better!
"""


def edit_distance(a: str, b: str) -> int:
    """
    Calculate the Levenshtein edit distance between two strings.
    
    This uses dynamic programming - a technique that solves complex problems
    by breaking them into smaller subproblems and storing results to avoid
    recalculating the same thing multiple times.
    
    Algorithm Overview:
    1. Create a table (matrix) where each cell represents the edit distance
       between substrings
    2. Fill the table from top-left to bottom-right
    3. Each cell considers three operations: insert, delete, or substitute
    4. The bottom-right cell contains the final answer
    
    Parameters:
        a: First string (e.g., reference code)
        b: Second string (e.g., generated code)
    
    Returns:
        Integer representing minimum number of edits needed
    """
    
    # ========================================================================
    # STEP 1: SET UP THE PROBLEM
    # ========================================================================
    # Get the lengths of both strings
    n = len(a)  # Length of string 'a' (reference code)
    m = len(b)  # Length of string 'b' (generated code)
    
    # ========================================================================
    # STEP 2: CREATE THE DYNAMIC PROGRAMMING TABLE
    # ========================================================================
    # Create a 2D table (matrix) with (n+1) rows and (m+1) columns
    # We use +1 because we need an extra row/column for empty strings
    # dp[i][j] = edit distance between first i characters of 'a' 
    #            and first j characters of 'b'
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    # ========================================================================
    # STEP 3: INITIALIZE BASE CASES
    # ========================================================================
    # Base case 1: To transform empty string to string of length i,
    #              we need i insertions (one character at a time)
    for i in range(n + 1):
        dp[i][0] = i
        # Example: "" -> "abc" needs 3 insertions
    
    # Base case 2: To transform string of length j to empty string,
    #              we need j deletions (one character at a time)
    for j in range(m + 1):
        dp[0][j] = j
        # Example: "abc" -> "" needs 3 deletions
    
    # ========================================================================
    # STEP 4: FILL THE TABLE USING DYNAMIC PROGRAMMING
    # ========================================================================
    # For each position in string 'a' (i) and string 'b' (j),
    # calculate the minimum edit distance
    for i in range(1, n + 1):      # Loop through each character in string 'a'
        for j in range(1, m + 1):  # Loop through each character in string 'b'
            
            # ================================================================
            # STEP 4a: CHECK IF CHARACTERS MATCH
            # ================================================================
            # If characters at current positions match, no substitution needed (cost = 0)
            # If they don't match, we need to substitute (cost = 1)
            # Note: We use i-1 and j-1 because dp table is 1-indexed but strings are 0-indexed
            if a[i - 1] == b[j - 1]:
                cost = 0  # Characters match - no substitution needed!
            else:
                cost = 1  # Characters differ - need to substitute
            
            # ================================================================
            # STEP 4b: CALCULATE MINIMUM OF THREE POSSIBLE OPERATIONS
            # ================================================================
            # To transform a[0..i-1] into b[0..j-1], we have three options:
            # 
            # Option 1: DELETE a[i-1]
            #   - Transform a[0..i-2] into b[0..j-1], then delete a[i-1]
            #   - Cost = dp[i-1][j] + 1 (previous result + 1 deletion)
            #
            # Option 2: INSERT b[j-1]
            #   - Transform a[0..i-1] into b[0..j-2], then insert b[j-1]
            #   - Cost = dp[i][j-1] + 1 (previous result + 1 insertion)
            #
            # Option 3: SUBSTITUTE (or keep if same)
            #   - Transform a[0..i-2] into b[0..j-2], then substitute a[i-1] with b[j-1]
            #   - Cost = dp[i-1][j-1] + cost (previous result + substitution cost)
            #
            # We pick the option with minimum cost (greedy approach)
            dp[i][j] = min(
                dp[i - 1][j] + 1,           # Option 1: Delete from 'a'
                dp[i][j - 1] + 1,           # Option 2: Insert into 'a'
                dp[i - 1][j - 1] + cost,    # Option 3: Substitute (or match)
            )
    
    # ========================================================================
    # STEP 5: RETURN THE RESULT
    # ========================================================================
    # The bottom-right cell (dp[n][m]) contains the edit distance between
    # the entire string 'a' and the entire string 'b'
    # This is our final answer!
    return dp[n][m]


# ============================================================================
# VISUAL EXAMPLE (for understanding)
# ============================================================================
"""
Example: edit_distance("cat", "bat")

Step-by-step table filling:

Initial table:
        ""  b   a   t
    ""   0   1   2   3
    c    1   ?   ?   ?
    a    2   ?   ?   ?
    t    3   ?   ?   ?

After filling:
        ""  b   a   t
    ""   0   1   2   3
    c    1   1   2   3
    a    2   2   1   2
    t    3   3   2   1

Explanation:
- dp[0][0] = 0: "" to "" needs 0 edits
- dp[1][1] = 1: "c" to "b" needs 1 substitution
- dp[2][2] = 1: "ca" to "ba" needs 1 substitution (c->b, a matches)
- dp[3][3] = 1: "cat" to "bat" needs 1 substitution (c->b, a matches, t matches)

Final answer: 1 (only need to change 'c' to 'b')
"""
