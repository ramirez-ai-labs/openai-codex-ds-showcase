# path: developer-telemetry-simulation/simulate_telemetry.py
"""
Simulate telemetry for an AI coding assistant.

This script creates a synthetic dataset of IDE-like events such as:
- suggestion acceptance
- edits
- compile/test results
- latency
- model versions

Goal:
- Mimic the type of data the Codex DS team might see
- Provide a rich playground for analysis notebooks

Usage:
    python simulate_telemetry.py

Output:
    telemetry_events.csv
"""

# Import necessary libraries
import numpy as np  # For numerical operations and random number generation
import pandas as pd  # For creating and working with data tables (DataFrames)
from datetime import datetime, timedelta  # For working with dates and times
import random  # For random choices and selections
import argparse  # For handling command-line arguments
from pathlib import Path  # For working with file paths

# ============================================================================
# CONSTANTS: Fixed values that define the simulation parameters
# ============================================================================

# Different AI model versions we're testing (v2 is "better" than v1)
MODEL_VERSIONS = ["model_v1", "model_v2"]

# Programming languages developers might be using
LANGUAGES = ["python", "javascript", "typescript", "go"]

# Types of errors that can occur in code
ERROR_TYPES = ["syntax_error", "runtime_error", "logical_bug", "none"]

# Developer experience levels (affects how they use AI suggestions)
USER_SEGMENTS = ["beginner", "intermediate", "expert"]


def simulate_sessions(
    n_sessions: int = 500,
    min_suggestions: int = 5,
    max_suggestions: int = 40,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate multiple coding sessions with AI suggestions.
    
    This function creates fake data that looks like real telemetry from developers
    using an AI coding assistant. Each session represents one coding session where
    a developer receives multiple AI suggestions.
    
    Parameters:
    - n_sessions: How many coding sessions to simulate (default: 500)
    - min_suggestions: Minimum AI suggestions per session (default: 5)
    - max_suggestions: Maximum AI suggestions per session (default: 40)
    - seed: Random seed for reproducibility (same seed = same results)
    
    Returns:
    - A pandas DataFrame with all the simulated telemetry events
    """
    
    # Set up random number generators (seed ensures reproducible results)
    rng = np.random.default_rng(seed)  # NumPy's random generator
    random.seed(seed)  # Python's random generator
    
    # List to store all the events we'll create
    rows = []
    
    # Start time for the simulation (all events will be relative to this)
    base_time = datetime.utcnow()

    # ========================================================================
    # OUTER LOOP: Create one coding session at a time
    # ========================================================================
    for session_id in range(1, n_sessions + 1):
        # Each session has a unique ID (1, 2, 3, ..., 500)
        
        # Randomly assign a developer (same developer can have multiple sessions)
        developer_id = rng.integers(1, 150)  # Pick a developer ID between 1-149
        
        # Randomly assign a coding task (what the developer is working on)
        task_id = rng.integers(1, 80)  # Pick a task ID between 1-79
        
        # Assign developer experience level (weighted: 40% beginner, 40% intermediate, 20% expert)
        # This affects how likely they are to accept AI suggestions
        user_segment = random.choices(USER_SEGMENTS, weights=[0.4, 0.4, 0.2])[0]
        
        # Randomly pick a programming language
        language = random.choice(LANGUAGES)

        # Randomly assign which AI model version this session uses
        # (We'll make v2 "better" than v1 to simulate A/B testing)
        model_version = random.choice(MODEL_VERSIONS)
        
        # How many AI suggestions will this session have? (random between min and max)
        n_suggestions = rng.integers(min_suggestions, max_suggestions + 1)
        
        # When did this session start? (stagger sessions by 3 minutes each)
        session_start = base_time + timedelta(minutes=session_id * 3)

        # ====================================================================
        # SET MODEL-SPECIFIC PROBABILITIES
        # Model v2 is "better" - this simulates an improved AI model
        # ====================================================================
        if model_version == "model_v1":
            # Older model: lower acceptance, more errors, slower
            base_accept_prob = 0.45      # 45% chance developer accepts suggestion
            base_compile_prob = 0.7      # 70% chance code compiles without errors
            base_test_prob = 0.6         # 60% chance tests pass (if it compiles)
            latency_mean = 850           # Average 850ms to generate suggestion
        else:  # model_v2
            # Newer model: higher acceptance, fewer errors, faster
            base_accept_prob = 0.58      # 58% chance developer accepts (better!)
            base_compile_prob = 0.8      # 80% chance code compiles (better!)
            base_test_prob = 0.72        # 72% chance tests pass (better!)
            latency_mean = 650           # Average 650ms (faster!)

        # ====================================================================
        # ADJUST ACCEPTANCE PROBABILITY BY USER EXPERIENCE LEVEL
        # Beginners accept more (trust AI more), experts accept less (more selective)
        # ====================================================================
        segment_boost = {
            "beginner": 0.05,      # +5% more likely to accept (trust AI more)
            "intermediate": 0.0,   # No adjustment (baseline)
            "expert": -0.05,       # -5% less likely to accept (more selective)
        }[user_segment]

        # ====================================================================
        # INNER LOOP: Create one AI suggestion event at a time
        # ====================================================================
        for suggestion_idx in range(1, n_suggestions + 1):
            # Each suggestion gets a unique ID within this session (1, 2, 3, ...)
            
            # ================================================================
            # TIMING: When did this suggestion appear?
            # ================================================================
            # Suggestions appear at random intervals (5-60 seconds apart)
            # This simulates real developer behavior (not constant rate)
            event_time = session_start + timedelta(
                seconds=int(suggestion_idx * rng.uniform(5, 60))
            )

            # ================================================================
            # LATENCY: How long did it take the AI to generate the suggestion?
            # ================================================================
            # Use normal distribution (bell curve) around the mean latency
            # Clamp to minimum 80ms (AI can't be instant)
            latency_ms = max(
                80, int(rng.normal(loc=latency_mean, scale=150))
            )  # scale=150 means standard deviation of 150ms

            # ================================================================
            # CODE SIZE: How long is the suggested code?
            # ================================================================
            # Random length between 5-40 lines (rough estimate)
            suggestion_length = rng.integers(5, 40)
            
            # Final code might be shorter (developer deleted parts) or longer (added edits)
            final_code_length = suggestion_length + rng.integers(-5, 20)

            # ================================================================
            # ACCEPTANCE: Did the developer accept this suggestion?
            # ================================================================
            # Combine base probability with user segment adjustment
            # Clip to ensure it stays between 5% and 95% (reasonable bounds)
            accept_prob = np.clip(base_accept_prob + segment_boost, 0.05, 0.95)
            
            # Roll the dice: random number between 0-1, if < probability, accept!
            accepted = rng.random() < accept_prob

            # ================================================================
            # EDITING BEHAVIOR: What did developer do with the suggestion?
            # ================================================================
            if accepted:
                # If accepted, 60% chance developer edits it afterward
                # (Real developers often tweak AI suggestions)
                edited = rng.random() < 0.6
                ignored = False  # Can't be ignored if accepted
            else:
                # If not accepted, developer either ignored it (70% chance)
                # or manually wrote code instead (30% chance)
                edited = False
                ignored = rng.random() < 0.7

            # ================================================================
            # CODE QUALITY: Does the code work?
            # ================================================================
            # First check: Does it compile/run without syntax errors?
            compile_success = rng.random() < base_compile_prob
            
            # Second check: If it compiles, do the tests pass?
            # (Only check tests if it compiled - can't test broken code!)
            test_pass = compile_success and (rng.random() < base_test_prob)

            # ================================================================
            # ERROR CLASSIFICATION: What went wrong (if anything)?
            # ================================================================
            if not compile_success:
                # Code doesn't compile - pick a syntax/runtime error type
                error_type = random.choice(ERROR_TYPES[:-1])  # Exclude "none"
            elif compile_success and not test_pass:
                # Code compiles but tests fail - it's a logical bug
                error_type = "logical_bug"
            else:
                # Everything works! No errors.
                error_type = "none"

            # ================================================================
            # HALLUCINATION DETECTION: Did AI generate nonsense code?
            # ================================================================
            # Hallucinations are more likely when:
            # - Code doesn't compile (suggests AI made up something)
            # - Code is long (more room for errors)
            # - Random chance (30% of failed long suggestions)
            hallucinated = (not compile_success) and suggestion_length > 25 and rng.random() < 0.3

            # ================================================================
            # STORE THE EVENT: Add all this information as one row of data
            # ================================================================
            rows.append(
                {
                    # Session and developer info
                    "session_id": session_id,              # Which coding session?
                    "developer_id": int(developer_id),      # Which developer?
                    "user_segment": user_segment,           # Beginner/intermediate/expert?
                    "task_id": int(task_id),                # What task are they working on?
                    "suggestion_id": suggestion_idx,         # Which suggestion in this session?
                    
                    # Timing
                    "timestamp": event_time.isoformat(),    # When did this happen? (ISO format)
                    
                    # Context
                    "language": language,                   # What programming language?
                    "model_version": model_version,         # Which AI model version?
                    
                    # Performance metrics
                    "latency_ms": latency_ms,               # How fast was the AI? (milliseconds)
                    "suggestion_length": int(suggestion_length),    # How long was the suggestion?
                    "final_code_length": int(final_code_length),    # How long after edits?
                    
                    # Developer behavior
                    "accepted": bool(accepted),             # Did developer accept it?
                    "edited_after_accept": bool(edited),    # Did they edit it after accepting?
                    "ignored": bool(ignored),               # Did they ignore it?
                    
                    # Code quality
                    "compile_success": bool(compile_success),  # Does it compile?
                    "test_pass": bool(test_pass),              # Do tests pass?
                    "error_type": error_type,                  # What type of error (if any)?
                    "hallucination_flag": bool(hallucinated),   # Was it nonsense code?
                }
            )

    # ========================================================================
    # CONVERT TO DATAFRAME: Transform our list of dictionaries into a table
    # ========================================================================
    # pandas DataFrame is like an Excel spreadsheet - rows and columns
    return pd.DataFrame(rows)


def main():
    """
    Main function: Entry point when script is run from command line.
    
    This function:
    1. Parses command-line arguments (how many sessions, where to save)
    2. Calls simulate_sessions() to generate the data
    3. Saves the data to a CSV file
    """
    
    # ========================================================================
    # SET UP COMMAND-LINE ARGUMENTS
    # ========================================================================
    # This allows users to run: python simulate_telemetry.py --sessions 1000
    parser = argparse.ArgumentParser(description="Generate synthetic telemetry data")
    parser.add_argument(
        "--sessions", 
        type=int, 
        default=500, 
        help="Number of coding sessions to simulate (default: 500)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None, 
        help="Where to save the CSV file (default: same folder as script)"
    )
    args = parser.parse_args()

    # ========================================================================
    # GENERATE THE DATA
    # ========================================================================
    # Call our simulation function with the number of sessions requested
    df = simulate_sessions(n_sessions=args.sessions)
    
    # ========================================================================
    # DETERMINE OUTPUT FILE PATH
    # ========================================================================
    if args.output is None:
        # If no path specified, save in the same folder as this script
        script_dir = Path(__file__).parent  # Get folder containing this script
        output_path = script_dir / "telemetry_events.csv"  # Add filename
    else:
        # Use the path the user specified
        output_path = Path(args.output)
    
    # ========================================================================
    # SAVE TO CSV FILE
    # ========================================================================
    # index=False means don't save the row numbers as a column
    df.to_csv(output_path, index=False)
    
    # Print confirmation message
    print(f"✅ Wrote {len(df)} telemetry rows to {output_path.resolve()}")


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================
# This runs main() only if the script is executed directly (not imported)
if __name__ == "__main__":
    main()
