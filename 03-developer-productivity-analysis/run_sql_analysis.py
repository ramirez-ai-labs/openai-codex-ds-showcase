"""
Codex DS Showcase: SQL Analysis Runner
=======================================

This script demonstrates SQL fluency by running analytical queries
on telemetry data. It uses SQLAlchemy to connect to a SQLite database
created from the telemetry CSV.

Usage:
    python run_sql_analysis.py

This shows how a Codex Data Scientist would:
- Load telemetry data into a queryable format
- Run complex SQL queries for segmentation and cohort analysis
- Present results in a clear, actionable format
"""

# Import necessary libraries
import pandas as pd              # For data manipulation and analysis
import sqlite3                   # For working with SQLite databases
from pathlib import Path         # For handling file paths
from sqlalchemy import create_engine, text  # For database connections and SQL execution
import sys                       # For system-specific functions


def create_database_from_csv(csv_path: str, db_path: str = "telemetry.db"):
    """
    Create a SQLite database from a CSV file containing telemetry data.
    
    Args:
        csv_path (str): Path to the input CSV file
        db_path (str, optional): Path where the SQLite database will be created. Defaults to "telemetry.db"
        
    Returns:
        str: Path to the created database file
    """
    print(f"📊 Loading telemetry data from {csv_path}...")
    
    # Read the CSV file into a pandas DataFrame
    # A DataFrame is like a spreadsheet or SQL table in memory
    df = pd.read_csv(csv_path)
    
    # Create a connection to a SQLite database
    # If the database doesn't exist, it will be created automatically
    conn = sqlite3.connect(db_path)
    
    # Save the DataFrame to the SQLite database as a table named 'telemetry_events'
    # if_exists='replace' means it will overwrite the table if it already exists
    # index=False means we don't want to write row indices to the database
    df.to_sql("telemetry_events", conn, if_exists="replace", index=False)
    
    # Close the database connection to free up resources
    conn.close()
    
    # Print a success message with the number of rows processed
    print(f"✅ Created database with {len(df)} rows at {db_path}")
    
    # Return the path to the created database file
    return db_path


def run_query(engine, query_name: str, query: str):
    """Run a SQL query and display results."""
    print(f"\n{'='*60}")
    print(f"Query: {query_name}")
    print(f"{'='*60}")
    
    with engine.connect() as conn:
        result = conn.execute(text(query))
        rows = result.fetchall()
        columns = result.keys()
        
        if rows:
            df = pd.DataFrame(rows, columns=columns)
            print(df.to_string(index=False))
            print(f"\nRows returned: {len(df)}")
        else:
            print("No results returned.")


def main():
    # Paths
    root = Path(__file__).parent.parent
    csv_path = root / "developer-telemetry-simulation" / "telemetry_events.csv"
    db_path = root / "developer-productivity-analysis" / "telemetry.db"
    
    if not csv_path.exists():
        print(f"❌ Error: {csv_path} not found.")
        print("   Run the telemetry simulation first:")
        print("   python app.py simulate")
        sys.exit(1)
    
    # Create database
    create_database_from_csv(str(csv_path), str(db_path))
    
    # Create SQLAlchemy engine
    engine = create_engine(f"sqlite:///{db_path}")
    
    # Run key queries
    queries = {
        "Overall Acceptance Rate": """
            SELECT 
                COUNT(*) as total_suggestions,
                SUM(CASE WHEN accepted = 1 THEN 1 ELSE 0 END) as accepted_count,
                ROUND(AVG(CASE WHEN accepted = 1 THEN 1.0 ELSE 0.0 END), 3) as acceptance_rate
            FROM telemetry_events;
        """,
        
        "Acceptance Rate by Model Version (A/B Test)": """
            SELECT 
                model_version,
                COUNT(*) as total_suggestions,
                ROUND(AVG(CASE WHEN accepted = 1 THEN 1.0 ELSE 0.0 END), 3) as acceptance_rate,
                ROUND(AVG(latency_ms), 0) as avg_latency_ms,
                ROUND(AVG(CASE WHEN compile_success = 1 THEN 1.0 ELSE 0.0 END), 3) as compile_success_rate,
                ROUND(AVG(CASE WHEN test_pass = 1 THEN 1.0 ELSE 0.0 END), 3) as test_pass_rate
            FROM telemetry_events
            GROUP BY model_version
            ORDER BY model_version;
        """,
        
        "Acceptance Rate by User Segment": """
            SELECT 
                user_segment,
                COUNT(*) as total_suggestions,
                ROUND(AVG(CASE WHEN accepted = 1 THEN 1.0 ELSE 0.0 END), 3) as acceptance_rate,
                ROUND(AVG(latency_ms), 0) as avg_latency_ms
            FROM telemetry_events
            GROUP BY user_segment
            ORDER BY acceptance_rate DESC;
        """,
        
        "Acceptance Rate by Language": """
            SELECT 
                language,
                COUNT(*) as total_suggestions,
                ROUND(AVG(CASE WHEN accepted = 1 THEN 1.0 ELSE 0.0 END), 3) as acceptance_rate,
                ROUND(AVG(CASE WHEN compile_success = 1 THEN 1.0 ELSE 0.0 END), 3) as compile_success_rate
            FROM telemetry_events
            GROUP BY language
            ORDER BY acceptance_rate DESC;
        """,
        
        "Session Productivity Metrics": """
            SELECT 
                user_segment,
                COUNT(DISTINCT session_id) as total_sessions,
                ROUND(AVG(session_suggestions), 1) as avg_suggestions_per_session,
                ROUND(AVG(session_acceptance_rate), 3) as avg_session_acceptance_rate,
                ROUND(AVG(task_completed), 3) as task_completion_rate
            FROM (
                SELECT 
                    session_id,
                    user_segment,
                    COUNT(*) as session_suggestions,
                    AVG(CASE WHEN accepted = 1 THEN 1.0 ELSE 0.0 END) as session_acceptance_rate,
                    MAX(CASE WHEN test_pass = 1 THEN 1 ELSE 0 END) as task_completed
                FROM telemetry_events
                GROUP BY session_id, user_segment
            ) session_metrics
            GROUP BY user_segment;
        """,
        
        "Error Type Distribution": """
            SELECT 
                error_type,
                COUNT(*) as error_count,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM telemetry_events), 2) as error_percentage
            FROM telemetry_events
            WHERE error_type != 'none'
            GROUP BY error_type
            ORDER BY error_count DESC;
        """,
        
        "Hallucination Analysis": """
            SELECT 
                model_version,
                COUNT(*) as total_suggestions,
                SUM(CASE WHEN hallucination_flag = 1 THEN 1 ELSE 0 END) as hallucination_count,
                ROUND(AVG(CASE WHEN hallucination_flag = 1 THEN 1.0 ELSE 0.0 END), 3) as hallucination_rate
            FROM telemetry_events
            GROUP BY model_version;
        """
    }
    
    print("\n" + "="*60)
    print("SQL Analysis Results")
    print("="*60)
    
    for query_name, query in queries.items():
        run_query(engine, query_name, query)
    
    print("\n" + "="*60)
    print("✅ SQL analysis complete!")
    print("="*60)
    print("\nFor more queries, see sql_queries.sql")


if __name__ == "__main__":
    main()

