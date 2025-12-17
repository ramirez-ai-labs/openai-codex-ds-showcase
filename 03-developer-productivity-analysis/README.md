# Developer Telemetry & Productivity Analysis — README
## What This Folder Is About

This folder demonstrates how AI-assisted coding systems are evaluated in practice, using developer telemetry data rather than just model accuracy.

Instead of asking “Is the model smart?”, these files help answer:

- Do developers accept AI suggestions?
- How much editing effort is required?
- Does lower latency or higher quality matter more?
- How do retries, failures, and hallucinations affect productivity?

This mirrors the kind of analysis done by Codex / AI productivity / developer experience teams.

##  Big Picture: The Analysis Pipeline

Think of this folder as a mini end-to-end analytics system:
```bash
Telemetry data
   ↓
Metrics & feature extraction
   ↓
Statistical / causal analysis
   ↓
Productivity insights

```
Each file contributes one piece of that pipeline

---

# File-by-File Guide
1. simulate_telemetry.py

Purpose: Create realistic fake developer telemetry data.

What it does:

- Simulates AI code suggestions
- Randomly assigns acceptance, latency, errors, retries, hallucinations
- Writes rows to a CSV file (telemetry_events.csv)

Why this exists:

- Real telemetry is private
- Simulation lets us reason about real systems safely
- Every downstream file depends on this data

📌 Start here if you want to understand where the data comes from.

---

2. telemetry_schema.md

Purpose: Define what each telemetry field means.

Examples:

- accepted: whether the developer kept the suggestion
- latency_ms: how long the suggestion took
- hallucination_flag: whether the suggestion was incorrect
- user_segment: novice vs experienced developer

Why this matters:

- Metrics are meaningless without definitions
- This mirrors real production telemetry specs

📌 Read this if you want to understand the data columns.

---

3. compute_edit_distance.py

Purpose: Measure how much effort a developer spent editing a suggestion.

What it does:

- Computes edit distance between:
    - AI suggestion
    - Final accepted code
- Lower distance = less work for the developer

Why this matters:
- Acceptance alone is not enough
- Edit distance approximates developer effort
- This metric is central to AI coding research

📌 This connects code quality to human effort.

---