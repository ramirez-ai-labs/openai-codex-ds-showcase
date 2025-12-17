"""
Codex DS Showcase: Causal Inference Analysis
============================================

This module demonstrates **causal inference**, a key skill for a Data Scientist
working on AI developer tools at OpenAI.

Why causal inference matters:
-----------------------------
In real telemetry data, developers who use an improved model (e.g., model_v2)
may already differ from others:
- They may work on different types of files
- They may do harder tasks
- They may be more advanced users
- They may type faster or use AI differently

If we only look at correlations, we may draw the wrong conclusions.

Causal inference helps us answer questions like:
  "Did the AI model *cause* higher acceptance rates?"
  "Does lower latency *cause* more developer satisfaction?"
  "What would have happened if the developer had used a different model?"

Techniques demonstrated here:
- Propensity Score Matching (PSM)
- Regression Adjustment
- Latency causal effect estimation
- Bootstrap confidence intervals

These are used by real-world DS teams to reason about model impact.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
from pathlib import Path
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# LOAD TELEMETRY DATA
# ============================================================

def load_telemetry_data(csv_path: str = None) -> pd.DataFrame:
    """
    Load developer telemetry data.

    Beginners:
    ----------
    This dataset contains rows like:
      - model_version
      - latency_ms
      - suggestion_length
      - accepted (0/1)
      - user_segment
      - language
      ...

    These are used to measure causal effects such as:
      “Does model_v2 increase acceptance?”
    """
    if csv_path is None:
        root = Path(__file__).parent.parent
        csv_path = root / "developer-telemetry-simulation" / "telemetry_events.csv"
    
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Telemetry data not found at {csv_path}")
    
    return pd.read_csv(csv_path)


# ============================================================
# PROPENSITY SCORE MATCHING (PSM)
# ============================================================

def estimate_ate_propensity_score_matching(
    df: pd.DataFrame,
    treatment_col: str = "model_version",
    treatment_value: str = "model_v2",
    outcome_col: str = "accepted",
    covariates: list = None
) -> Dict:
    """
    Estimate ATE (Average Treatment Effect) using Propensity Score Matching.

    Beginner explanation:
    ---------------------
    ATE tells us:
       "On average, how much did the treatment change the outcome?"

    Example:
       treatment = using model_v2
       outcome   = suggestion accepted (0/1)

    Why matching?
    -------------
    If advanced developers use model_v2 more often,
    then model_v2 might *look* better simply because advanced users accept suggestions more.

    Propensity Score Matching solves this by:
      1. Estimating the probability of receiving the treatment
         (propensity score)
      2. Matching treated users with similar untreated users
         → balances the comparison
    """
    if covariates is None:
        covariates = ["latency_ms", "suggestion_length", "language", "user_segment"]
    
    df_clean = df.copy()
    df_clean["treatment"] = (df_clean[treatment_col] == treatment_value).astype(int)

    # One-hot encode categorical columns
    df_encoded = pd.get_dummies(df_clean, columns=["language", "user_segment"], drop_first=True)

    # Build covariate list
    numeric_covariates = [
        c for c in covariates
        if c in df_encoded.columns and df_encoded[c].dtype in [np.int64, np.float64]
    ]

    all_covariates = (
        numeric_covariates +
        [c for c in df_encoded.columns if c.startswith("language_")] +
        [c for c in df_encoded.columns if c.startswith("user_segment_")]
    )

    # ------------------------------------------------------------
    # STEP 1: Estimate propensity score = probability of treatment
    # ------------------------------------------------------------
    X_ps = df_encoded[all_covariates].fillna(0)
    y_ps = df_encoded["treatment"]

    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(X_ps, y_ps)

    df_encoded["propensity_score"] = ps_model.predict_proba(X_ps)[:, 1]

    # ------------------------------------------------------------
    # STEP 2: Pair treated and control units with similar scores
    # ------------------------------------------------------------
    treated = df_encoded[df_encoded["treatment"] == 1].copy()
    control = df_encoded[df_encoded["treatment"] == 0].copy()

    matched_pairs = []
    control_used = set()

    for idx, t_row in treated.iterrows():
        ps_t = t_row["propensity_score"]

        # Find nearest available control
        distances = np.abs(control["propensity_score"] - ps_t)
        available = control[~control.index.isin(control_used)]

        if len(available) > 0:
            closest_idx = available.loc[distances[available.index].idxmin()].name
            matched_pairs.append((idx, closest_idx))
            control_used.add(closest_idx)

    # ------------------------------------------------------------
    # STEP 3: Compute ATE on matched sample
    # ------------------------------------------------------------
    if len(matched_pairs) == 0:
        return {"method": "PSM", "error": "No matched pairs found"}

    treated_outcomes = [df_encoded.loc[t_idx, outcome_col] for t_idx, _ in matched_pairs]
    control_outcomes = [df_encoded.loc[c_idx, outcome_col] for _, c_idx in matched_pairs]

    ate = np.mean(treated_outcomes) - np.mean(control_outcomes)

    # Bootstrap CI for stability
    bootstrap_ates = []
    for _ in range(1000):
        idxs = np.random.choice(len(matched_pairs), len(matched_pairs), replace=True)
        boot_t = [treated_outcomes[i] for i in idxs]
        boot_c = [control_outcomes[i] for i in idxs]
        bootstrap_ates.append(np.mean(boot_t) - np.mean(boot_c))

    ci_lower = np.percentile(bootstrap_ates, 2.5)
    ci_upper = np.percentile(bootstrap_ates, 97.5)

    return {
        "method": "Propensity Score Matching",
        "ate": ate,
        "ate_percentage": ate * 100,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "n_matched_pairs": len(matched_pairs),
        "interpretation": f"Using {treatment_value} increases {outcome_col} by {ate:.1%} (95% CI: {ci_lower:.1%} → {ci_upper:.1%})"
    }


# ============================================================
# REGRESSION ADJUSTMENT
# ============================================================

def regression_adjustment(
    df: pd.DataFrame,
    treatment_col="model_version",
    treatment_value="model_v2",
    outcome_col="accepted",
    covariates=None
) -> Dict:
    """
    Estimate ATE using regression adjustment.

    Beginner explanation:
    ---------------------
    Here we fit a regression model:
       outcome = treatment + covariates

    The coefficient on "treatment" tells us:
       “How much does the treatment change the probability of acceptance,
        holding everything else constant?”

    This is a widely used causal inference technique in industry.
    """
    if covariates is None:
        covariates = ["latency_ms", "suggestion_length", "language", "user_segment"]
    
    df_clean = df.copy()
    df_clean["treatment"] = (df_clean[treatment_col] == treatment_value).astype(int)

    df_encoded = pd.get_dummies(df_clean, columns=["language", "user_segment"], drop_first=True)

    numeric_covariates = [
        c for c in covariates
        if c in df_encoded.columns and df_encoded[c].dtype in [np.int64, np.float64]
    ]

    all_features = (
        ["treatment"] +
        numeric_covariates +
        [c for c in df_encoded.columns if c.startswith("language_")] +
        [c for c in df_encoded.columns if c.startswith("user_segment_")]
    )

    X = df_encoded[all_features].fillna(0)
    y = df_encoded[outcome_col]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Predict outcomes for treated vs control worlds
    X_treated = X.copy()
    X_treated["treatment"] = 1

    X_control = X.copy()
    X_control["treatment"] = 0

    y_treated = model.predict_proba(X_treated)[:, 1]
    y_control = model.predict_proba(X_control)[:, 1]

    ate = np.mean(y_treated - y_control)

    # Bootstrap CI
    boot_ates = []
    for _ in range(1000):
        idxs = np.random.choice(len(X), len(X), replace=True)
        X_b = X.iloc[idxs]
        y_b = y.iloc[idxs]

        m_b = LogisticRegression(max_iter=1000).fit(X_b, y_b)

        Xt_b = X_b.copy(); Xt_b["treatment"] = 1
        Xc_b = X_b.copy(); Xc_b["treatment"] = 0

        boot_ates.append(
            np.mean(
                m_b.predict_proba(Xt_b)[:, 1]
                - m_b.predict_proba(Xc_b)[:, 1]
            )
        )

    ci_lower = np.percentile(boot_ates, 2.5)
    ci_upper = np.percentile(boot_ates, 97.5)

    return {
        "method": "Regression Adjustment",
        "ate": ate,
        "ate_percentage": ate * 100,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "treatment_coefficient": model.coef_[0][0],
        "interpretation": f"Treatment increases {outcome_col} by {ate:.1%} (95% CI: {ci_lower:.1%} → {ci_upper:.1%})"
    }


# ============================================================
# LATENCY IMPACT ANALYSIS
# ============================================================

def analyze_latency_impact(df: pd.DataFrame) -> Dict:
    """
    Estimate causal effect of latency on acceptance.

    Beginner explanation:
    ---------------------
    Latency is continuous, not binary (unlike model version).
    So we ask:
        "How does acceptance probability change when latency increases?"

    Logistic regression gives a coefficient β for latency.
    We convert β into a more intuitive metric:
        → Change in acceptance probability per 100 ms
    """
    df_clean = df.copy()
    df_encoded = pd.get_dummies(
        df_clean, 
        columns=["language", "user_segment", "model_version"], 
        drop_first=True
    )

    features = ["latency_ms", "suggestion_length"] + [
        c for c in df_encoded.columns
        if c.startswith(("language_", "user_segment_", "model_version_"))
    ]

    X = df_encoded[features].fillna(0)
    y = df_encoded["accepted"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    latency_coef = model.coef_[0][X.columns.get_loc("latency_ms")]

    # Marginal effect for logistic regression
    p = y.mean()
    marginal_effect = latency_coef * p * (1 - p)

    return {
        "latency_coefficient": latency_coef,
        "marginal_effect_per_100ms": marginal_effect * 100,
        "interpretation": (
            f"Each 100 ms of extra latency decreases acceptance by "
            f"{abs(marginal_effect * 100):.1%}."
        ),
    }


# ============================================================
# OUTPUT RESULTS
# ============================================================

def print_causal_results(results: Dict):
    """Pretty-print causal inference findings."""
    print("\n" + "="*70)
    print("CAUSAL INFERENCE ANALYSIS")
    print("="*70)

    if "propensity_score" in results:
        r = results["propensity_score"]
        print("\n📊 PROPENSITY SCORE MATCHING")
        print("-"*70)
        for k,v in r.items():
            print(f"{k}: {v}")

    if "regression_adjustment" in results:
        r = results["regression_adjustment"]
        print("\n📈 REGRESSION ADJUSTMENT")
        print("-"*70)
        for k,v in r.items():
            print(f"{k}: {v}")

    if "latency_impact" in results:
        r = results["latency_impact"]
        print("\n⏱ LATENCY IMPACT (Continuous Treatment)")
        print("-"*70)
        for k,v in r.items():
            print(f"{k}: {v}")

    print("\n" + "="*70)
    print("KEY TAKEAWAYS FOR BEGINNERS")
    print("="*70)
    print("1. Correlation is NOT causation.")
    print("2. Propensity matching creates fair comparisons.")
    print("3. Regression controls for confounders.")
    print("4. Latency can be treated as a continuous 'treatment'.")
    print("5. Confidence intervals quantify uncertainty.")
    print("="*70 + "\n")


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    """Run causal inference workflows end-to-end."""
    try:
        df = load_telemetry_data()

        results = {}
        print("Running Propensity Score Matching...")
        results["propensity_score"] = estimate_ate_propensity_score_matching(df)

        print("Running Regression Adjustment...")
        results["regression_adjustment"] = regression_adjustment(df)

        print("Estimating Latency Impact...")
        results["latency_impact"] = analyze_latency_impact(df)

        print_causal_results(results)

    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\nRun telemetry simulation first:\n   python app.py simulate")


if __name__ == "__main__":
    main()
