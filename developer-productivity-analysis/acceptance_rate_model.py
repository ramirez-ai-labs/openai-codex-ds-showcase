# path: developer-productivity-analysis/acceptance_rate_model.py
"""
Simple acceptance-rate model for AI code suggestions.

Loads telemetry_events.csv and:
- Computes basic aggregates
- Fits a logistic regression to model acceptance probability
- Prints feature importances (coefficients)

Intended as a teaching example for:
- DS working on developer telemetry
- Connecting features like latency, model_version, user_segment to acceptance.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report


def load_data(csv_path: str = None) -> pd.DataFrame:
    if csv_path is None:
        root = Path(__file__).parent.parent
        csv_path = root / "developer-telemetry-simulation" / "telemetry_events.csv"
    return pd.read_csv(csv_path)


def main():
    root = Path(__file__).parent.parent
    csv_path = root / "developer-telemetry-simulation" / "telemetry_events.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run 'python app.py simulate' first to generate telemetry data."
        )

    df = load_data()

    # Basic aggregates
    print("=== Overall Acceptance Rate ===")
    print(df["accepted"].mean())

    print("\n=== Acceptance Rate by Model Version ===")
    print(df.groupby("model_version")["accepted"].mean())

    print("\n=== Acceptance Rate by User Segment ===")
    print(df.groupby("user_segment")["accepted"].mean())

    # Prepare features for a simple logistic regression
    feature_cols_num = ["latency_ms", "suggestion_length", "final_code_length"]
    feature_cols_cat = ["model_version", "user_segment", "language"]
    target_col = "accepted"

    X = df[feature_cols_num + feature_cols_cat]
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    numeric_transformer = "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_cols_num),
            ("cat", categorical_transformer, feature_cols_cat),
        ]
    )

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=3))

    print("\n=== ROC AUC ===")
    print(roc_auc_score(y_test, y_prob))

    # Show top coefficients (approximate feature importance)
    model: LogisticRegression = clf.named_steps["model"]
    ohe: OneHotEncoder = clf.named_steps["preprocess"].named_transformers_["cat"]

    cat_feature_names = ohe.get_feature_names_out(feature_cols_cat)
    all_feature_names = feature_cols_num + list(cat_feature_names)
    coefs = model.coef_[0]

    coef_df = (
        pd.DataFrame({"feature": all_feature_names, "coef": coefs})
        .sort_values("coef", ascending=False)
    )

    print("\n=== Top Positive Features (higher → more likely to accept) ===")
    print(coef_df.head(10))

    print("\n=== Top Negative Features (lower → less likely to accept) ===")
    print(coef_df.tail(10))


if __name__ == "__main__":
    main()
