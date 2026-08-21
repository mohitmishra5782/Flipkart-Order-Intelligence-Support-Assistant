import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

def main():
    print("=== PART 1: RETURN RISK MODEL TRAINING & ANALYSIS ===")

    if not os.path.exists("orders_dataset.csv"):
        raise FileNotFoundError("orders_dataset.csv not found.")

    df = pd.read_csv("orders_dataset.csv")
    print(f"Total Rows: {len(df)} | Overall Return Rate: {df['returned'].mean():.4f}")

    X = df.drop(columns=["order_id", "returned"])
    y = df["returned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    num_cols = ["price_inr", "discount_pct", "customer_tenure_days", "num_previous_orders", 
                "num_previous_returns", "delivery_distance_km", "delivery_days", 
                "is_weekend_order", "rating_given"]
    cat_cols = ["product_category", "payment_method"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols)
    ])

    rf_pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=42))
    ])

    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [6, 10, None]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(rf_pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_rf = grid.best_estimator_
    rf_probs = best_rf.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_probs)

    print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")
    print(f"Test Set ROC-AUC: {rf_auc:.4f}")

    thresholds = np.arange(0.1, 0.92, 0.02)
    best_rf_t = 0.5
    best_rf_f1 = 0.0
    for t in thresholds:
        preds = (rf_probs >= t).astype(int)
        f1 = f1_score(y_test, preds, zero_division=0)
        if f1 > best_rf_f1:
            best_rf_f1 = f1
            best_rf_t = round(t, 2)

    print(f"Random Forest Optimal Threshold: {best_rf_t} (F1 Score: {best_rf_f1:.4f})")

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_rf, "models/return_risk_model.pkl")
    print("Saved models/return_risk_model.pkl successfully!")

if __name__ == "__main__":
    main()
